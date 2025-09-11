import React, { useState, useRef } from 'react';
import { Upload, Download, FileText, BarChart3 } from 'lucide-react';
import { predictBatchTweets, TweetResponse } from '../services/api';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { motion } from 'framer-motion';

const BatchAnalysis: React.FC = () => {
  const { modelChoice } = useApi();
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<TweetResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
    } else {
      alert('Please select a valid CSV file');
    }
  };

  const parseCsv = (csvText: string): { text: string }[] => {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const textColumnIndex = headers.findIndex(h => h.toLowerCase() === 'text');
    
    if (textColumnIndex === -1) {
      throw new Error('CSV must contain a "text" column');
    }

    return lines.slice(1)
      .filter(line => line.trim())
      .map(line => {
        const columns = line.split(',');
        return {
          text: columns[textColumnIndex]?.replace(/"/g, '').trim() || ''
        };
      })
      .filter(row => row.text);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    
    try {
      const csvText = await file.text();
      const tweets = parseCsv(csvText);
      
      if (tweets.length === 0) {
        setError('No valid tweets found in the CSV file');
        return;
      }

      const response = await predictBatchTweets(
        tweets,
        modelChoice === 'Transformer'
      );

      if (response.success && response.data) {
        setResults(response.data.predictions);
      } else {
        setError(response.error || 'Failed to analyze batch tweets');
        setResults([]);
      }
    } catch (error) {
      console.error('Error analyzing batch:', error);
      setError('Error processing file: ' + (error as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = (format: 'csv' | 'txt') => {
    if (results.length === 0) return;

    let content = '';
    let filename = '';
    let mimeType = '';

    if (format === 'csv') {
      content = 'Text,Prediction,Confidence,AI Response\n' + 
        results.map(result => {
          const aiResponse = generateProactiveResponse(result.prediction);
          return `"${result.text}","${result.prediction}","${(result.confidence * 100).toFixed(2)}%","${aiResponse}"`;
        }).join('\n');
      filename = `safaricom_tweet_analysis_${new Date().toISOString().split('T')[0]}.csv`;
      mimeType = 'text/csv';
    } else {
      content = `Safaricom Tweet Analysis Report\nGenerated on: ${new Date().toLocaleString()}\n\n`;
      results.forEach((result, index) => {
        content += `--- Tweet ${index + 1} ---\n`;
        content += `Tweet: ${result.text}\n`;
        content += `Prediction: ${result.prediction} (Confidence: ${(result.confidence * 100).toFixed(2)}%)\n`;
        content += `AI Response: ${generateProactiveResponse(result.prediction)}\n\n`;
      });
      filename = `safaricom_tweet_analysis_report_${new Date().toISOString().split('T')[0]}.txt`;
      mimeType = 'text/plain';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const generateProactiveResponse = (prediction: string): string => {
    const responses: { [key: string]: string } = {
      'MPESA complaint': 'It looks like you\'re having an MPESA issue. We\'re sorry for the inconvenience. Please rest assured that your transaction is being reviewed and we\'ll get back to you shortly.',
      'Customer care complaint': 'Thank you for reaching out to Safaricom Care. A customer representative will assist you shortly.',
      'Network reliability problem': 'Our network is currently experiencing technical issues in some areas. Our technical team is working round the clock to restore full service.',
      'Data protection and privacy concern': 'Thank you for raising this concern. Safaricom takes data protection seriously and we are reviewing the matter.',
      'Internet or airtime bundle complaint': 'We acknowledge the reported internet bundles problem. Our team is looking to improve the data deals and coverage for ease of using internet bundles.',
      'Neutral': 'We\'re glad you\'re enjoying our services! Your positive feedback keeps us going. Thank you!',
      'Hate Speech': 'We are sorry if our services are not up to per with your expectations. We are working round the clock to provide reliable services.'
    };

    return responses[prediction] || responses['Neutral'];
  };

  // Prepare chart data
  const predictionCounts = results.reduce((acc, result) => {
    acc[result.prediction] = (acc[result.prediction] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const pieData = Object.entries(predictionCounts).map(([key, value]) => ({
    name: key,
    value,
    percentage: ((value / results.length) * 100).toFixed(1)
  }));

  const barData = Object.entries(predictionCounts).map(([key, value]) => ({
    prediction: key.length > 20 ? key.substring(0, 20) + '...' : key,
    count: value,
    percentage: ((value / results.length) * 100).toFixed(1)
  }));

  // Enhanced color palette
  const COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#AED6F1'
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  return (
    <motion.div 
      className="max-w-6xl mx-auto space-y-8 pb-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="text-center space-y-4" variants={itemVariants}>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          📋 Batch Tweet Analysis
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          Upload a CSV file containing tweets for bulk classification and download the results.
        </p>
      </motion.div>

      {/* Upload Section */}
      <motion.div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-8" variants={itemVariants}>
        <div className="space-y-6">
          <div 
            className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
              Upload CSV file with tweets
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              File should have a 'text' column containing tweet content
            </p>
            {file && (
              <div className="bg-blue-50 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-4 py-2 rounded-lg inline-block">
                📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </div>
            )}
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            className="hidden"
          />

          {file && (
            <div className="flex justify-center">
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-4 px-8 rounded-lg transition-colors duration-200 flex items-center space-x-2"
              >
                {loading ? (
                  <LoadingSpinner size="sm" className="border-white border-t-blue-200" />
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    <span>Analyze All Tweets</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </motion.div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
                <span className="text-red-600 dark:text-red-400 text-sm">⚠️</span>
              </div>
            </div>
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-1">
                Batch Analysis Failed
              </h4>
              <p className="text-red-700 dark:text-red-300 text-sm leading-relaxed">
                {error}
              </p>
              <button
                onClick={() => setError(null)}
                className="mt-3 text-sm text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200 underline"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {results.length > 0 && !error && (
        <motion.div 
          className="space-y-8"
          variants={itemVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Summary */}
          <div className="bg-green-50 dark:bg-green-900 border border-green-200 dark:border-green-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-green-800 dark:text-green-200 mb-2">
              ✅ Analysis Complete!
            </h3>
            <p className="text-green-700 dark:text-green-300">
              Successfully analyzed {results.length} tweets
            </p>
          </div>

          {/* Enhanced Charts with better visualizations */}
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Enhanced Pie Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-6 text-center flex items-center justify-center">
                <div className="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mr-2"></div>
                Prediction Distribution
              </h4>
              <ResponsiveContainer width="100%" height={400}>
                <PieChart>
                  <defs>
                    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#FF6B6B" />
                      <stop offset="100%" stopColor="#FF8E8E" />
                    </linearGradient>
                    <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#4ECDC4" />
                      <stop offset="100%" stopColor="#6EDDD6" />
                    </linearGradient>
                    <linearGradient id="gradient3" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#45B7D1" />
                      <stop offset="100%" stopColor="#67C6E3" />
                    </linearGradient>
                    <linearGradient id="gradient4" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#96CEB4" />
                      <stop offset="100%" stopColor="#A8D8C6" />
                    </linearGradient>
                    <linearGradient id="gradient5" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#FFEAA7" />
                      <stop offset="100%" stopColor="#FFEFB9" />
                    </linearGradient>
                    <linearGradient id="gradient6" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#DDA0DD" />
                      <stop offset="100%" stopColor="#E8B5E8" />
                    </linearGradient>
                    <linearGradient id="gradient7" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#98D8C8" />
                      <stop offset="100%" stopColor="#AAE1D3" />
                    </linearGradient>
                    <linearGradient id="gradient8" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#F7DC6F" />
                      <stop offset="100%" stopColor="#F9E481" />
                    </linearGradient>
                  </defs>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }: any) => `${name.length > 15 ? name.substring(0, 15) + '...' : name}: ${percentage}%`}
                    outerRadius={120}
                    innerRadius={40}
                    fill="#8884d8"
                    dataKey="value"
                    stroke="#fff"
                    strokeWidth={2}
                  >
                    {pieData.map((_, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={COLORS[index % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: any) => [value, 'Count']}
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              
              {/* Custom Legend */}
              <div className="mt-4 grid grid-cols-1 gap-2 max-h-32 overflow-y-auto">
                {pieData.map((entry, index) => (
                  <div key={entry.name} className="flex items-center justify-between text-sm">
                    <div className="flex items-center">
                      <div 
                        className="w-3 h-3 rounded-full mr-2 border border-white shadow-sm"
                        style={{ backgroundColor: COLORS[index % COLORS.length] }}
                      ></div>
                      <span className="text-gray-700 dark:text-gray-300 truncate max-w-32" title={entry.name}>
                        {entry.name}
                      </span>
                    </div>
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {entry.value} ({entry.percentage}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Enhanced Bar Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-6 text-center flex items-center justify-center">
                <div className="w-3 h-3 bg-gradient-to-r from-green-500 to-blue-500 rounded-full mr-2"></div>
                Classification Counts
              </h4>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <defs>
                    <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.9}/>
                      <stop offset="95%" stopColor="#7C3AED" stopOpacity={0.7}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} />
                  <XAxis 
                    dataKey="prediction" 
                    angle={-45} 
                    textAnchor="end" 
                    height={80}
                    fontSize={11}
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                  />
                  <YAxis 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                  />
                  <Tooltip 
                    formatter={(value: any) => [value, 'Count']}
                    labelFormatter={(label) => `Prediction: ${label}`}
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                  <Bar 
                    dataKey="count" 
                    fill="url(#barGradient)"
                    radius={[4, 4, 0, 0]}
                    stroke="#4F46E5"
                    strokeWidth={1}
                  />
                </BarChart>
              </ResponsiveContainer>
              
              {/* Bar Chart Summary */}
              <div className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="text-center text-sm text-gray-600 dark:text-gray-400">
                  <span className="font-semibold">Total Classifications:</span> {results.length}
                  <span className="mx-2">•</span>
                  <span className="font-semibold">Categories:</span> {Object.keys(predictionCounts).length}
                </div>
              </div>
            </div>
          </div>

          {/* Download Section */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📥 Download Results</h4>
            <div className="flex flex-wrap gap-4">
              <button
                onClick={() => downloadResults('csv')}
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200 flex items-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Download CSV</span>
              </button>
              <button
                onClick={() => downloadResults('txt')}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200 flex items-center space-x-2"
              >
                <FileText className="w-5 h-5" />
                <span>Download Report</span>
              </button>
            </div>
          </div>

          {/* Enhanced Results Table */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-700 dark:to-gray-600">
              <h4 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                <div className="w-4 h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded mr-2"></div>
                📋 Detailed Results
              </h4>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-600">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 dark:text-gray-200 uppercase tracking-wider border-r border-gray-200 dark:border-gray-600">
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        Tweet
                      </div>
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 dark:text-gray-200 uppercase tracking-wider border-r border-gray-200 dark:border-gray-600">
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        Prediction
                      </div>
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 dark:text-gray-200 uppercase tracking-wider border-r border-gray-200 dark:border-gray-600">
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></div>
                        Confidence
                      </div>
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-700 dark:text-gray-200 uppercase tracking-wider">
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
                        AI Response
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {results.slice(0, 100).map((result, index) => {
                    const confidenceColor = result.confidence > 0.8 ? 'text-green-600 dark:text-green-400' : 
                                           result.confidence > 0.6 ? 'text-yellow-600 dark:text-yellow-400' : 
                                           'text-red-600 dark:text-red-400';
                    const confidenceBg = result.confidence > 0.8 ? 'bg-green-100 dark:bg-green-900' : 
                                        result.confidence > 0.6 ? 'bg-yellow-100 dark:bg-yellow-900' : 
                                        'bg-red-100 dark:bg-red-900';
                    
                    return (
                      <tr key={index} className="hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 dark:hover:from-gray-700 dark:hover:to-gray-600 transition-all duration-200">
                        <td className="px-6 py-4 text-sm text-gray-900 dark:text-white max-w-xs border-r border-gray-100 dark:border-gray-700">
                          <div className="truncate" title={result.text}>
                            <span className="text-blue-800 dark:text-blue-300 font-medium">
                              {result.text}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm border-r border-gray-100 dark:border-gray-700">
                          <div className="flex items-center">
                            <div 
                              className="w-3 h-3 rounded-full mr-2 border border-white shadow-sm"
                              style={{ 
                                backgroundColor: COLORS[Object.keys(predictionCounts).indexOf(result.prediction) % COLORS.length] 
                              }}
                            ></div>
                            <span className="font-semibold text-gray-900 dark:text-white">
                              {result.prediction}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm border-r border-gray-100 dark:border-gray-700">
                          <div className="flex items-center">
                            <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${confidenceBg} ${confidenceColor}`}>
                              {(result.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-700 dark:text-gray-300 max-w-sm">
                          <div className="truncate bg-gray-50 dark:bg-gray-700 p-2 rounded-lg" title={generateProactiveResponse(result.prediction)}>
                            <span className="italic">
                              {generateProactiveResponse(result.prediction)}
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {results.length > 100 && (
              <div className="px-6 py-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 text-center">
                <div className="text-sm text-gray-600 dark:text-gray-400 flex items-center justify-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                  <span className="font-semibold">Showing first 100 results</span>
                  <span className="mx-2">•</span>
                  <span>Download full results using the buttons above</span>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default BatchAnalysis;