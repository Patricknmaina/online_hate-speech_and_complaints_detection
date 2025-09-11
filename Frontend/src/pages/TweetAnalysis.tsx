import React, { useState } from 'react';
import { Search, MessageSquare } from 'lucide-react';
import { predictTweet, TweetResponse } from '../services/api';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { motion } from 'framer-motion';

const TweetAnalysis: React.FC = () => {
  const { modelChoice } = useApi();
  const [tweetText, setTweetText] = useState('');
  const [userId, setUserId] = useState('');
  const [result, setResult] = useState<TweetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getSentimentColor = (prediction: string): string => {
    const pred = prediction.toLowerCase();
    if (pred.includes('positive') || pred.includes('neutral')) return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-300';
    if (pred.includes('complaint')) return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-300';
    if (pred.includes('network') || pred.includes('reliability')) return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-300';
    if (pred.includes('privacy') || pred.includes('hate')) return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-300';
    return 'text-blue-600 bg-blue-100 dark:bg-blue-900 dark:text-blue-300';
  };

  const generateProactiveResponse = (prediction: string): string => {
    const responses: { [key: string]: string } = {
      'MPESA complaint': '📢 It looks like you\'re having an MPESA issue. We\'re sorry for the inconvenience. Please rest assured that your transaction is being reviewed and we\'ll get back to you shortly.',
      'Customer care complaint': '🙋‍♂️ Thank you for reaching out to Safaricom Care. A customer representative will assist you shortly.',
      'Network reliability problem': '📶 Our network is currently experiencing technical issues in some areas. Our technical team is working round the clock to restore full service.',
      'Data protection and privacy concern': '🔐 Thank you for raising this concern. Safaricom takes data protection seriously and we are reviewing the matter.',
      'Internet or airtime bundle complaint': '📲 We acknowledge the reported internet bundles problem. Our team is looking to improve the data deals and coverage for ease of using internet bundles.',
      'Neutral': '🎉 We\'re glad you\'re enjoying our services! Your positive feedback keeps us going. Thank you!',
      'Hate Speech': '🤖 We are sorry if our services are not up to per with your expectations. We are working round the clock to provide reliable services.'
    };

    return responses[prediction] || responses['Neutral'];
  };

  const handleAnalyze = async () => {
    if (!tweetText.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await predictTweet(
        { text: tweetText, user_id: userId || undefined },
        modelChoice === 'Transformer'
      );
      
      if (response.success && response.data) {
        setResult(response.data);
      } else {
        setError(response.error || 'Failed to analyze tweet');
        setResult(null);
      }
    } catch (error) {
      console.error('Error analyzing tweet:', error);
      setError('An unexpected error occurred');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const pieData = result ? Object.entries(result.probabilities).map(([key, value]) => ({
    name: key,
    value: value * 100,
  })) : [];

  const barData = result ? Object.entries(result.probabilities).map(([key, value]) => ({
    class: key.length > 15 ? key.substring(0, 15) + '...' : key,
    probability: value * 100,
  })) : [];

  const COLORS = [
    '#3B82F6', // Blue
    '#10B981', // Emerald  
    '#F59E0B', // Amber
    '#EF4444', // Red
    '#8B5CF6', // Violet
    '#14B8A6', // Teal
    '#F97316', // Orange
    '#EC4899', // Pink
    '#6366F1', // Indigo
    '#84CC16'  // Lime
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
      className="max-w-7xl mx-auto space-y-10 pb-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="text-center space-y-4" variants={itemVariants}>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          📝 Single Tweet Analysis
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          Enter a tweet to get an instant classification and proactive response.
        </p>
      </motion.div>

      {/* Input Form */}
      <motion.div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-10" variants={itemVariants}>
        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <label htmlFor="tweet" className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-4">
              Tweet Content
            </label>
            <textarea
              id="tweet"
              value={tweetText}
              onChange={(e) => setTweetText(e.target.value)}
              placeholder="Enter the tweet text you want to analyze. Our AI will classify it and provide insights about sentiment, complaints, and other relevant categories..."
              className="w-full p-6 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white resize-none text-lg leading-relaxed"
              rows={6}
            />
          </div>

          <div className="space-y-6">
            <div>
              <label htmlFor="userId" className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-4">
                User ID (Optional)
              </label>
              <input
                id="userId"
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Enter user ID"
                className="w-full p-4 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white text-lg"
              />
            </div>
            
            {/* Analysis Info */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">Analysis Features</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-2">
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                  Sentiment Classification
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                  Complaint Detection
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                  AI-Generated Responses
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                  Confidence Scoring
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-8 flex justify-center">

          <button
            onClick={handleAnalyze}
            disabled={!tweetText.trim() || loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-4 px-6 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
          >
            {loading ? (
              <LoadingSpinner size="sm" className="border-white border-t-blue-200" />
            ) : (
              <>
                <Search className="w-5 h-5" />
                <span>Analyze Tweet</span>
              </>
            )}
          </button>
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
                Analysis Failed
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

      {/* Results */}
      {result && !error && (
        <motion.div 
          className="space-y-8"
          variants={itemVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Prediction Result */}
          <div className={`rounded-xl p-6 ${getSentimentColor(result.prediction)} border`}>
            <div className="text-center">
              <h3 className="text-2xl font-bold mb-2">🎯 Prediction: {result.prediction.toUpperCase()}</h3>
              <p className="text-lg">
                <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Proactive Response */}
          <div className="bg-green-50 dark:bg-green-900 border-2 border-green-200 dark:border-green-700 rounded-xl p-6">
            <h4 className="text-xl font-bold text-green-800 dark:text-green-200 mb-4 flex items-center">
              <MessageSquare className="w-6 h-6 mr-2" />
              AI Assistant Response
            </h4>
            <p className="text-green-700 dark:text-green-300 leading-relaxed">
              {generateProactiveResponse(result.prediction)}
            </p>
          </div>

          {/* Enhanced Charts Section */}
          <div className="space-y-8">
            {/* Main Charts Grid - Larger and More Prominent */}
            <div className="grid lg:grid-cols-2 gap-10">
              {/* Enhanced Pie Chart with Larger Size */}
              <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-800 dark:via-gray-850 dark:to-gray-900 rounded-2xl shadow-2xl border-2 border-blue-200 dark:border-gray-600 p-8 transition-all duration-300 hover:shadow-3xl hover:scale-[1.02]">
                <div className="flex items-center justify-center mb-8">
                  <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-4 rounded-full mr-4 shadow-lg">
                    <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                      <span className="text-blue-600 text-lg font-bold">📊</span>
                    </div>
                  </div>
                  <h4 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Probability Distribution
                  </h4>
                </div>
                
                <ResponsiveContainer width="100%" height={450}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={false}
                      outerRadius={130}
                      innerRadius={60}
                      fill="#8884d8"
                      dataKey="value"
                      stroke="rgba(255, 255, 255, 0.8)"
                      strokeWidth={3}
                    >
                      {pieData.map((_, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={COLORS[index % COLORS.length]} 
                          className="drop-shadow-xl hover:opacity-90 transition-all duration-300 cursor-pointer"
                          style={{
                            filter: `drop-shadow(0 4px 8px ${COLORS[index % COLORS.length]}40)`
                          }}
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]}
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.98)',
                        borderRadius: '12px',
                        border: 'none',
                        boxShadow: '0 15px 35px rgba(0, 0, 0, 0.15)',
                        fontSize: '15px',
                        padding: '12px 16px'
                      }}
                      labelStyle={{ fontWeight: 'bold', color: '#1F2937', fontSize: '16px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                
                {/* Enhanced Custom Legend */}
                <div className="mt-6 space-y-3 max-h-48 overflow-y-auto">
                  {pieData.map((entry, index) => (
                    <div key={entry.name} className="flex items-center justify-between p-3 bg-white dark:bg-gray-700 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 border border-gray-100 dark:border-gray-600">
                      <div className="flex items-center">
                        <div 
                          className="w-5 h-5 rounded-full mr-4 shadow-lg border-2 border-white" 
                          style={{ 
                            backgroundColor: COLORS[index % COLORS.length],
                            boxShadow: `0 2px 8px ${COLORS[index % COLORS.length]}50`
                          }}
                        ></div>
                        <span className="text-sm font-semibold text-gray-800 dark:text-gray-200 truncate max-w-[180px]" title={entry.name}>
                          {entry.name}
                        </span>
                      </div>
                      <div className="flex items-center">
                        <span className="text-lg font-bold text-gray-900 dark:text-white">
                          {entry.value.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Enhanced Bar Chart with Larger Size */}
              <div className="bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 dark:from-gray-800 dark:via-gray-850 dark:to-gray-900 rounded-2xl shadow-2xl border-2 border-green-200 dark:border-gray-600 p-8 transition-all duration-300 hover:shadow-3xl hover:scale-[1.02]">
                <div className="flex items-center justify-center mb-8">
                  <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-4 rounded-full mr-4 shadow-lg">
                    <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                      <span className="text-green-600 text-lg font-bold">📈</span>
                    </div>
                  </div>
                  <h4 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Detailed Probabilities
                  </h4>
                </div>
                
                <ResponsiveContainer width="100%" height={450}>
                  <BarChart 
                    data={barData} 
                    margin={{ top: 30, right: 40, left: 30, bottom: 100 }}
                  >
                    <CartesianGrid 
                      strokeDasharray="4 4" 
                      stroke="rgba(156, 163, 175, 0.4)"
                      horizontal={true}
                      vertical={false}
                    />
                    <XAxis 
                      dataKey="class" 
                      angle={-30} 
                      textAnchor="end" 
                      height={90}
                      fontSize={12}
                      fontWeight="600"
                      stroke="#374151"
                      tick={{ fill: '#374151' }}
                      interval={0}
                    />
                    <YAxis 
                      fontSize={12}
                      fontWeight="600"
                      stroke="#374151"
                      tick={{ fill: '#374151' }}
                      label={{ 
                        value: 'Probability (%)', 
                        angle: -90, 
                        position: 'insideLeft', 
                        style: { textAnchor: 'middle', fill: '#374151', fontSize: '14px', fontWeight: 'bold' } 
                      }}
                    />
                    <Tooltip 
                      formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Probability']}
                      contentStyle={{
                        backgroundColor: 'rgba(255, 255, 255, 0.98)',
                        borderRadius: '12px',
                        border: 'none',
                        boxShadow: '0 15px 35px rgba(0, 0, 0, 0.15)',
                        fontSize: '15px',
                        padding: '12px 16px'
                      }}
                      labelStyle={{ fontWeight: 'bold', color: '#1F2937', fontSize: '16px' }}
                    />
                    <Bar 
                      dataKey="probability" 
                      radius={[6, 6, 0, 0]}
                      className="drop-shadow-md"
                    >
                      {barData.map((_, index) => (
                        <Cell 
                          key={`bar-cell-${index}`} 
                          fill={`url(#enhanced-gradient-${index})`}
                          className="hover:opacity-90 transition-all duration-300 cursor-pointer"
                        />
                      ))}
                    </Bar>
                    <defs>
                      {barData.map((_, index) => (
                        <linearGradient key={`enhanced-gradient-${index}`} id={`enhanced-gradient-${index}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={COLORS[index % COLORS.length]} stopOpacity={1} />
                          <stop offset="50%" stopColor={COLORS[index % COLORS.length]} stopOpacity={0.8} />
                          <stop offset="100%" stopColor={COLORS[index % COLORS.length]} stopOpacity={0.6} />
                        </linearGradient>
                      ))}
                    </defs>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Enhanced Probabilities Table */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-xl border border-purple-200 dark:border-gray-700 overflow-hidden">
            <div className="px-6 py-4 border-b border-purple-200 dark:border-gray-700 bg-white dark:bg-gray-800">
              <div className="flex items-center">
                <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full mr-3">
                  <div className="w-5 h-5 bg-purple-600 dark:bg-purple-400 rounded-full"></div>
                </div>
                <h4 className="text-xl font-bold text-gray-900 dark:text-white">📊 Classification Confidence</h4>
              </div>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {Object.entries(result.probabilities)
                  .sort(([,a], [,b]) => b - a) // Sort by probability descending
                  .map(([key, value], index) => (
                    <div 
                      key={key} 
                      className="bg-white dark:bg-gray-700 rounded-lg p-4 shadow-md border border-gray-100 dark:border-gray-600 transition-all duration-200 hover:shadow-lg hover:scale-[1.02]"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <div 
                            className="w-4 h-4 rounded-full mr-3 shadow-sm flex-shrink-0" 
                            style={{ backgroundColor: COLORS[index % COLORS.length] }}
                          ></div>
                          <span className="font-semibold text-gray-800 dark:text-gray-200 text-sm">
                            {key}
                          </span>
                        </div>
                        <span className="font-bold text-lg text-gray-900 dark:text-white">
                          {(value * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      {/* Progress Bar */}
                      <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-3 shadow-inner">
                        <div 
                          className="h-3 rounded-full shadow-sm transition-all duration-700 ease-out"
                          style={{ 
                            width: `${value * 100}%`,
                            background: `linear-gradient(90deg, ${COLORS[index % COLORS.length]}, ${COLORS[index % COLORS.length]}90)`
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default TweetAnalysis;