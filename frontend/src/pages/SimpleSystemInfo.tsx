import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Info, Zap, Brain, MessageSquare, BarChart3 } from 'lucide-react';
import { checkApiHealth, getModelInfo, getChatStatus, HealthResponse, ModelInfo, ChatStatus } from '../services/api';
import StatusBadge from '../components/SimpleStatusBadge';
import LoadingSpinner from '../components/LoadingSpinner';

const SimpleSystemInfo: React.FC = () => {
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [error, setError] = useState<string | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean>(false);

  const fetchSystemInfo = async () => {
    setLoading(true);
    setError(null);

    try {
      const [healthResult, modelResult, chatResult] = await Promise.all([
        checkApiHealth(),
        getModelInfo(),
        getChatStatus()
      ]);

      // Handle health check result
      if (healthResult.success && healthResult.data) {
        setHealthInfo(healthResult.data);
        setApiHealthy(true);
      } else {
        setError(healthResult.error || 'Failed to connect to FastAPI backend');
        setApiHealthy(false);
        setHealthInfo(null);
      }

      // Handle model info result
      if (modelResult.success && modelResult.data) {
        setModelInfo(modelResult.data);
      } else {
        // Only show model error if API is healthy
        if (apiHealthy) {
          setModelInfo(null);
        }
      }

      // Handle chat status result
      if (chatResult.success && chatResult.data) {
        setChatStatus(chatResult.data);
      } else {
        setChatStatus(null);
      }

      setLastRefresh(new Date());
    } catch (error) {
      console.error('Error fetching system info:', error);
      setError('Failed to fetch system information');
      setApiHealthy(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemInfo();
  }, []);

  const formatUptime = () => {
    const diff = Date.now() - lastRefresh.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m`;
  };

  return (
    <motion.div
      className="max-w-4xl mx-auto space-y-8 pb-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div
        className="text-center space-y-4"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white flex items-center justify-center gap-3">
          <Info className="w-10 h-10 text-blue-600 dark:text-blue-400" />
          System Information
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          View the current status of the API, AI models, and system resources.
        </p>
        <motion.button
          onClick={fetchSystemInfo}
          disabled={loading}
          className="inline-flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {loading ? (
            <LoadingSpinner size="sm" className="border-white border-t-blue-200" />
          ) : (
            <motion.div
              animate={{ rotate: loading ? 360 : 0 }}
              transition={{ duration: 1, repeat: loading ? Infinity : 0 }}
            >
              <RefreshCw className="w-5 h-5" />
            </motion.div>
          )}
          <span>Refresh Status</span>
        </motion.button>
      </motion.div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
                <span className="text-red-600 dark:text-red-400 text-sm">!</span>
              </div>
            </div>
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-1">
                Connection Failed
              </h4>
              <p className="text-red-700 dark:text-red-300 text-sm leading-relaxed mb-3">
                {error}
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>Quick Fix:</strong> Make sure your FastAPI backend is running on http://localhost:8000
                </p>
                <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                  Start the backend: <code className="bg-yellow-100 dark:bg-yellow-800 px-2 py-1 rounded">cd FastAPI && python main.py</code>
                </p>
              </div>
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

      {/* API Status */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="w-6 h-6 text-green-500" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">API Status</h2>
        </div>

        {apiHealthy && healthInfo ? (
          <div className="space-y-4">
            <StatusBadge type="success">
              <strong>FastAPI Endpoint:</strong> Connected
            </StatusBadge>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Status Information</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Status:</span> {healthInfo.status}</p>
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Message:</span> {healthInfo.message}</p>
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Last Checked:</span> {lastRefresh.toLocaleString()}</p>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">System Resources</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Memory Usage:</span> {healthInfo.system_info.memory_usage_percent.toFixed(1)}%
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Available Memory:</span> {healthInfo.system_info.available_memory_gb.toFixed(2)} GB
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">CPU Usage:</span> {healthInfo.system_info.cpu_usage_percent.toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <StatusBadge type="error">
              <strong>FastAPI Endpoint:</strong> Disconnected
            </StatusBadge>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Unable to connect to the FastAPI backend. The system status cannot be determined.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* AI Models Status */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="w-6 h-6 text-purple-500" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">AI Models</h2>
        </div>

        {healthInfo ? (
          <div className="space-y-4">
            {/* OpenAI GPT-4o-mini */}
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">OpenAI GPT-4o-mini</h3>
                  <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded-full">
                    Primary
                  </span>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  healthInfo.model_info.openai_available
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                }`}>
                  {healthInfo.model_info.openai_available ? 'Available' : 'Unavailable'}
                </span>
              </div>
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <p><span className="font-medium">Model:</span> {healthInfo.model_info.openai_model || 'gpt-4o-mini'}</p>
                <p><span className="font-medium">Use Case:</span> Tweet classification, AI chat assistant</p>
                <p><span className="font-medium">Features:</span> High accuracy, multilingual support (English, Swahili, Sheng)</p>
              </div>
            </div>

            {/* Sklearn Model */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${healthInfo.model_info.sklearn_loaded ? 'bg-blue-500' : 'bg-gray-400'}`}></div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">Sklearn Classifier</h3>
                  <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full">
                    Fallback
                  </span>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  healthInfo.model_info.sklearn_loaded
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                    : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                }`}>
                  {healthInfo.model_info.sklearn_loaded ? 'Loaded' : 'Not Loaded'}
                </span>
              </div>
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <p><span className="font-medium">Type:</span> {healthInfo.model_info.sklearn_model_type || 'Machine Learning Classifier'}</p>
                <p><span className="font-medium">Use Case:</span> Offline classification, fallback when OpenAI unavailable</p>
                <p><span className="font-medium">Features:</span> Fast inference, no API dependency</p>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
              <p className="text-sm text-purple-800 dark:text-purple-200">
                <strong>How it works:</strong> The system primarily uses OpenAI GPT-4o-mini for accurate classification with reasoning. If OpenAI is unavailable, it automatically falls back to the sklearn model for uninterrupted service.
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-4">
            <p className="text-gray-500 dark:text-gray-400">
              No model information available. Ensure API is running and accessible.
            </p>
          </div>
        )}
      </div>

      {/* Chatbot Status */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <MessageSquare className="w-6 h-6 text-blue-500" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">AI Assistant Status</h2>
        </div>

        {chatStatus ? (
          <div className="space-y-4">
            <StatusBadge type={chatStatus.openai_available ? "success" : "warning"}>
              <strong>
                {chatStatus.openai_available ? "GPT-4o-mini Mode Active" : "Fallback Mode Active"}
              </strong> - {chatStatus.openai_available ? "Full AI assistant capabilities" : "Using classification-based responses"}
            </StatusBadge>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Assistant Configuration</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">OpenAI Available:</span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      chatStatus.openai_available
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
                    }`}>
                      {chatStatus.openai_available ? 'Yes' : 'No'}
                    </span>
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Model:</span> {chatStatus.model}
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Status:</span>
                    <span className="ml-2 px-2 py-1 rounded text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      {chatStatus.status}
                    </span>
                  </p>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Capabilities</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Mode:</span>
                    {chatStatus.openai_available ? " Advanced AI Chat" : " Smart Fallback"}
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Fallback Active:</span>
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      chatStatus.fallback_mode
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                    }`}>
                      {chatStatus.fallback_mode ? 'Yes' : 'No'}
                    </span>
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>How it works:</strong> {chatStatus.openai_available
                  ? "The AI assistant uses GPT-4o-mini to provide intelligent, context-aware responses about Safaricom services. It understands English, Swahili, and Sheng."
                  : "The assistant uses ML classification to understand your message category and provides helpful pre-configured responses based on the detected issue type."
                }
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <StatusBadge type="error">
              <strong>Assistant Status Unknown</strong> - Unable to connect to chatbot service
            </StatusBadge>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Unable to determine assistant status. Ensure the FastAPI backend is running and accessible.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Classification Categories */}
      {modelInfo && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <BarChart3 className="w-6 h-6 text-orange-500" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Classification Categories</h2>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {modelInfo.config.categories.map((category, index) => (
              <div
                key={index}
                className="bg-gray-50 dark:bg-gray-700 rounded-lg px-4 py-3 text-sm font-medium text-gray-800 dark:text-gray-200 flex items-center space-x-2"
              >
                <span className={`w-2 h-2 rounded-full ${
                  category === 'Hate Speech' ? 'bg-red-500' :
                  category === 'Neutral' ? 'bg-gray-400' :
                  category.includes('MPESA') ? 'bg-green-500' :
                  category.includes('Network') ? 'bg-yellow-500' :
                  category.includes('Customer') ? 'bg-blue-500' :
                  category.includes('Data') ? 'bg-purple-500' :
                  'bg-orange-500'
                }`}></span>
                <span>{category}</span>
              </div>
            ))}
          </div>

          <div className="mt-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
            <p className="text-sm text-orange-800 dark:text-orange-200">
              <strong>Note:</strong> These are the 7 categories used for classifying tweets about Safaricom services. The AI models are trained to accurately identify complaints, hate speech, and neutral messages.
            </p>
          </div>
        </div>
      )}

      {/* Performance Features */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="w-6 h-6 text-yellow-500" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Performance Features</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Concurrent Batch Processing</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Batch analysis uses asyncio.gather() for parallel processing, providing up to 10x faster results when analyzing multiple tweets.
            </p>
          </div>

          <div className="bg-gradient-to-br from-cyan-50 to-sky-50 dark:from-cyan-900/20 dark:to-sky-900/20 border border-cyan-200 dark:border-cyan-800 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Thread Pool Optimization</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              10 concurrent workers handle API requests efficiently, ensuring fast response times even under heavy load.
            </p>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Lazy Model Loading</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Models are loaded on-demand with LRU caching, minimizing memory usage and startup time.
            </p>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Automatic Fallback</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              If OpenAI API is unavailable, the system seamlessly falls back to sklearn for uninterrupted service.
            </p>
          </div>
        </div>
      </div>

      {/* App Statistics */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <BarChart3 className="w-6 h-6 text-indigo-500" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">App Statistics</h2>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {new Date().toLocaleDateString()}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Current Date</div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {new Date().toLocaleTimeString()}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Current Time</div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              4.0.0
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">API Version</div>
          </div>
        </div>

        <div className="mt-4 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <p className="text-sm text-gray-700 dark:text-gray-300">
            <span className="font-medium text-gray-900 dark:text-white">Last Status Check:</span> {lastRefresh.toLocaleString()} ({formatUptime()} ago)
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default SimpleSystemInfo;
