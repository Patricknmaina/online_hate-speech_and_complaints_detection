import React, { useState, useEffect } from 'react';
import { RefreshCw, Server, Database, Cpu, Clock } from 'lucide-react';
import { checkApiHealth, getModelInfo, getChatStatus, HealthResponse, ModelInfo, ChatStatus } from '../services/api';
import StatusBadge from '../components/StatusBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import { motion } from 'framer-motion';

const SystemInfo: React.FC = () => {
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
      className="max-w-4xl mx-auto space-y-8 pb-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="text-center space-y-4" variants={itemVariants}>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          ⚙️ System Information
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          View the current status of the API, chatbot, and model details.
        </p>
        <button
          onClick={fetchSystemInfo}
          disabled={loading}
          className="inline-flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
        >
          {loading ? (
            <LoadingSpinner size="sm" className="border-white border-t-blue-200" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
          <span>Refresh Status</span>
        </button>
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
                Connection Failed
              </h4>
              <p className="text-red-700 dark:text-red-300 text-sm leading-relaxed mb-3">
                {error}
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>💡 Quick Fix:</strong> Make sure your FastAPI backend is running on http://localhost:8000
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
          <Server className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">🚀 API Status</h2>
        </div>
        
        {apiHealthy && healthInfo ? (
          <div className="space-y-4">
            <StatusBadge type="success">
              <strong>FastAPI Endpoint:</strong> Connected ✅
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
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Model Loading Status</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Sklearn Model:</span> 
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      healthInfo.model_info.sklearn_loaded 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {healthInfo.model_info.sklearn_loaded ? 'Loaded' : 'Not Loaded'}
                    </span>
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Transformer Model:</span> 
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      healthInfo.model_info.transformer_loaded 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {healthInfo.model_info.transformer_loaded ? 'Loaded' : 'Not Loaded'}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <StatusBadge type="error">
              <strong>FastAPI Endpoint:</strong> Disconnected ❌
            </StatusBadge>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Unable to connect to the FastAPI backend. The system status cannot be determined.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Chatbot Status */}
      <motion.div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6" variants={itemVariants}>
        <div className="flex items-center space-x-3 mb-4">
          <Database className="w-6 h-6 text-teal-600 dark:text-teal-400" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">🤖 Chatbot Status</h2>
        </div>
        
        {chatStatus ? (
          <div className="space-y-4">
            <StatusBadge type={chatStatus.rasa_available ? "success" : "warning"}>
              <strong>
                {chatStatus.rasa_available ? "Advanced AI Mode Active" : "Fallback Mode Active"}
              </strong> - {chatStatus.rasa_available ? "Full Rasa integration available" : "Using intelligent fallback responses"}
            </StatusBadge>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Rasa Server Status</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Rasa Available:</span> 
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      chatStatus.rasa_available 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                        : 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
                    }`}>
                      {chatStatus.rasa_available ? 'Online' : 'Offline'}
                    </span>
                  </p>
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Overall Status:</span> 
                    <span className="ml-2 px-2 py-1 rounded text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      {chatStatus.status}
                    </span>
                  </p>
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Current Mode</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium text-gray-900 dark:text-white">Mode:</span> 
                    {chatStatus.rasa_available ? " Advanced AI" : " Intelligent Fallback"}
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
                <strong>💡 How it works:</strong> {chatStatus.rasa_available 
                  ? "The chatbot is running in Advanced AI Mode with full Rasa integration, providing sophisticated conversational AI responses."
                  : "The chatbot is using Intelligent Fallback Mode, analyzing your messages with ML models and providing smart, contextual responses based on tweet classification."
                }
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <StatusBadge type="error">
              <strong>Chatbot Status Unknown</strong> - Unable to connect to chatbot service
            </StatusBadge>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Unable to determine chatbot status. Ensure the FastAPI backend is running and accessible.
              </p>
            </div>
          </div>
        )}
      </motion.div>

      {/* Model Details */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Cpu className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">🧠 Model Details</h2>
        </div>
        
        {modelInfo ? (
          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Transformer Model</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Type:</span> {modelInfo.transformer_model_type || 'Not Available'}</p>
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Classes:</span></p>
                  {modelInfo.transformer_classes ? (
                    <div className="ml-4 space-y-1">
                      {Object.values(modelInfo.transformer_classes).map((className, index) => (
                        <div key={index} className="text-xs bg-white dark:bg-gray-600 text-gray-800 dark:text-gray-200 px-2 py-1 rounded">
                          {className}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="ml-4 text-gray-500 dark:text-gray-400">Not Available</p>
                  )}
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Sklearn Model</h3>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Type:</span> {modelInfo.sklearn_model_type || 'Not Available'}</p>
                  <p className="text-gray-700 dark:text-gray-300"><span className="font-medium text-gray-900 dark:text-white">Status:</span> {modelInfo.sklearn_model_type ? 'Active' : 'Not Loaded'}</p>
                </div>
              </div>
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

      {/* App Statistics */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Clock className="w-6 h-6 text-green-600 dark:text-green-400" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">📈 App Statistics</h2>
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
              1.0.0
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">App Version</div>
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

export default SystemInfo;