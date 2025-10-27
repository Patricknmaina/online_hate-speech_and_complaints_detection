import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  MessageSquare, 
  Upload, 
  Bot, 
  TrendingUp, 
  Shield, 
  Zap, 
  BarChart3,
  ArrowRight,
  Sparkles,
  CheckCircle
} from 'lucide-react';

const SimpleHomePage: React.FC = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 100
      }
    }
  };

  return (
    <motion.div 
      className="space-y-16 pb-8"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      {/* Hero Section */}
      <motion.section className="text-center space-y-8" variants={itemVariants}>
        <motion.div 
          className="bg-gradient-to-br from-blue-50 to-teal-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-12 shadow-xl"
          whileHover={{ scale: 1.02 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            Safarimeter: The Pulse of Public Opinion
            <motion.span
              className="inline-block ml-3"
              animate={{ rotate: [0, 10, -10, 10, 0] }}
              transition={{ duration: 1, repeat: Infinity, repeatDelay: 2 }}
            >
              <Sparkles className="inline w-12 h-12 text-blue-600 dark:text-blue-400" />
            </motion.span>
          </motion.h1>
          <motion.p 
            className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            AI that listens. Instantly spot complaints, understand sentiment, and take action — all from Twitter conversations.
          </motion.p>
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link
              to="/analysis"
              className="inline-flex items-center space-x-3 bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-8 rounded-xl text-lg transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              <span>Get Started Now!</span>
              <ArrowRight className="w-6 h-6" />
            </Link>
          </motion.div>
        </motion.div>
      </motion.section>

      <hr className="border-gray-200 dark:border-gray-700" />

      {/* Platform Capabilities */}
      <motion.section className="space-y-8" variants={itemVariants}>
        <motion.h2 
          className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <Sparkles className="inline w-8 h-8 mr-2 text-purple-600" />
          Platform Capabilities
        </motion.h2>

        <div className="grid md:grid-cols-3 gap-8">
          <motion.div 
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300"
            whileHover={{ scale: 1.05, y: -5 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="flex items-center space-x-3 mb-6">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <MessageSquare className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </motion.div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">Tweet Analysis</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Classify individual tweets to understand their sentiment, and identify specific issues like network reliability complaints, MPESA complaints, Customer care issues or hate speech towards Safaricom.
            </p>
          </motion.div>

          <motion.div 
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300"
            whileHover={{ scale: 1.05, y: -5 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <div className="flex items-center space-x-3 mb-6">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <Upload className="w-8 h-8 text-green-600 dark:text-green-400" />
              </motion.div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">Batch Analysis</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Upload CSV files containing multiple tweets for bulk classification. Visualize the distribution of the predictions and download detailed results for further analysis.
            </p>
          </motion.div>

          <motion.div 
            className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300"
            whileHover={{ scale: 1.05, y: -5 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <div className="flex items-center space-x-3 mb-6">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <Bot className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              </motion.div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">AI Assistant</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Interact with an AI-powered chatbot that can provide insights and responses based on tweet classifications, helping with automating customer service interactions.
            </p>
          </motion.div>
        </div>
      </motion.section>

      <hr className="border-gray-200 dark:border-gray-700" />

      {/* How It Works */}
      <motion.section className="space-y-8" variants={itemVariants}>
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white">
          <TrendingUp className="inline w-8 h-8 mr-2 text-blue-600" />
          Basic Workflow
        </h2>

        <motion.div 
          className="bg-gray-50 dark:bg-gray-800 rounded-xl p-8 shadow-lg"
          whileHover={{ scale: 1.01 }}
        >
          <div className="space-y-6">
            <motion.div 
              className="bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-blue-500"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.5 }}
              whileHover={{ x: 10 }}
            >
              <p className="text-gray-700 dark:text-gray-300 flex items-start">
                <Upload className="w-6 h-6 mr-3 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-1" />
                <span>
                  <strong className="text-blue-600 dark:text-blue-400">1. Data Ingestion:</strong> Tweets are fed into the system either individually or in batches via CSV uploads.
                </span>
              </p>
            </motion.div>

            <motion.div 
              className="bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-green-500"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
              whileHover={{ x: 10 }}
            >
              <p className="text-gray-700 dark:text-gray-300 flex items-start">
                <Bot className="w-6 h-6 mr-3 text-green-600 dark:text-green-400 flex-shrink-0 mt-1" />
                <span>
                  <strong className="text-green-600 dark:text-green-400">2. AI-Powered Classification:</strong> Our robust FastAPI backend, powered by Transformer-based and Scikit-learn models, processes the tweets to classify their sentiment and intent.
                </span>
              </p>
            </motion.div>

            <motion.div 
              className="bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-yellow-500"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.7 }}
              whileHover={{ x: 10 }}
            >
              <p className="text-gray-700 dark:text-gray-300 flex items-start">
                <Sparkles className="w-6 h-6 mr-3 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-1" />
                <span>
                  <strong className="text-yellow-600 dark:text-yellow-400">3. Instant Insights:</strong> Get immediate predictions, confidence scores, and probability distributions for each tweet.
                </span>
              </p>
            </motion.div>

            <motion.div 
              className="bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-purple-500"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.8 }}
              whileHover={{ x: 10 }}
            >
              <p className="text-gray-700 dark:text-gray-300 flex items-start">
                <MessageSquare className="w-6 h-6 mr-3 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-1" />
                <span>
                  <strong className="text-purple-600 dark:text-purple-400">4. Proactive Engagement:</strong> The AI Assistant generates automated, context-aware responses, streamlining your customer service workflow.
                </span>
              </p>
            </motion.div>
          </div>
        </motion.div>
      </motion.section>

      {/* Features Grid */}
      <motion.section className="space-y-8" variants={itemVariants}>
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white">
          Why Choose Safarimeter?
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <motion.div 
            className="flex items-start space-x-4 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md"
            whileHover={{ scale: 1.05, boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)" }}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
          >
            <motion.div
              whileHover={{ scale: 1.2, rotate: 360 }}
              transition={{ duration: 0.5 }}
            >
              <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            </motion.div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Real-time Analytics</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Get instant insights and predictions with confidence scores
              </p>
            </div>
          </motion.div>

          <motion.div 
            className="flex items-start space-x-4 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md"
            whileHover={{ scale: 1.05, boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)" }}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.8 }}
          >
            <motion.div
              whileHover={{ scale: 1.2, rotate: 360 }}
              transition={{ duration: 0.5 }}
            >
              <Shield className="w-8 h-8 text-green-600 dark:text-green-400" />
            </motion.div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Enterprise Ready</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Production-ready with robust error handling and monitoring
              </p>
            </div>
          </motion.div>

          <motion.div 
            className="flex items-start space-x-4 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md"
            whileHover={{ scale: 1.05, boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)" }}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.9 }}
          >
            <motion.div
              whileHover={{ scale: 1.2, rotate: 360 }}
              transition={{ duration: 0.5 }}
            >
              <Zap className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </motion.div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Lightning Fast</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Process thousands of tweets in seconds with batch analysis
              </p>
            </div>
          </motion.div>
        </div>
      </motion.section>
    </motion.div>
  );
};

export default SimpleHomePage;
