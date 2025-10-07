import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, MessageSquare, Upload, Bot, BarChart3, Shield, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

const HomePage: React.FC = () => {
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
      className="space-y-16 pb-8"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Hero Section */}
      <motion.section className="text-center space-y-8" variants={itemVariants}>
        <div className="bg-gradient-to-br from-blue-50 to-teal-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-12 shadow-xl">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
            Safarimeter: The Pulse of Public Opinion 📲
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8 max-w-4xl mx-auto leading-relaxed">
            AI that listens. Instantly spot complaints, understand sentiment, and take action — all from Twitter conversations.
          </p>
          <Link
            to="/analysis"
            className="inline-flex items-center space-x-3 bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-8 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
          >
            <span>Get Started Now!</span>
            <ArrowRight className="w-6 h-6" />
          </Link>
        </div>
      </motion.section>

      <motion.hr className="border-gray-200 dark:border-gray-700" variants={itemVariants} />

      {/* Platform Capabilities */}
      <motion.section className="space-y-8" variants={itemVariants}>
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white">
          🪄 Platform Capabilities
        </h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:transform hover:scale-105 hover:shadow-xl">
            <div className="flex items-center space-x-3 mb-6">
              <MessageSquare className="w-8 h-8 text-teal-600 dark:text-teal-400" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">Tweet Analysis</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Classify individual tweets to understand their sentiment, and identify specific issues like network reliability complaints, MPESA complaints, Customer care issues or hate speech towards Safaricom. Get instant proactive responses.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:transform hover:scale-105 hover:shadow-xl">
            <div className="flex items-center space-x-3 mb-6">
              <Upload className="w-8 h-8 text-teal-600 dark:text-teal-400" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">Batch Analysis</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Upload CSV files containing multiple tweets for bulk classification. Visualize the distribution of the predictions and download detailed results for further analysis.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:transform hover:scale-105 hover:shadow-xl">
            <div className="flex items-center space-x-3 mb-6">
              <Bot className="w-8 h-8 text-teal-600 dark:text-teal-400" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">AI Assistant</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              Interact with an AI-powered chatbot that can provide insights and responses based on tweet classifications, helping with automating customer service interactions.
            </p>
          </div>
        </div>
      </motion.section>

      <motion.hr className="border-gray-200 dark:border-gray-700" variants={itemVariants} />

      {/* How It Works */}
      <motion.section className="space-y-8" variants={itemVariants}>
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white">
          ⛏️ Basic Workflow
        </h2>
        
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-8 shadow-lg">
          <div className="space-y-6">
            <div className="group bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-blue-500 transition-all duration-300 hover:border-l-8 hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:shadow-lg hover:transform hover:translate-x-2 cursor-pointer">
              <p className="text-gray-700 dark:text-gray-300 group-hover:text-blue-800 dark:group-hover:text-blue-200 transition-colors duration-300">
                <strong className="text-blue-600 dark:text-blue-400 group-hover:text-blue-700 dark:group-hover:text-blue-300">1. Data Ingestion:</strong> 📥 Tweets are fed into the system either individually or in batches via CSV uploads.
              </p>
            </div>
            
            <div className="group bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-green-500 transition-all duration-300 hover:border-l-8 hover:border-green-600 hover:bg-green-50 dark:hover:bg-green-900/20 hover:shadow-lg hover:transform hover:translate-x-2 cursor-pointer">
              <p className="text-gray-700 dark:text-gray-300 group-hover:text-green-800 dark:group-hover:text-green-200 transition-colors duration-300">
                <strong className="text-green-600 dark:text-green-400 group-hover:text-green-700 dark:group-hover:text-green-300">2. AI-Powered Classification:</strong> 🧠 Our robust FastAPI backend, powered by Transformer-based and Scikit-learn models, processes the tweets to classify their sentiment and intent.
              </p>
            </div>
            
            <div className="group bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-yellow-500 transition-all duration-300 hover:border-l-8 hover:border-yellow-600 hover:bg-yellow-50 dark:hover:bg-yellow-900/20 hover:shadow-lg hover:transform hover:translate-x-2 cursor-pointer">
              <p className="text-gray-700 dark:text-gray-300 group-hover:text-yellow-800 dark:group-hover:text-yellow-200 transition-colors duration-300">
                <strong className="text-yellow-600 dark:text-yellow-400 group-hover:text-yellow-700 dark:group-hover:text-yellow-300">3. Instant Insights:</strong> 💡 Get immediate predictions, confidence scores, and probability distributions for each tweet.
              </p>
            </div>
            
            <div className="group bg-white dark:bg-gray-700 rounded-lg p-6 border-l-4 border-purple-500 transition-all duration-300 hover:border-l-8 hover:border-purple-600 hover:bg-purple-50 dark:hover:bg-purple-900/20 hover:shadow-lg hover:transform hover:translate-x-2 cursor-pointer">
              <p className="text-gray-700 dark:text-gray-300 group-hover:text-purple-800 dark:group-hover:text-purple-200 transition-colors duration-300">
                <strong className="text-purple-600 dark:text-purple-400 group-hover:text-purple-700 dark:group-hover:text-purple-300">4. Proactive Engagement:</strong> 💬 The AI Assistant generates automated, context-aware responses, streamlining your customer service workflow.
              </p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Features Grid */}
      <motion.section className="space-y-8" variants={itemVariants}>
        <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 dark:text-white">
          Why Choose Safarimeter?
        </h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="group flex items-start space-x-4 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md transition-all duration-300 hover:shadow-xl hover:transform hover:scale-105 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer border border-transparent hover:border-blue-200 dark:hover:border-blue-700">
            <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900 group-hover:bg-blue-200 dark:group-hover:bg-blue-800 transition-colors duration-300">
              <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400 flex-shrink-0 group-hover:scale-110 transition-transform duration-300" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-blue-700 dark:group-hover:text-blue-300 transition-colors duration-300">Real-time Analytics</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm group-hover:text-blue-600 dark:group-hover:text-blue-200 transition-colors duration-300">
                Get instant insights and predictions with confidence scores
              </p>
            </div>
          </div>
          
          <div className="group flex items-start space-x-4 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md transition-all duration-300 hover:shadow-xl hover:transform hover:scale-105 hover:bg-green-50 dark:hover:bg-green-900/20 cursor-pointer border border-transparent hover:border-green-200 dark:hover:border-green-700">
            <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900 group-hover:bg-green-200 dark:group-hover:bg-green-800 transition-colors duration-300">
              <Shield className="w-8 h-8 text-green-600 dark:text-green-400 flex-shrink-0 group-hover:scale-110 transition-transform duration-300" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-green-700 dark:group-hover:text-green-300 transition-colors duration-300">Enterprise Ready</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm group-hover:text-green-600 dark:group-hover:text-green-200 transition-colors duration-300">
                Production-ready with robust error handling and monitoring
              </p>
            </div>
          </div>
          
          <div className="group flex items-start space-x-4 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md transition-all duration-300 hover:shadow-xl hover:transform hover:scale-105 hover:bg-yellow-50 dark:hover:bg-yellow-900/20 cursor-pointer border border-transparent hover:border-yellow-200 dark:hover:border-yellow-700">
            <div className="p-2 rounded-lg bg-yellow-100 dark:bg-yellow-900 group-hover:bg-yellow-200 dark:group-hover:bg-yellow-800 transition-colors duration-300">
              <Zap className="w-8 h-8 text-yellow-600 dark:text-yellow-400 flex-shrink-0 group-hover:scale-110 transition-transform duration-300" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-yellow-700 dark:group-hover:text-yellow-300 transition-colors duration-300">Lightning Fast</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm group-hover:text-yellow-600 dark:group-hover:text-yellow-200 transition-colors duration-300">
                Process thousands of tweets in seconds with batch analysis
              </p>
            </div>
          </div>
        </div>
      </motion.section>
    </motion.div>
  );
};

export default HomePage;