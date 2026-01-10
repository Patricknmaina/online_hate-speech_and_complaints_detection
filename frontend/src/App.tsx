import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ThemeProvider } from './contexts/ThemeContext';
import { ApiProvider } from './contexts/ApiContext';
import { ToastProvider } from './contexts/ToastContext';
import SimpleHeader from './components/SimpleHeader';
import SimpleHomePage from './pages/SimpleHomePage';
import SimpleTweetAnalysis from './pages/SimpleTweetAnalysis';
import SimpleBatchAnalysis from './pages/SimpleBatchAnalysis';
import SimpleAiAssistant from './pages/SimpleAiAssistant';
import SimpleSystemInfo from './pages/SimpleSystemInfo';

function App() {
  console.log('App component rendering - Enhanced with framer-motion animations and lucide-react icons!');

  return (
    <ThemeProvider>
      <ApiProvider>
        <ToastProvider>
          <Router>
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
            <SimpleHeader />
            <main className="container mx-auto px-4 py-8">
              <Routes>
                <Route path="/" element={<SimpleHomePage />} />
                <Route path="/analysis" element={<SimpleTweetAnalysis />} />
                <Route path="/batch" element={<SimpleBatchAnalysis />} />
                <Route path="/chat" element={<SimpleAiAssistant />} />
                <Route path="/system" element={<SimpleSystemInfo />} />
              </Routes>
            </main>
            <motion.footer 
              className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <div className="container mx-auto px-4 py-8 text-center">
                <div className="text-gray-600 dark:text-gray-400">
                  <p className="mb-2">AI-enabled Tweet classifier, powered by Scikit-Learn and Hugging Face Transformers</p>
                  <p className="text-sm">Developed by Patrick Maina, Christine Ndungu, Teresia Njoki and George Nyandusi</p>
                  <p className="text-sm mt-2">&copy; 2025 All Rights Reserved.</p>
                </div>
              </div>
            </motion.footer>
          </div>
          </Router>
        </ToastProvider>
      </ApiProvider>
    </ThemeProvider>
  );
}

export default App;