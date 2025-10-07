import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { ApiProvider } from './contexts/ApiContext';
import Header from './components/Header';
import PageTransition from './components/PageTransition';
import HomePage from './pages/HomePage';
import TweetAnalysis from './pages/TweetAnalysis';
import BatchAnalysis from './pages/BatchAnalysis';
import AiAssistant from './pages/AiAssistant';
import SystemInfo from './pages/SystemInfo';

function App() {
  return (
    <ThemeProvider>
      <ApiProvider>
        <Router>
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
            <Header />
            <main className="container mx-auto px-4 py-8">
              <PageTransition>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/analysis" element={<TweetAnalysis />} />
                  <Route path="/batch" element={<BatchAnalysis />} />
                  <Route path="/chat" element={<AiAssistant />} />
                  <Route path="/system" element={<SystemInfo />} />
                </Routes>
              </PageTransition>
            </main>
            <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16">
              <div className="container mx-auto px-4 py-8 text-center">
                <div className="text-gray-600 dark:text-gray-400">
                  <p className="mb-2">AI-enabled Tweet classifier, powered by Scikit-Learn and Hugging Face Transformers</p>
                  <p className="text-sm">Developed by Patrick Maina, Christine Ndungu, Teresia Njoki and George Nyandusi</p>
                  <p className="text-sm mt-2">&copy; 2025 All Rights Reserved.</p>
                </div>
              </div>
            </footer>
          </div>
        </Router>
      </ApiProvider>
    </ThemeProvider>
  );
}

export default App;