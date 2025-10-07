import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { MessageSquare, Sun, Moon, Settings, Home, FileText, Upload, Bot, Info } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useApi } from '../contexts/ApiContext';

const Header: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const { modelChoice, setModelChoice } = useApi();
  const location = useLocation();

  const navigationItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/analysis', icon: FileText, label: 'Analysis' },
    { path: '/batch', icon: Upload, label: 'Batch' },
    { path: '/chat', icon: Bot, label: 'Assistant' },
    { path: '/system', icon: Info, label: 'System' },
  ];

  return (
    <header className="bg-white dark:bg-gray-800 shadow-lg border-b border-gray-200 dark:border-gray-700">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between py-4">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <MessageSquare className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                Safarimeter
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                The Pulse of Public Opinion
              </p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex space-x-1">
            {navigationItems.map(({ path, icon: Icon, label }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors duration-200 ${
                  location.pathname === path
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{label}</span>
              </Link>
            ))}
          </nav>

          {/* Controls */}
          <div className="flex items-center space-x-4">
            {/* Model Selection */}
            <div className="flex items-center space-x-2">
              <Settings className="w-4 h-4 text-gray-500 dark:text-gray-400" />
              <select
                value={modelChoice}
                onChange={(e) => setModelChoice(e.target.value as 'Transformer' | 'Sklearn')}
                className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1 text-sm text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Transformer">Transformer</option>
                <option value="Sklearn">Sklearn</option>
              </select>
            </div>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5 text-yellow-500" />
              ) : (
                <Moon className="w-5 h-5 text-gray-600" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden border-t border-gray-200 dark:border-gray-700 pt-4 pb-4">
          <nav className="flex overflow-x-auto space-x-1 pb-2">
            {navigationItems.map(({ path, icon: Icon, label }) => (
              <Link
                key={path}
                to={path}
                className={`flex-shrink-0 flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors duration-200 ${
                  location.pathname === path
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium whitespace-nowrap">{label}</span>
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;