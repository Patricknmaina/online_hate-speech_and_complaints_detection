import React from 'react';

interface StatusBadgeProps {
  type: 'success' | 'error' | 'warning' | 'info';
  children: React.ReactNode;
}

const SimpleStatusBadge: React.FC<StatusBadgeProps> = ({ type, children }) => {
  const configs = {
    success: {
      emoji: '✅',
      classes: 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border-green-200 dark:border-green-800',
    },
    error: {
      emoji: '❌',
      classes: 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-200 dark:border-red-800',
    },
    warning: {
      emoji: '⚠️',
      classes: 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 border-yellow-200 dark:border-yellow-800',
    },
    info: {
      emoji: 'ℹ️',
      classes: 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-800',
    },
  };

  const { emoji, classes } = configs[type];

  return (
    <div className={`flex items-center space-x-2 px-4 py-3 rounded-lg border font-medium ${classes}`}>
      <span className="text-xl flex-shrink-0">{emoji}</span>
      <div>{children}</div>
    </div>
  );
};

export default SimpleStatusBadge;
