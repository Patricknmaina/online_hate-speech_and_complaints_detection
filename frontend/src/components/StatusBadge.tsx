import React from 'react';
import { CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react';

interface StatusBadgeProps {
  type: 'success' | 'error' | 'warning' | 'info';
  children: React.ReactNode;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ type, children }) => {
  const configs = {
    success: {
      icon: CheckCircle,
      classes: 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border-green-200 dark:border-green-800',
    },
    error: {
      icon: XCircle,
      classes: 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border-red-200 dark:border-red-800',
    },
    warning: {
      icon: AlertCircle,
      classes: 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 border-yellow-200 dark:border-yellow-800',
    },
    info: {
      icon: Info,
      classes: 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-800',
    },
  };

  const { icon: Icon, classes } = configs[type];

  return (
    <div className={`flex items-center space-x-2 px-4 py-3 rounded-lg border font-medium ${classes}`}>
      <Icon className="w-5 h-5 flex-shrink-0" />
      <div>{children}</div>
    </div>
  );
};

export default StatusBadge;