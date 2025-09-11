import React, { createContext, useContext, useState } from 'react';

interface ApiContextType {
  modelChoice: 'Transformer' | 'Sklearn';
  setModelChoice: (model: 'Transformer' | 'Sklearn') => void;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

export const ApiProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [modelChoice, setModelChoice] = useState<'Transformer' | 'Sklearn'>('Transformer');

  return (
    <ApiContext.Provider value={{ modelChoice, setModelChoice }}>
      {children}
    </ApiContext.Provider>
  );
};