import React, { createContext, useContext, useState } from 'react';
import { ModelType } from '../services/api';

interface ApiContextType {
  modelChoice: ModelType;
  setModelChoice: (model: ModelType) => void;
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
  // Default to OpenAI as the primary model
  const [modelChoice, setModelChoice] = useState<ModelType>('OpenAI');

  return (
    <ApiContext.Provider value={{ modelChoice, setModelChoice }}>
      {children}
    </ApiContext.Provider>
  );
};
