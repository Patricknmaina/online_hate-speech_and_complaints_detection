import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, Send, Trash2, MessageCircle, User, Sparkles } from 'lucide-react';
import { sendChatMessage, getChatStatus, ChatRequest, ChatStatus } from '../services/api';

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const SimpleAiAssistant: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatStatus, setChatStatus] = useState<ChatStatus | null>(null);
  const [sessionId] = useState(() => `web_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  // Check chatbot status on component mount
  useEffect(() => {
    const checkStatus = async () => {
      const statusResponse = await getChatStatus();
      if (statusResponse.success && statusResponse.data) {
        setChatStatus(statusResponse.data);
      }
    };
    checkStatus();
  }, []);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);

    try {
      const chatRequest: ChatRequest = {
        message: inputText.trim(),
        sender_id: sessionId
      };

      const response = await sendChatMessage(chatRequest);

      if (response.success && response.data) {
        // Handle multiple responses from the chatbot
        const botResponses = response.data.responses.map((botMsg, index) => ({
          id: Date.now() + index + 1,
          role: 'assistant' as const,
          content: botMsg.text || 'I received your message but I\'m not sure how to respond.',
          timestamp: new Date()
        }));

        setMessages(prev => [...prev, ...botResponses]);
      } else {
        const errorMessage: ChatMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.error || 'Sorry, I encountered an error while processing your message. Please try again.',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch {
      const errorMessage: ChatMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your message. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const handleSampleQuestion = (question: string) => {
    setInputText(question);
  };

  const sampleQuestions = [
    'Hello, I need help with my MPESA transaction',
    'Why is my network so slow today?',
    'Thank you for your excellent customer service!',
    'How can I buy data bundles?',
    'I have a complaint about my bill',
    'Can you help me with my account issues?'
  ];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <motion.div 
      className="max-w-4xl mx-auto space-y-8 pb-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div 
        className="text-center space-y-4"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white flex items-center justify-center gap-3">
          <motion.div
            animate={{ rotate: [0, 10, -10, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          >
            <Bot className="w-10 h-10 text-purple-600 dark:text-purple-400" />
          </motion.div>
          AI Assistant
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          Chat with the AI about Safaricom services and get instant responses.
        </p>
        {chatStatus && (
          <motion.div
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium bg-gray-100 dark:bg-gray-700"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200 }}
          >
            <motion.div
              className={`w-2 h-2 rounded-full ${chatStatus.openai_available ? 'bg-green-500' : 'bg-yellow-500'}`}
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
            <span className="text-gray-700 dark:text-gray-300">
              {chatStatus.openai_available ? `GPT-4o-mini` : 'Fallback Mode'}
            </span>
          </motion.div>
        )}
      </motion.div>

      {/* Chat Container */}
      <motion.div 
        className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
      >
        {/* Chat Messages */}
        <div className="h-96 overflow-y-auto p-6 space-y-4 bg-gray-50 dark:bg-gray-900">
          {messages.length === 0 ? (
            <motion.div 
              className="text-center py-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <MessageCircle className="w-16 h-16 mx-auto text-blue-600 dark:text-blue-400" />
              </motion.div>
              <p className="text-gray-500 dark:text-gray-400 text-lg mt-4">
                Start a conversation by typing a tweet or question below
              </p>
            </motion.div>
          ) : (
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  initial={{ opacity: 0, y: 20, scale: 0.8 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ type: "spring", stiffness: 200 }}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg shadow-sm ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white ml-auto'
                        : 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center space-x-2 mb-1">
                      {message.role === 'user' ? (
                        <User className="w-4 h-4" />
                      ) : (
                        <Bot className="w-4 h-4" />
                      )}
                      <span className="text-xs font-medium">
                        {message.role === 'user' ? 'You' : 'AI Assistant'}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed">{message.content}</p>
                    <p className="text-xs opacity-70 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white dark:bg-gray-700 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-600 max-w-xs lg:max-w-md px-4 py-3 rounded-lg shadow-sm">
                <div className="flex items-center space-x-2">
                  <span className="text-base">🤖</span>
                  <span className="text-xs font-medium">AI Assistant</span>
                </div>
                <div className="flex items-center space-x-2 mt-2">
                  <div className="animate-bounce w-2 h-2 bg-blue-600 rounded-full"></div>
                  <div className="animate-bounce w-2 h-2 bg-blue-600 rounded-full" style={{ animationDelay: '0.1s' }}></div>
                  <div className="animate-bounce w-2 h-2 bg-blue-600 rounded-full" style={{ animationDelay: '0.2s' }}></div>
                  <span className="text-sm text-gray-500 dark:text-gray-400">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex space-x-4">
            <div className="flex-1">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a tweet or question..."
                className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white resize-none"
                rows={2}
              />
            </div>
            <div className="flex flex-col space-y-2">
              <motion.button
                onClick={handleSendMessage}
                disabled={!inputText.trim() || loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white p-3 rounded-lg transition-colors duration-200 flex items-center justify-center"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Send className="w-5 h-5" />
              </motion.button>
              <motion.button
                onClick={clearChat}
                className="bg-red-600 hover:bg-red-700 text-white p-3 rounded-lg transition-colors duration-200 flex items-center justify-center"
                title="Clear chat"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Trash2 className="w-5 h-5" />
              </motion.button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Sample Questions */}
      <motion.div 
        className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-yellow-500" />
          Try these examples:
        </h3>
        <div className="grid md:grid-cols-2 gap-3">
          {sampleQuestions.map((question, index) => (
            <motion.button
              key={index}
              onClick={() => handleSampleQuestion(question)}
              className="text-left p-3 bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg border border-gray-200 dark:border-gray-600 transition-colors duration-200"
              whileHover={{ scale: 1.02, x: 5 }}
              whileTap={{ scale: 0.98 }}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.05 }}
            >
              <span className="text-sm text-gray-700 dark:text-gray-300 flex items-center gap-2">
                <MessageCircle className="w-4 h-4 flex-shrink-0" />
                {question.length > 50 ? question.substring(0, 50) + '...' : question}
              </span>
            </motion.button>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default SimpleAiAssistant;
