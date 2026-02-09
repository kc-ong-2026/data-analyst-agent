import React, { useState, useEffect } from 'react';
import { Settings, ChevronDown } from 'lucide-react';
import { useChatStore } from '../store/chatStore';
import { configApi } from '../api/client';
import type { ProvidersResponse } from '../types';

export const Header: React.FC = () => {
  const { selectedProvider, selectedModel, setProvider, setModel } = useChatStore();
  const [showSettings, setShowSettings] = useState(false);
  const [providers, setProviders] = useState<ProvidersResponse | null>(null);

  useEffect(() => {
    configApi.getProviders()
      .then(setProviders)
      .catch(console.error);
  }, []);

  const handleProviderChange = (provider: string) => {
    setProvider(provider);
    // Set default model for the provider
    if (providers?.llm_providers[provider]) {
      setModel(providers.llm_providers[provider].default_model);
    }
  };

  const availableModels = providers?.llm_providers[selectedProvider]?.models || [];

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">G</span>
          </div>
          <div>
            <h1 className="font-semibold text-gray-900">Govtech Chat Assistant</h1>
            <p className="text-xs text-gray-500">Singapore Government Data Analysis</p>
          </div>
        </div>

        <div className="relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600
                       hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Settings size={18} />
            <span className="hidden sm:inline">
              {selectedProvider} / {selectedModel.split('-').slice(0, 2).join('-')}
            </span>
            <ChevronDown size={16} />
          </button>

          {showSettings && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setShowSettings(false)}
              />
              <div className="absolute right-0 top-full mt-2 w-72 bg-white rounded-xl
                              shadow-lg border border-gray-200 z-20 p-4">
                <h3 className="font-medium text-gray-900 mb-3">Model Settings</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Provider
                    </label>
                    <select
                      value={selectedProvider}
                      onChange={(e) => handleProviderChange(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg
                                 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      {providers && Object.entries(providers.llm_providers).map(([key, val]) => (
                        <option key={key} value={key} disabled={!val.has_api_key}>
                          {key} {!val.has_api_key && '(no API key)'}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Model
                    </label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg
                                 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      {availableModels.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-gray-200">
                  <p className="text-xs text-gray-500">
                    Configure API keys in the backend .env file
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
