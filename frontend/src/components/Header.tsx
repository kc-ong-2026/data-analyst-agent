import React from 'react';

export const Header: React.FC = () => {
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
      </div>
    </header>
  );
};
