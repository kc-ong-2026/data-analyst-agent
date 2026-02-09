import React from 'react';
import { Header } from './components/Header';
import { ChatPanel } from './components/ChatPanel';
import { VisualizationPanel } from './components/VisualizationPanel';

const App: React.FC = () => {
  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <Header />

      <main className="flex-1 flex overflow-hidden">
        {/* Chat Panel - Left Side */}
        <div className="w-1/2 border-r border-gray-200 flex flex-col">
          <ChatPanel />
        </div>

        {/* Visualization Panel - Right Side */}
        <div className="w-1/2 flex flex-col">
          <VisualizationPanel />
        </div>
      </main>
    </div>
  );
};

export default App;
