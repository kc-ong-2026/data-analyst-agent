import React from 'react';
import { CheckCircle2, Loader2, Circle } from 'lucide-react';
import { useChatStore } from '../store/chatStore';

export const AgentStatusIndicator: React.FC = () => {
  const { agentStages, currentStage, isLoading } = useChatStore();

  if (!isLoading && agentStages.length === 0) return null;

  const stages = [
    { key: 'verification', label: 'Verifying' },
    { key: 'coordinator', label: 'Planning' },
    { key: 'extraction', label: 'Extracting' },
    { key: 'analytics', label: 'Analyzing' },
  ];

  const getStageStatus = (stageKey: string): 'pending' | 'running' | 'complete' | 'error' => {
    const stage = agentStages.find(s => s.agent === stageKey);
    if (!stage) return 'pending';
    return stage.status;
  };

  return (
    <div className="flex gap-4 p-4 bg-gray-50 rounded-lg">
      {/* Current stage indicator */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-700 text-white
                      flex items-center justify-center">
        <Loader2 size={18} className="animate-spin" />
      </div>

      <div className="flex-1">
        <div className="font-medium text-sm text-gray-900 mb-2">
          {currentStage || 'Processing...'}
        </div>

        {/* Progress pipeline */}
        <div className="flex items-center gap-2 flex-wrap">
          {stages.map((stage, index) => {
            const status = getStageStatus(stage.key);

            return (
              <div key={stage.key} className="flex items-center gap-2">
                {/* Stage icon */}
                {status === 'complete' && (
                  <CheckCircle2 className="w-4 h-4 text-green-600" />
                )}
                {status === 'running' && (
                  <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                )}
                {status === 'pending' && (
                  <Circle className="w-4 h-4 text-gray-300" />
                )}
                {status === 'error' && (
                  <Circle className="w-4 h-4 text-red-600" />
                )}

                {/* Stage label */}
                <span className={`text-xs whitespace-nowrap ${
                  status === 'complete' ? 'text-green-700' :
                  status === 'running' ? 'text-blue-700 font-medium' :
                  status === 'error' ? 'text-red-700' :
                  'text-gray-400'
                }`}>
                  {stage.label}
                </span>

                {/* Connector */}
                {index < stages.length - 1 && (
                  <div className={`w-6 h-0.5 ${
                    status === 'complete' ? 'bg-green-300' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
