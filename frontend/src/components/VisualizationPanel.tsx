import React from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { BarChart2, X, Table } from 'lucide-react';
import { useChatStore } from '../store/chatStore';
import type { VisualizationData } from '../types';

const COLORS = [
  '#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

interface ChartProps {
  data: VisualizationData;
}

const BarChartComponent: React.FC<ChartProps> = ({ data }) => (
  <ResponsiveContainer width="100%" height={400}>
    <BarChart data={data.data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis
        dataKey={data.x_axis}
        angle={-45}
        textAnchor="end"
        height={80}
        interval={0}
        tick={{ fontSize: 12 }}
      />
      <YAxis tick={{ fontSize: 12 }} />
      <Tooltip />
      <Legend />
      <Bar dataKey={data.y_axis || 'value'} fill="#0ea5e9" />
    </BarChart>
  </ResponsiveContainer>
);

const LineChartComponent: React.FC<ChartProps> = ({ data }) => (
  <ResponsiveContainer width="100%" height={400}>
    <LineChart data={data.data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis
        dataKey={data.x_axis}
        angle={-45}
        textAnchor="end"
        height={80}
        tick={{ fontSize: 12 }}
      />
      <YAxis tick={{ fontSize: 12 }} />
      <Tooltip />
      <Legend />
      <Line
        type="monotone"
        dataKey={data.y_axis || 'value'}
        stroke="#0ea5e9"
        strokeWidth={2}
        dot={{ fill: '#0ea5e9' }}
      />
    </LineChart>
  </ResponsiveContainer>
);

const PieChartComponent: React.FC<ChartProps> = ({ data }) => (
  <ResponsiveContainer width="100%" height={400}>
    <PieChart>
      <Pie
        data={data.data}
        dataKey={data.y_axis || 'value'}
        nameKey={data.x_axis || 'name'}
        cx="50%"
        cy="50%"
        outerRadius={150}
        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
      >
        {data.data.map((_, index) => (
          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
        ))}
      </Pie>
      <Tooltip />
      <Legend />
    </PieChart>
  </ResponsiveContainer>
);

const ScatterChartComponent: React.FC<ChartProps> = ({ data }) => (
  <ResponsiveContainer width="100%" height={400}>
    <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey={data.x_axis} name={data.x_axis} tick={{ fontSize: 12 }} />
      <YAxis dataKey={data.y_axis} name={data.y_axis} tick={{ fontSize: 12 }} />
      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
      <Legend />
      <Scatter name="Data" data={data.data} fill="#0ea5e9" />
    </ScatterChart>
  </ResponsiveContainer>
);

const TableComponent: React.FC<ChartProps> = ({ data }) => {
  if (!data.data.length) return <p className="text-gray-500">No data available</p>;

  const columns = Object.keys(data.data[0]);

  return (
    <div className="overflow-auto max-h-[400px]">
      <table className="w-full border-collapse">
        <thead className="sticky top-0 bg-gray-100">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="border border-gray-200 px-3 py-2 text-left text-sm font-semibold"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.data.map((row, i) => (
            <tr key={i} className="hover:bg-gray-50">
              {columns.map((col) => (
                <td key={col} className="border border-gray-200 px-3 py-2 text-sm">
                  {String(row[col] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const VisualizationPanel: React.FC = () => {
  const { currentVisualization, setVisualization } = useChatStore();

  const renderChart = () => {
    if (!currentVisualization) return null;

    switch (currentVisualization.chart_type) {
      case 'bar':
        return <BarChartComponent data={currentVisualization} />;
      case 'line':
        return <LineChartComponent data={currentVisualization} />;
      case 'pie':
        return <PieChartComponent data={currentVisualization} />;
      case 'scatter':
        return <ScatterChartComponent data={currentVisualization} />;
      case 'table':
        return <TableComponent data={currentVisualization} />;
      default:
        return <BarChartComponent data={currentVisualization} />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-2">
          <BarChart2 className="text-primary-600" size={20} />
          <h2 className="font-semibold text-gray-800">Visualization</h2>
        </div>

        {currentVisualization && (
          <button
            onClick={() => setVisualization(null)}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100
                       rounded-lg transition-colors"
            title="Clear visualization"
          >
            <X size={18} />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {currentVisualization ? (
          <div className="bg-white rounded-xl p-4 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              {currentVisualization.title}
            </h3>

            {currentVisualization.description && (
              <p className="text-sm text-gray-600 mb-4">
                {currentVisualization.description}
              </p>
            )}

            {renderChart()}

            <div className="mt-4 flex gap-2">
              <span className="text-xs text-gray-500">
                Chart type: {currentVisualization.chart_type}
              </span>
              {currentVisualization.x_axis && (
                <span className="text-xs text-gray-500">
                  | X: {currentVisualization.x_axis}
                </span>
              )}
              {currentVisualization.y_axis && (
                <span className="text-xs text-gray-500">
                  | Y: {currentVisualization.y_axis}
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Table size={48} className="mb-4 text-gray-300" />
            <h3 className="text-lg font-medium mb-2">No visualization</h3>
            <p className="text-sm text-center max-w-sm">
              Ask questions about data and visualizations will appear here when available.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
