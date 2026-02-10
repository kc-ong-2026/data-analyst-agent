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

const BarChartComponent: React.FC<ChartProps> = ({ data }) => {
  // Safety check: return null if data is empty
  if (!data.data || data.data.length === 0) {
    return <p className="text-gray-500">No data available</p>;
  }

  // Calculate width based on number of data points (min 50px per bar)
  const chartWidth = Math.max(700, data.data.length * 50);
  const chartHeight = 450;

  return (
    <div style={{ width: chartWidth, height: chartHeight, minWidth: chartWidth }}>
      <BarChart
        width={chartWidth}
        height={chartHeight}
        data={data.data}
        margin={{ top: 30, right: 30, left: 150, bottom: 70 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey={data.x_axis}
          angle={-45}
          textAnchor="end"
          height={60}
          interval={0}
          tick={{ fontSize: 12 }}
          label={{
            value: data.x_label || data.x_axis || 'Category',
            position: 'insideBottom',
            offset: -5,
            style: { fontSize: 14, fontWeight: 600 }
          }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          label={{
            value: data.y_label || data.y_axis || 'Value',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 14, fontWeight: 600, textAnchor: 'middle' }
          }}
        />
        <Tooltip />
        <Bar dataKey={data.y_axis || 'value'} fill="#0ea5e9" />
      </BarChart>
    </div>
  );
};

const LineChartComponent: React.FC<ChartProps> = ({ data }) => {
  // Safety check: return null if data is empty
  if (!data.data || data.data.length === 0) {
    return <p className="text-gray-500">No data available</p>;
  }

  // Calculate width based on number of data points (min 40px per point)
  const chartWidth = Math.max(700, data.data.length * 40);
  const chartHeight = 450;

  return (
    <div style={{ width: chartWidth, height: chartHeight, minWidth: chartWidth }}>
      <LineChart
        width={chartWidth}
        height={chartHeight}
        data={data.data}
        margin={{ top: 30, right: 30, left: 150, bottom: 70 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey={data.x_axis}
          angle={-45}
          textAnchor="end"
          height={60}
          interval={0}
          tick={{ fontSize: 12 }}
          label={{
            value: data.x_label || data.x_axis || 'Category',
            position: 'insideBottom',
            offset: -5,
            style: { fontSize: 14, fontWeight: 600 }
          }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          label={{
            value: data.y_label || data.y_axis || 'Value',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 14, fontWeight: 600, textAnchor: 'middle' }
          }}
        />
        <Tooltip />
        <Line
          type="monotone"
          dataKey={data.y_axis || 'value'}
          stroke="#0ea5e9"
          strokeWidth={2}
          dot={{ fill: '#0ea5e9' }}
        />
      </LineChart>
    </div>
  );
};

const PieChartComponent: React.FC<ChartProps> = ({ data }) => {
  // Safety check: return null if data is empty
  if (!data.data || data.data.length === 0) {
    return <p className="text-gray-500">No data available</p>;
  }

  const chartWidth = 700;
  const chartHeight = 450;

  return (
    <div style={{ width: chartWidth, height: chartHeight, minWidth: chartWidth }}>
      <PieChart width={chartWidth} height={chartHeight}>
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
      </PieChart>
    </div>
  );
};

const ScatterChartComponent: React.FC<ChartProps> = ({ data }) => {
  // Safety check: return null if data is empty
  if (!data.data || data.data.length === 0) {
    return <p className="text-gray-500">No data available</p>;
  }

  const chartWidth = 900;
  const chartHeight = 450;

  return (
    <div style={{ width: chartWidth, height: chartHeight, minWidth: chartWidth }}>
      <ScatterChart
        width={chartWidth}
        height={chartHeight}
        margin={{ top: 30, right: 30, left: 150, bottom: 60 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey={data.x_axis}
          name={data.x_axis}
          tick={{ fontSize: 12 }}
          label={{
            value: data.x_axis || 'X',
            position: 'insideBottom',
            offset: -5,
            style: { fontSize: 14, fontWeight: 600 }
          }}
        />
        <YAxis
          dataKey={data.y_axis}
          name={data.y_axis}
          tick={{ fontSize: 12 }}
          label={{
            value: data.y_axis || 'Y',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 14, fontWeight: 600, textAnchor: 'middle' }
          }}
        />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Scatter name="Data" data={data.data} fill="#0ea5e9" />
      </ScatterChart>
    </div>
  );
};

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

// Component to render Plotly HTML with script execution using iframe
const PlotlyHTMLRenderer: React.FC<{ html: string }> = ({ html }) => {
  const iframeRef = React.useRef<HTMLIFrameElement>(null);

  React.useEffect(() => {
    if (!iframeRef.current || !html) return;

    const iframe = iframeRef.current;
    const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;

    if (iframeDoc) {
      iframeDoc.open();
      iframeDoc.write(html);
      iframeDoc.close();

      // Auto-resize iframe to fit content
      const resizeIframe = () => {
        if (iframeDoc.body) {
          const height = Math.max(
            iframeDoc.body.scrollHeight,
            iframeDoc.documentElement?.scrollHeight || 0,
            500 // minimum height
          );
          iframe.style.height = height + 'px';
        }
      };

      // Resize multiple times to ensure Plotly has finished rendering
      setTimeout(resizeIframe, 100);
      setTimeout(resizeIframe, 500);
      setTimeout(resizeIframe, 1000);
      setTimeout(resizeIframe, 2000);
    }
  }, [html]);

  return (
    <iframe
      ref={iframeRef}
      className="w-full border-0"
      style={{ minHeight: '500px', height: '500px' }}
      title="Plotly Visualization"
    />
  );
};

export const VisualizationPanel: React.FC = () => {
  const { currentVisualization, setVisualization } = useChatStore();

  const renderChart = () => {
    if (!currentVisualization) return null;

    // Priority 1: Render HTML chart if available (Plotly interactive chart)
    if (currentVisualization.html_chart) {
      return <PlotlyHTMLRenderer html={currentVisualization.html_chart} />;
    }

    // Priority 2: Fallback to Recharts (only if data array has content)
    if (!currentVisualization.data || currentVisualization.data.length === 0) {
      return (
        <div className="text-gray-500 text-center py-8">
          <p>Unable to generate visualization</p>
        </div>
      );
    }

    // Recharts fallback for backward compatibility
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
              {(currentVisualization.x_label || currentVisualization.x_axis) && (
                <span className="text-xs text-gray-500">
                  | X: {currentVisualization.x_label || currentVisualization.x_axis}
                </span>
              )}
              {(currentVisualization.y_label || currentVisualization.y_axis) && (
                <span className="text-xs text-gray-500">
                  | Y: {currentVisualization.y_label || currentVisualization.y_axis}
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
