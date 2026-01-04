import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface BenchmarkRecord {
  id: number;
  createdAt: Date | string;
  avgSpeedup: number | null;
  maxSpeedup: number | null;
  minSpeedup: number | null;
  gpuName: string | null;
  success: number;
}

interface BenchmarkChartProps {
  history: BenchmarkRecord[];
}

export function BenchmarkChart({ history }: BenchmarkChartProps) {
  const chartData = useMemo(() => {
    // Filter successful benchmarks and reverse for chronological order
    const successfulRuns = history
      .filter(h => h.success === 1 && h.avgSpeedup !== null)
      .reverse();

    const labels = successfulRuns.map(h => {
      const date = new Date(h.createdAt);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });

    return {
      labels,
      datasets: [
        {
          label: 'Avg Speedup',
          data: successfulRuns.map(h => h.avgSpeedup),
          borderColor: 'rgb(0, 255, 255)',
          backgroundColor: 'rgba(0, 255, 255, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: 'Max Speedup',
          data: successfulRuns.map(h => h.maxSpeedup),
          borderColor: 'rgb(74, 222, 128)',
          backgroundColor: 'rgba(74, 222, 128, 0.1)',
          fill: false,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 5,
          borderDash: [5, 5],
        },
        {
          label: 'Min Speedup',
          data: successfulRuns.map(h => h.minSpeedup),
          borderColor: 'rgb(251, 146, 60)',
          backgroundColor: 'rgba(251, 146, 60, 0.1)',
          fill: false,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 5,
          borderDash: [2, 2],
        },
      ],
    };
  }, [history]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: 'rgba(255, 255, 255, 0.7)',
          font: {
            size: 11,
          },
          boxWidth: 12,
          padding: 10,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'rgb(0, 255, 255)',
        bodyColor: 'rgba(255, 255, 255, 0.9)',
        borderColor: 'rgba(0, 255, 255, 0.3)',
        borderWidth: 1,
        padding: 10,
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.y?.toFixed(1)}x`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.5)',
          font: {
            size: 10,
          },
        },
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.5)',
          font: {
            size: 10,
          },
          callback: function(value: any) {
            return value + 'x';
          },
        },
        beginAtZero: true,
      },
    },
  };

  if (history.filter(h => h.success === 1).length < 2) {
    return (
      <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
        Run at least 2 benchmarks to see trend chart
      </div>
    );
  }

  return (
    <div className="h-48">
      <Line data={chartData} options={options} />
    </div>
  );
}
