import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
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
  ScatterController,
} from "chart.js";
import { Line, Scatter } from "react-chartjs-2";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ScatterController
);

// Jarvis color palette
const JARVIS_COLORS = {
  primary: "rgb(0, 212, 255)",
  primaryDim: "rgba(0, 212, 255, 0.3)",
  secondary: "rgb(0, 150, 200)",
  accent: "rgb(0, 255, 200)",
  warning: "rgb(255, 200, 0)",
  danger: "rgb(255, 80, 80)",
  grid: "rgba(0, 212, 255, 0.1)",
  text: "rgba(200, 230, 255, 0.8)",
};

interface ChartProps {
  className?: string;
  title?: string;
}

interface TimeSeriesChartProps extends ChartProps {
  data: number[];
  labels?: string[];
  yLabel?: string;
  xLabel?: string;
  fill?: boolean;
}

export function TimeSeriesChart({
  data,
  labels,
  yLabel = "Amplitude",
  xLabel = "Sample",
  fill = true,
  className,
  title,
}: TimeSeriesChartProps) {
  const chartLabels = labels || data.map((_, i) => i.toString());

  const chartData = {
    labels: chartLabels,
    datasets: [
      {
        label: yLabel,
        data: data,
        borderColor: JARVIS_COLORS.primary,
        backgroundColor: fill ? JARVIS_COLORS.primaryDim : "transparent",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: fill,
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0,
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: !!title,
        text: title,
        color: JARVIS_COLORS.text,
        font: {
          family: "Orbitron",
          size: 12,
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 20, 40, 0.9)",
        borderColor: JARVIS_COLORS.primary,
        borderWidth: 1,
        titleColor: JARVIS_COLORS.primary,
        bodyColor: JARVIS_COLORS.text,
        titleFont: {
          family: "Rajdhani",
        },
        bodyFont: {
          family: "Rajdhani",
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: xLabel,
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 10,
          },
        },
        grid: {
          color: JARVIS_COLORS.grid,
        },
        ticks: {
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 9,
          },
          maxTicksLimit: 10,
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: yLabel,
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 10,
          },
        },
        grid: {
          color: JARVIS_COLORS.grid,
        },
        ticks: {
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 9,
          },
        },
      },
    },
  };

  return (
    <div className={cn("h-full w-full", className)}>
      <Line data={chartData} options={options} />
    </div>
  );
}

interface FrequencySpectrumProps extends ChartProps {
  magnitudes: number[];
  frequencies?: number[];
  peakFrequency?: number;
}

export function FrequencySpectrum({
  magnitudes,
  frequencies,
  className,
  title,
}: FrequencySpectrumProps) {
  const freqLabels = frequencies || magnitudes.map((_, i) => i);

  const chartData = {
    labels: freqLabels,
    datasets: [
      {
        label: "Magnitude (dB)",
        data: magnitudes,
        borderColor: JARVIS_COLORS.primary,
        backgroundColor: JARVIS_COLORS.primaryDim,
        borderWidth: 1,
        pointRadius: 0,
        fill: true,
        tension: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0,
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: !!title,
        text: title,
        color: JARVIS_COLORS.text,
        font: {
          family: "Orbitron",
          size: 12,
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Frequency (Hz)",
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 10,
          },
        },
        grid: {
          color: JARVIS_COLORS.grid,
        },
        ticks: {
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 9,
          },
          maxTicksLimit: 10,
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Magnitude (dB)",
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 10,
          },
        },
        grid: {
          color: JARVIS_COLORS.grid,
        },
        ticks: {
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 9,
          },
        },
      },
    },
  };

  return (
    <div className={cn("h-full w-full", className)}>
      <Line data={chartData} options={options} />
    </div>
  );
}

interface ConstellationDiagramProps extends ChartProps {
  iData: number[];
  qData: number[];
}

export function ConstellationDiagram({
  iData,
  qData,
  className,
  title,
}: ConstellationDiagramProps) {
  const scatterData = iData.map((i, idx) => ({ x: i, y: qData[idx] }));

  const chartData = {
    datasets: [
      {
        label: "I/Q",
        data: scatterData,
        backgroundColor: JARVIS_COLORS.primary,
        borderColor: JARVIS_COLORS.primary,
        pointRadius: 2,
        pointHoverRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0,
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: !!title,
        text: title,
        color: JARVIS_COLORS.text,
        font: {
          family: "Orbitron",
          size: 12,
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "In-Phase (I)",
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 10,
          },
        },
        grid: {
          color: JARVIS_COLORS.grid,
        },
        ticks: {
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 9,
          },
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Quadrature (Q)",
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 10,
          },
        },
        grid: {
          color: JARVIS_COLORS.grid,
        },
        ticks: {
          color: JARVIS_COLORS.text,
          font: {
            family: "Rajdhani",
            size: 9,
          },
        },
      },
    },
  };

  return (
    <div className={cn("h-full w-full", className)}>
      <Scatter data={chartData} options={options} />
    </div>
  );
}

interface WaterfallDisplayProps extends ChartProps {
  data: number[][];
  maxRows?: number;
}

export function WaterfallDisplay({
  data,
  maxRows = 100,
  className,
  title,
}: WaterfallDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const rows = Math.min(data.length, maxRows);
    const cols = data[0].length;

    const cellWidth = width / cols;
    const cellHeight = height / rows;

    // Find min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (const row of data) {
      for (const val of row) {
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }
    const range = max - min || 1;

    // Draw waterfall
    ctx.clearRect(0, 0, width, height);
    
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const normalized = (data[row][col] - min) / range;
        const hue = 180 + normalized * 60; // Cyan to blue
        const lightness = 20 + normalized * 50;
        ctx.fillStyle = `hsl(${hue}, 100%, ${lightness}%)`;
        ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth + 1, cellHeight + 1);
      }
    }
  }, [data, maxRows]);

  return (
    <div className={cn("h-full w-full relative", className)}>
      {title && (
        <div className="absolute top-2 left-2 text-xs font-semibold uppercase tracking-wider jarvis-text">
          {title}
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={512}
        height={256}
        className="w-full h-full"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}

// Demo data generators for testing
export function generateDemoTimeSeries(length: number = 500): number[] {
  const data: number[] = [];
  for (let i = 0; i < length; i++) {
    const t = i / 50;
    const signal =
      Math.sin(2 * Math.PI * t) * 0.5 +
      Math.sin(2 * Math.PI * t * 3) * 0.3 +
      Math.sin(2 * Math.PI * t * 7) * 0.1 +
      (Math.random() - 0.5) * 0.1;
    data.push(signal);
  }
  return data;
}

export function generateDemoSpectrum(length: number = 256): number[] {
  const data: number[] = [];
  for (let i = 0; i < length; i++) {
    const freq = i / length;
    const peak1 = Math.exp(-Math.pow((freq - 0.2) * 20, 2)) * 60;
    const peak2 = Math.exp(-Math.pow((freq - 0.5) * 30, 2)) * 40;
    const noise = Math.random() * 5 - 60;
    data.push(Math.max(peak1 + peak2 + noise, -80));
  }
  return data;
}

export function generateDemoConstellation(length: number = 200): { i: number[]; q: number[] } {
  const iData: number[] = [];
  const qData: number[] = [];
  
  // Generate QPSK-like constellation
  const symbols = [
    { i: 1, q: 1 },
    { i: 1, q: -1 },
    { i: -1, q: 1 },
    { i: -1, q: -1 },
  ];
  
  for (let n = 0; n < length; n++) {
    const symbol = symbols[Math.floor(Math.random() * symbols.length)];
    iData.push(symbol.i + (Math.random() - 0.5) * 0.3);
    qData.push(symbol.q + (Math.random() - 0.5) * 0.3);
  }
  
  return { i: iData, q: qData };
}

export function generateDemoWaterfall(rows: number = 50, cols: number = 256): number[][] {
  const data: number[][] = [];
  for (let row = 0; row < rows; row++) {
    const rowData: number[] = [];
    for (let col = 0; col < cols; col++) {
      const freq = col / cols;
      const time = row / rows;
      const peak1 = Math.exp(-Math.pow((freq - 0.3 - time * 0.1) * 20, 2)) * 50;
      const peak2 = Math.exp(-Math.pow((freq - 0.6 + time * 0.05) * 25, 2)) * 35;
      const noise = Math.random() * 10 - 50;
      rowData.push(Math.max(peak1 + peak2 + noise, -60));
    }
    data.push(rowData);
  }
  return data;
}
