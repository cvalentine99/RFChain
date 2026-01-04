import { JarvisLayout } from "@/components/JarvisLayout";
import { JarvisPanel } from "@/components/JarvisPanel";
import { GitCompare, ArrowRight, ArrowUp, ArrowDown, Minus, ChevronDown, Loader2 } from "lucide-react";
import { trpc } from "@/lib/trpc";
import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type AnalysisResult = {
  id: number;
  signalId: number;
  avgPowerDbm: number | null;
  peakPowerDbm: number | null;
  paprDb: number | null;
  snrEstimateDb: number | null;
  sampleCount: number | null;
  bandwidthHz: number | null;
  freqOffsetHz: number | null;
  iqImbalanceDb: number | null;
  dcOffsetReal: number | null;
  dcOffsetImag: number | null;
  anomalies: Record<string, boolean> | null;
  plotUrls: Record<string, string> | null;
  fullMetrics: Record<string, unknown> | null;
  createdAt: Date;
};

export default function Compare() {
  const { data: analyses, isLoading } = trpc.analysis.getRecent.useQuery({ limit: 50 });
  const { data: uploads } = trpc.signal.getUploads.useQuery({ limit: 50 });
  
  const [leftId, setLeftId] = useState<number | null>(null);
  const [rightId, setRightId] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"sideBySide" | "overlay">("sideBySide");

  // Get filename for an analysis
  const getFilename = (signalId: number) => {
    const upload = uploads?.find((u: { id: number; originalName: string }) => u.id === signalId);
    return upload?.originalName || `Signal #${signalId}`;
  };

  // Get selected analyses
  const leftAnalysis = analyses?.find((a: { id: number }) => a.id === leftId) as AnalysisResult | undefined;
  const rightAnalysis = analyses?.find((a: { id: number }) => a.id === rightId) as AnalysisResult | undefined;

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h2 className="text-2xl font-bold tracking-wider jarvis-text">SIGNAL COMPARISON</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Compare two signal analyses side-by-side to identify differences
          </p>
        </div>

        {/* Selection Panel */}
        <JarvisPanel title="Select Signals to Compare" glowPulse>
          <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-center">
            {/* Left Signal Selector */}
            <SignalSelector
              label="Signal A"
              analyses={analyses || []}
              uploads={uploads || []}
              selectedId={leftId}
              onSelect={setLeftId}
              isLoading={isLoading}
              excludeId={rightId}
            />

            {/* Compare Arrow */}
            <div className="hidden md:flex items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center">
                <GitCompare className="w-6 h-6 text-primary" />
              </div>
            </div>

            {/* Right Signal Selector */}
            <SignalSelector
              label="Signal B"
              analyses={analyses || []}
              uploads={uploads || []}
              selectedId={rightId}
              onSelect={setRightId}
              isLoading={isLoading}
              excludeId={leftId}
            />
          </div>
        </JarvisPanel>

        {/* Comparison Results */}
        {leftAnalysis && rightAnalysis ? (
          <>
            {/* View Mode Toggle */}
            <div className="flex justify-end gap-2">
              <Button
                variant={viewMode === "sideBySide" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("sideBySide")}
              >
                Side by Side
              </Button>
              <Button
                variant={viewMode === "overlay" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("overlay")}
              >
                Overlay
              </Button>
            </div>

            {/* Metric Comparison */}
            <MetricComparison
              left={leftAnalysis}
              right={rightAnalysis}
              leftName={getFilename(leftAnalysis.signalId)}
              rightName={getFilename(rightAnalysis.signalId)}
            />

            {/* Visualization Comparison */}
            <VisualizationComparison
              left={leftAnalysis}
              right={rightAnalysis}
              leftName={getFilename(leftAnalysis.signalId)}
              rightName={getFilename(rightAnalysis.signalId)}
              viewMode={viewMode}
            />

            {/* Comparison Summary */}
            <ComparisonSummary
              left={leftAnalysis}
              right={rightAnalysis}
              leftName={getFilename(leftAnalysis.signalId)}
              rightName={getFilename(rightAnalysis.signalId)}
            />
          </>
        ) : (
          <JarvisPanel title="Comparison Results">
            <div className="text-center py-12">
              <GitCompare className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">Select two signals to compare</p>
              <p className="text-sm text-muted-foreground mt-1">
                Choose Signal A and Signal B from the dropdowns above
              </p>
            </div>
          </JarvisPanel>
        )}
      </div>
    </JarvisLayout>
  );
}

function SignalSelector({
  label,
  analyses,
  uploads,
  selectedId,
  onSelect,
  isLoading,
  excludeId,
}: {
  label: string;
  analyses: { id: number; signalId: number; createdAt: Date }[];
  uploads: { id: number; originalName: string }[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  isLoading: boolean;
  excludeId: number | null;
}) {
  const [isOpen, setIsOpen] = useState(false);

  const getFilename = (signalId: number) => {
    const upload = uploads.find(u => u.id === signalId);
    return upload?.originalName || `Signal #${signalId}`;
  };

  const selectedAnalysis = analyses.find(a => a.id === selectedId);
  const filteredAnalyses = analyses.filter(a => a.id !== excludeId);

  return (
    <div className="space-y-2">
      <label className="text-xs uppercase tracking-wider text-muted-foreground">{label}</label>
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full p-4 bg-secondary/30 rounded-lg border border-border/50 hover:border-primary/50 transition-all text-left flex items-center justify-between"
        >
          {isLoading ? (
            <span className="text-muted-foreground flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading...
            </span>
          ) : selectedAnalysis ? (
            <div>
              <p className="font-medium">{getFilename(selectedAnalysis.signalId)}</p>
              <p className="text-xs text-muted-foreground">
                Analysis #{selectedAnalysis.id} • {new Date(selectedAnalysis.createdAt).toLocaleDateString()}
              </p>
            </div>
          ) : (
            <span className="text-muted-foreground">Select a signal...</span>
          )}
          <ChevronDown className={cn("w-4 h-4 transition-transform", isOpen && "rotate-180")} />
        </button>

        {isOpen && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-background border border-border rounded-lg shadow-lg z-50 max-h-64 overflow-auto">
            {filteredAnalyses.length === 0 ? (
              <div className="p-4 text-center text-muted-foreground">
                No analyses available
              </div>
            ) : (
              filteredAnalyses.map((analysis) => (
                <button
                  key={analysis.id}
                  onClick={() => {
                    onSelect(analysis.id);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "w-full p-3 text-left hover:bg-secondary/50 transition-colors border-b border-border/30 last:border-0",
                    selectedId === analysis.id && "bg-primary/10"
                  )}
                >
                  <p className="font-medium">{getFilename(analysis.signalId)}</p>
                  <p className="text-xs text-muted-foreground">
                    Analysis #{analysis.id} • {new Date(analysis.createdAt).toLocaleDateString()}
                  </p>
                </button>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricComparison({
  left,
  right,
  leftName,
  rightName,
}: {
  left: AnalysisResult;
  right: AnalysisResult;
  leftName: string;
  rightName: string;
}) {
  const metrics = [
    { key: "avgPowerDbm", label: "Avg Power", unit: "dBm", precision: 2 },
    { key: "peakPowerDbm", label: "Peak Power", unit: "dBm", precision: 2 },
    { key: "paprDb", label: "PAPR", unit: "dB", precision: 2 },
    { key: "snrEstimateDb", label: "SNR Estimate", unit: "dB", precision: 2 },
    { key: "bandwidthHz", label: "Bandwidth", unit: "Hz", precision: 0, format: "frequency" },
    { key: "freqOffsetHz", label: "Freq Offset", unit: "Hz", precision: 2 },
    { key: "iqImbalanceDb", label: "I/Q Imbalance", unit: "dB", precision: 3 },
    { key: "sampleCount", label: "Sample Count", unit: "", precision: 0, format: "number" },
  ];

  const formatValue = (value: number | null, metric: typeof metrics[0]) => {
    if (value === null) return "N/A";
    if (metric.format === "frequency") {
      if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)} MHz`;
      if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(2)} kHz`;
      return `${value.toFixed(metric.precision)} Hz`;
    }
    if (metric.format === "number") {
      return value.toLocaleString();
    }
    return `${value.toFixed(metric.precision)} ${metric.unit}`;
  };

  const calculateDiff = (leftVal: number | null, rightVal: number | null) => {
    if (leftVal === null || rightVal === null) return null;
    if (leftVal === 0 && rightVal === 0) return 0;
    if (leftVal === 0) return 100;
    return ((rightVal - leftVal) / Math.abs(leftVal)) * 100;
  };

  return (
    <JarvisPanel title="Metric Comparison">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/50">
              <th className="text-left py-3 px-4 text-xs uppercase tracking-wider text-muted-foreground">Metric</th>
              <th className="text-right py-3 px-4 text-xs uppercase tracking-wider text-primary">{leftName}</th>
              <th className="text-center py-3 px-4 text-xs uppercase tracking-wider text-muted-foreground">Diff</th>
              <th className="text-left py-3 px-4 text-xs uppercase tracking-wider text-cyan-400">{rightName}</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => {
              const leftVal = left[metric.key as keyof AnalysisResult] as number | null;
              const rightVal = right[metric.key as keyof AnalysisResult] as number | null;
              const diff = calculateDiff(leftVal, rightVal);
              const isSignificant = diff !== null && Math.abs(diff) > 10;

              return (
                <tr key={metric.key} className="border-b border-border/30 hover:bg-secondary/20">
                  <td className="py-3 px-4 font-medium">{metric.label}</td>
                  <td className="py-3 px-4 text-right font-mono text-primary">
                    {formatValue(leftVal, metric)}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <DiffIndicator diff={diff} isSignificant={isSignificant} />
                  </td>
                  <td className="py-3 px-4 text-left font-mono text-cyan-400">
                    {formatValue(rightVal, metric)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </JarvisPanel>
  );
}

function DiffIndicator({ diff, isSignificant }: { diff: number | null; isSignificant: boolean }) {
  if (diff === null) {
    return <span className="text-muted-foreground">—</span>;
  }

  const absPercentage = Math.abs(diff).toFixed(1);
  
  if (Math.abs(diff) < 0.1) {
    return (
      <span className="inline-flex items-center gap-1 text-muted-foreground">
        <Minus className="w-3 h-3" />
        <span className="text-xs">0%</span>
      </span>
    );
  }

  if (diff > 0) {
    return (
      <span className={cn(
        "inline-flex items-center gap-1",
        isSignificant ? "text-green-400" : "text-muted-foreground"
      )}>
        <ArrowUp className="w-3 h-3" />
        <span className="text-xs">+{absPercentage}%</span>
      </span>
    );
  }

  return (
    <span className={cn(
      "inline-flex items-center gap-1",
      isSignificant ? "text-red-400" : "text-muted-foreground"
    )}>
      <ArrowDown className="w-3 h-3" />
      <span className="text-xs">-{absPercentage}%</span>
    </span>
  );
}

function VisualizationComparison({
  left,
  right,
  leftName,
  rightName,
  viewMode,
}: {
  left: AnalysisResult;
  right: AnalysisResult;
  leftName: string;
  rightName: string;
  viewMode: "sideBySide" | "overlay";
}) {
  const plotTypes = [
    { key: "time_domain", label: "Time Domain" },
    { key: "frequency_domain", label: "Frequency Domain" },
    { key: "spectrogram", label: "Spectrogram" },
    { key: "constellation", label: "Constellation" },
  ];

  const leftPlots = (left.fullMetrics as { plotUrls?: Record<string, string> } | null)?.plotUrls || {};
  const rightPlots = (right.fullMetrics as { plotUrls?: Record<string, string> } | null)?.plotUrls || {};

  return (
    <JarvisPanel title="Visualization Comparison">
      <div className="space-y-6">
        {plotTypes.map((plot) => {
          const leftUrl = leftPlots[plot.key];
          const rightUrl = rightPlots[plot.key];

          if (!leftUrl && !rightUrl) return null;

          return (
            <div key={plot.key} className="space-y-2">
              <h4 className="text-sm uppercase tracking-wider text-muted-foreground">{plot.label}</h4>
              
              {viewMode === "sideBySide" ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-xs text-primary">{leftName}</p>
                    {leftUrl ? (
                      <img
                        src={leftUrl}
                        alt={`${leftName} - ${plot.label}`}
                        className="w-full rounded-lg border border-border/50"
                      />
                    ) : (
                      <div className="h-48 bg-secondary/30 rounded-lg flex items-center justify-center text-muted-foreground">
                        No visualization
                      </div>
                    )}
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-cyan-400">{rightName}</p>
                    {rightUrl ? (
                      <img
                        src={rightUrl}
                        alt={`${rightName} - ${plot.label}`}
                        className="w-full rounded-lg border border-border/50"
                      />
                    ) : (
                      <div className="h-48 bg-secondary/30 rounded-lg flex items-center justify-center text-muted-foreground">
                        No visualization
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="relative">
                  {leftUrl && (
                    <img
                      src={leftUrl}
                      alt={`${leftName} - ${plot.label}`}
                      className="w-full rounded-lg border border-primary/50"
                    />
                  )}
                  {rightUrl && (
                    <img
                      src={rightUrl}
                      alt={`${rightName} - ${plot.label}`}
                      className="absolute inset-0 w-full rounded-lg border border-cyan-400/50 opacity-50 mix-blend-screen"
                    />
                  )}
                  <div className="absolute bottom-2 right-2 flex gap-2">
                    <span className="px-2 py-1 bg-primary/80 text-xs rounded">{leftName}</span>
                    <span className="px-2 py-1 bg-cyan-400/80 text-xs rounded text-black">{rightName}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </JarvisPanel>
  );
}

function ComparisonSummary({
  left,
  right,
  leftName,
  rightName,
}: {
  left: AnalysisResult;
  right: AnalysisResult;
  leftName: string;
  rightName: string;
}) {
  const findings = useMemo(() => {
    const results: { type: "info" | "warning" | "success"; message: string }[] = [];

    // Power comparison
    if (left.avgPowerDbm !== null && right.avgPowerDbm !== null) {
      const powerDiff = right.avgPowerDbm - left.avgPowerDbm;
      if (Math.abs(powerDiff) > 3) {
        results.push({
          type: powerDiff > 0 ? "warning" : "info",
          message: `Signal B has ${Math.abs(powerDiff).toFixed(1)} dB ${powerDiff > 0 ? "higher" : "lower"} average power`,
        });
      }
    }

    // Bandwidth comparison
    if (left.bandwidthHz !== null && right.bandwidthHz !== null) {
      const bwRatio = right.bandwidthHz / left.bandwidthHz;
      if (bwRatio > 1.2 || bwRatio < 0.8) {
        results.push({
          type: "info",
          message: `Bandwidth differs by ${((bwRatio - 1) * 100).toFixed(0)}% (${(left.bandwidthHz / 1000).toFixed(1)} kHz vs ${(right.bandwidthHz / 1000).toFixed(1)} kHz)`,
        });
      }
    }

    // PAPR comparison
    if (left.paprDb !== null && right.paprDb !== null) {
      const paprDiff = Math.abs(right.paprDb - left.paprDb);
      if (paprDiff > 2) {
        results.push({
          type: "warning",
          message: `PAPR differs by ${paprDiff.toFixed(1)} dB - signals may have different modulation characteristics`,
        });
      }
    }

    // I/Q imbalance comparison
    if (left.iqImbalanceDb !== null && right.iqImbalanceDb !== null) {
      const iqDiff = Math.abs(right.iqImbalanceDb - left.iqImbalanceDb);
      if (iqDiff > 1) {
        results.push({
          type: "warning",
          message: `I/Q imbalance differs significantly (${left.iqImbalanceDb.toFixed(2)} dB vs ${right.iqImbalanceDb.toFixed(2)} dB)`,
        });
      }
    }

    // Anomaly comparison
    const leftAnomalies = left.anomalies || {};
    const rightAnomalies = right.anomalies || {};
    const leftHasAnomalies = Object.values(leftAnomalies).some(v => v);
    const rightHasAnomalies = Object.values(rightAnomalies).some(v => v);

    if (leftHasAnomalies && !rightHasAnomalies) {
      results.push({
        type: "success",
        message: `Signal A has anomalies but Signal B is clean`,
      });
    } else if (!leftHasAnomalies && rightHasAnomalies) {
      results.push({
        type: "warning",
        message: `Signal B has anomalies but Signal A is clean`,
      });
    } else if (leftHasAnomalies && rightHasAnomalies) {
      results.push({
        type: "warning",
        message: `Both signals have detected anomalies`,
      });
    } else {
      results.push({
        type: "success",
        message: `No anomalies detected in either signal`,
      });
    }

    // Sample count comparison
    if (left.sampleCount !== null && right.sampleCount !== null) {
      if (left.sampleCount !== right.sampleCount) {
        results.push({
          type: "info",
          message: `Different sample counts: ${left.sampleCount.toLocaleString()} vs ${right.sampleCount.toLocaleString()}`,
        });
      }
    }

    return results;
  }, [left, right]);

  return (
    <JarvisPanel title="Comparison Summary">
      <div className="space-y-3">
        {findings.map((finding, index) => (
          <div
            key={index}
            className={cn(
              "p-3 rounded-lg border",
              finding.type === "success" && "bg-green-500/10 border-green-500/30",
              finding.type === "warning" && "bg-yellow-500/10 border-yellow-500/30",
              finding.type === "info" && "bg-blue-500/10 border-blue-500/30"
            )}
          >
            <p className={cn(
              "text-sm",
              finding.type === "success" && "text-green-400",
              finding.type === "warning" && "text-yellow-400",
              finding.type === "info" && "text-blue-400"
            )}>
              {finding.message}
            </p>
          </div>
        ))}
      </div>
    </JarvisPanel>
  );
}
