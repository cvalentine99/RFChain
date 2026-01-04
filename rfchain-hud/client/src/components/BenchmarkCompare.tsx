import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { GitCompare, Check, ArrowUp, ArrowDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BenchmarkRecord {
  id: number;
  createdAt: Date | string;
  avgSpeedup: number | null;
  maxSpeedup: number | null;
  minSpeedup: number | null;
  gpuName: string | null;
  gpuMemoryMb: number | null;
  benchmarkResults: any;
  success: number;
}

interface BenchmarkCompareProps {
  history: BenchmarkRecord[];
}

export function BenchmarkCompare({ history }: BenchmarkCompareProps) {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  const successfulRuns = history.filter(h => h.success === 1);

  const toggleSelection = (id: number) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(i => i !== id));
    } else if (selectedIds.length < 2) {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const getComparison = () => {
    if (selectedIds.length !== 2) return null;
    
    const [first, second] = selectedIds.map(id => 
      history.find(h => h.id === id)
    ).filter(Boolean) as BenchmarkRecord[];

    if (!first || !second) return null;

    // Determine which is older (baseline) and which is newer
    const baseline = new Date(first.createdAt) < new Date(second.createdAt) ? first : second;
    const current = baseline === first ? second : first;

    return { baseline, current };
  };

  const comparison = getComparison();

  const getDiffIndicator = (baseline: number | null, current: number | null) => {
    if (baseline === null || current === null) return null;
    const diff = ((current - baseline) / baseline) * 100;
    
    if (Math.abs(diff) < 1) {
      return <span className="flex items-center text-muted-foreground"><Minus className="w-3 h-3 mr-1" /> No change</span>;
    }
    if (diff > 0) {
      return <span className="flex items-center text-green-400"><ArrowUp className="w-3 h-3 mr-1" /> +{diff.toFixed(1)}%</span>;
    }
    return <span className="flex items-center text-red-400"><ArrowDown className="w-3 h-3 mr-1" /> {diff.toFixed(1)}%</span>;
  };

  const ComparisonRow = ({ 
    label, 
    baselineValue, 
    currentValue, 
    format = (v: number) => `${v.toFixed(2)}x` 
  }: { 
    label: string; 
    baselineValue: number | null; 
    currentValue: number | null;
    format?: (v: number) => string;
  }) => (
    <div className="grid grid-cols-4 gap-4 py-3 border-b border-border/30">
      <div className="font-medium text-muted-foreground">{label}</div>
      <div className="text-center font-mono">
        {baselineValue !== null ? format(baselineValue) : '-'}
      </div>
      <div className="text-center font-mono text-primary">
        {currentValue !== null ? format(currentValue) : '-'}
      </div>
      <div className="text-center text-sm">
        {getDiffIndicator(baselineValue, currentValue)}
      </div>
    </div>
  );

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <GitCompare className="w-4 h-4" />
          Compare
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-primary" />
            Compare Benchmarks
          </DialogTitle>
        </DialogHeader>

        {!comparison ? (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Select 2 benchmarks to compare ({selectedIds.length}/2 selected)
            </p>
            
            <div className="max-h-60 overflow-y-auto space-y-2">
              {successfulRuns.map(record => (
                <button
                  key={record.id}
                  onClick={() => toggleSelection(record.id)}
                  className={cn(
                    "w-full p-3 rounded-lg text-left transition-all flex items-center justify-between",
                    selectedIds.includes(record.id)
                      ? "bg-primary/20 border border-primary"
                      : "bg-secondary/30 border border-transparent hover:border-primary/50"
                  )}
                >
                  <div>
                    <div className="font-medium">
                      {new Date(record.createdAt).toLocaleString()}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {record.gpuName} • Avg: {record.avgSpeedup?.toFixed(1)}x
                    </div>
                  </div>
                  {selectedIds.includes(record.id) && (
                    <Check className="w-5 h-5 text-primary" />
                  )}
                </button>
              ))}
            </div>

            {successfulRuns.length < 2 && (
              <p className="text-sm text-muted-foreground text-center py-4">
                Run at least 2 successful benchmarks to compare
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Header */}
            <div className="grid grid-cols-4 gap-4 pb-2 border-b border-border">
              <div className="font-semibold">Metric</div>
              <div className="text-center font-semibold text-muted-foreground">Baseline</div>
              <div className="text-center font-semibold text-primary">Current</div>
              <div className="text-center font-semibold">Change</div>
            </div>

            {/* Date info */}
            <div className="grid grid-cols-4 gap-4 text-xs text-muted-foreground pb-2">
              <div></div>
              <div className="text-center">
                {new Date(comparison.baseline.createdAt).toLocaleDateString()}
              </div>
              <div className="text-center">
                {new Date(comparison.current.createdAt).toLocaleDateString()}
              </div>
              <div></div>
            </div>

            {/* Comparison rows */}
            <ComparisonRow 
              label="Avg Speedup" 
              baselineValue={comparison.baseline.avgSpeedup} 
              currentValue={comparison.current.avgSpeedup} 
            />
            <ComparisonRow 
              label="Max Speedup" 
              baselineValue={comparison.baseline.maxSpeedup} 
              currentValue={comparison.current.maxSpeedup} 
            />
            <ComparisonRow 
              label="Min Speedup" 
              baselineValue={comparison.baseline.minSpeedup} 
              currentValue={comparison.current.minSpeedup} 
            />
            <ComparisonRow 
              label="GPU Memory" 
              baselineValue={comparison.baseline.gpuMemoryMb} 
              currentValue={comparison.current.gpuMemoryMb}
              format={(v) => `${(v / 1024).toFixed(1)} GB`}
            />

            {/* Summary */}
            <div className="mt-4 p-4 bg-secondary/30 rounded-lg">
              <h4 className="font-medium mb-2">Summary</h4>
              {comparison.current.avgSpeedup && comparison.baseline.avgSpeedup && (
                <p className="text-sm text-muted-foreground">
                  {comparison.current.avgSpeedup > comparison.baseline.avgSpeedup ? (
                    <span className="text-green-400">
                      Performance improved by {((comparison.current.avgSpeedup - comparison.baseline.avgSpeedup) / comparison.baseline.avgSpeedup * 100).toFixed(1)}%
                    </span>
                  ) : comparison.current.avgSpeedup < comparison.baseline.avgSpeedup ? (
                    <span className="text-red-400">
                      Performance decreased by {((comparison.baseline.avgSpeedup - comparison.current.avgSpeedup) / comparison.baseline.avgSpeedup * 100).toFixed(1)}%
                    </span>
                  ) : (
                    <span>Performance remained stable</span>
                  )}
                  {' '}between {new Date(comparison.baseline.createdAt).toLocaleDateString()} and {new Date(comparison.current.createdAt).toLocaleDateString()}.
                </p>
              )}
            </div>

            <Button 
              variant="outline" 
              onClick={() => setSelectedIds([])}
              className="w-full"
            >
              Select Different Benchmarks
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
