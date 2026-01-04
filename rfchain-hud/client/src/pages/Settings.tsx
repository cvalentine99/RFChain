import { JarvisLayout } from "@/components/JarvisLayout";
import { JarvisPanel, JarvisStatusIndicator, JarvisProgress } from "@/components/JarvisPanel";
import { Settings as SettingsIcon, Key, Cpu, Volume2, Mic, Save, RefreshCw, Brain, Database, Loader2, Zap, Play, Thermometer, Gauge, History, TrendingUp, Trash2, ChevronDown, ChevronUp } from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { BenchmarkChart } from "@/components/BenchmarkChart";
import { BenchmarkExport } from "@/components/BenchmarkExport";
import { BenchmarkCompare } from "@/components/BenchmarkCompare";

type LLMProvider = "builtin" | "openai" | "anthropic" | "local";

export default function Settings() {
  const { data: llmConfig, isLoading } = trpc.llmConfig.get.useQuery();
  const updateConfig = trpc.llmConfig.update.useMutation({
    onSuccess: () => {
      toast.success("Settings saved successfully");
    },
    onError: (error) => {
      toast.error(`Failed to save settings: ${error.message}`);
    },
  });

  const [provider, setProvider] = useState<LLMProvider>("builtin");
  const [apiKey, setApiKey] = useState("");
  const [localEndpoint, setLocalEndpoint] = useState("");
  const [model, setModel] = useState("");

  useEffect(() => {
    if (llmConfig) {
      setProvider(llmConfig.provider);
      setApiKey(llmConfig.apiKey || "");
      setLocalEndpoint(llmConfig.localEndpoint || "");
      setModel(llmConfig.model || "");
    }
  }, [llmConfig]);

  const handleSave = () => {
    updateConfig.mutate({
      provider,
      apiKey: apiKey || undefined,
      localEndpoint: localEndpoint || undefined,
      model: model || undefined,
    });
  };

  const providers: { id: LLMProvider; name: string; description: string }[] = [
    { id: "builtin", name: "Built-in AI", description: "Use the platform's built-in LLM service" },
    { id: "openai", name: "OpenAI", description: "Use OpenAI GPT models (requires API key)" },
    { id: "anthropic", name: "Anthropic", description: "Use Claude models (requires API key)" },
    { id: "local", name: "Local LLM", description: "Connect to a local LLM server" },
  ];

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h2 className="text-2xl font-bold tracking-wider jarvis-text">SYSTEM SETTINGS</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Configure AI assistant, voice settings, and system preferences
          </p>
        </div>

        {/* LLM Configuration */}
        <JarvisPanel title="AI Model Configuration" glowPulse>
          <div className="space-y-6">
            {/* Provider Selection */}
            <div>
              <Label className="text-sm uppercase tracking-wider mb-3 block">
                LLM Provider
              </Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {providers.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setProvider(p.id)}
                    className={cn(
                      "p-4 rounded-lg border text-left transition-all",
                      provider === p.id
                        ? "border-primary bg-primary/10"
                        : "border-border/50 hover:border-primary/50"
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <Cpu className={cn("w-4 h-4", provider === p.id ? "text-primary" : "text-muted-foreground")} />
                      <span className="font-semibold">{p.name}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">{p.description}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* API Key Input (for OpenAI/Anthropic) */}
            {(provider === "openai" || provider === "anthropic") && (
              <div>
                <Label htmlFor="apiKey" className="text-sm uppercase tracking-wider mb-2 block">
                  API Key
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      id="apiKey"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={`Enter your ${provider === "openai" ? "OpenAI" : "Anthropic"} API key`}
                      className="pl-10"
                    />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Your API key is encrypted and stored securely
                </p>
              </div>
            )}

            {/* Local Endpoint (for Local LLM) */}
            {provider === "local" && (
              <div>
                <Label htmlFor="endpoint" className="text-sm uppercase tracking-wider mb-2 block">
                  Local Endpoint
                </Label>
                <Input
                  id="endpoint"
                  value={localEndpoint}
                  onChange={(e) => setLocalEndpoint(e.target.value)}
                  placeholder="http://localhost:11434/v1"
                />
                <p className="text-xs text-muted-foreground mt-2">
                  OpenAI-compatible API endpoint (e.g., Ollama, LM Studio)
                </p>
              </div>
            )}

            {/* Model Selection */}
            {provider !== "builtin" && (
              <div>
                <Label htmlFor="model" className="text-sm uppercase tracking-wider mb-2 block">
                  Model Name
                </Label>
                <Input
                  id="model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder={
                    provider === "openai"
                      ? "gpt-4-turbo-preview"
                      : provider === "anthropic"
                      ? "claude-3-opus-20240229"
                      : "llama2"
                  }
                />
              </div>
            )}

            {/* Save Button */}
            <div className="flex justify-end">
              <Button onClick={handleSave} disabled={updateConfig.isPending} className="gap-2">
                {updateConfig.isPending ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                Save Configuration
              </Button>
            </div>
          </div>
        </JarvisPanel>

        {/* Voice Settings */}
        <JarvisPanel title="Voice Settings">
          <div className="space-y-6">
            {/* Voice Output */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Volume2 className="w-4 h-4 text-primary" />
                <Label className="text-sm uppercase tracking-wider">Voice Output</Label>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <VoiceOption name="JARVIS Classic" description="British male voice" selected />
                <VoiceOption name="FRIDAY" description="American female voice" />
                <VoiceOption name="System Default" description="Browser TTS" />
              </div>
            </div>

            {/* Voice Input */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Mic className="w-4 h-4 text-primary" />
                <Label className="text-sm uppercase tracking-wider">Voice Input</Label>
              </div>
              <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
                <div>
                  <p className="font-medium">Whisper Transcription</p>
                  <p className="text-xs text-muted-foreground">
                    Use OpenAI Whisper for voice-to-text
                  </p>
                </div>
                <JarvisStatusIndicator status="online" label="Active" />
              </div>
            </div>
          </div>
        </JarvisPanel>

        {/* RAG Settings */}
        <RagSettingsPanel />
        
        {/* GPU Benchmark */}
        <GpuBenchmarkPanel />

        {/* System Information */}
        <JarvisPanel title="System Information">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-secondary/30 rounded-lg">
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                Analysis Engine
              </p>
              <p className="font-medium">RF Forensic Analyzer v2.2.2</p>
            </div>
            <div className="p-4 bg-secondary/30 rounded-lg">
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                Hash Algorithms
              </p>
              <p className="font-medium">SHA-256 + SHA3-256</p>
            </div>
            <div className="p-4 bg-secondary/30 rounded-lg">
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                Forensic Checkpoints
              </p>
              <p className="font-medium">6-Stage Pipeline</p>
            </div>
            <div className="p-4 bg-secondary/30 rounded-lg">
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                Compliance
              </p>
              <p className="font-medium">NIST SP 800-86, ISO/IEC 27037</p>
            </div>
          </div>
        </JarvisPanel>
      </div>
    </JarvisLayout>
  );
}

function RagSettingsPanel() {
  const { data: ragData, isLoading } = trpc.ragSettings.get.useQuery();
  const updateSettings = trpc.ragSettings.update.useMutation({
    onSuccess: () => {
      toast.success("RAG settings saved");
    },
    onError: (error) => {
      toast.error(`Failed to save: ${error.message}`);
    },
  });
  const backfill = trpc.embedding.backfill.useMutation({
    onSuccess: (result) => {
      toast.success(`Backfill complete: ${result.success} of ${result.total} analyses embedded`);
    },
    onError: (error) => {
      toast.error(`Backfill failed: ${error.message}`);
    },
  });

  const [enabled, setEnabled] = useState(true);
  const [threshold, setThreshold] = useState(0.7);
  const [maxResults, setMaxResults] = useState(5);
  const [autoEmbed, setAutoEmbed] = useState(true);

  useEffect(() => {
    if (ragData?.settings) {
      setEnabled(ragData.settings.enabled === 1);
      setThreshold(ragData.settings.similarityThreshold);
      setMaxResults(ragData.settings.maxResults);
      setAutoEmbed(ragData.settings.autoEmbed === 1);
    }
  }, [ragData]);

  const handleSave = () => {
    updateSettings.mutate({
      enabled: enabled ? 1 : 0,
      similarityThreshold: threshold,
      maxResults,
      autoEmbed: autoEmbed ? 1 : 0,
    });
  };

  const handleBackfill = () => {
    backfill.mutate({ limit: 100 });
  };

  if (isLoading) {
    return (
      <JarvisPanel title="RAG Configuration">
        <div className="h-40 flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
        </div>
      </JarvisPanel>
    );
  }

  const stats = ragData?.stats || { total: 0, embedded: 0 };
  const embeddingProgress = stats.total > 0 ? Math.round((stats.embedded / stats.total) * 100) : 0;

  return (
    <JarvisPanel title="RAG Configuration" glowPulse>
      <div className="space-y-6">
        {/* RAG Status */}
        <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
          <div className="flex items-center gap-3">
            <Brain className="w-5 h-5 text-primary" />
            <div>
              <p className="font-medium">Retrieval-Augmented Generation</p>
              <p className="text-xs text-muted-foreground">
                Search past analyses for context-aware responses
              </p>
            </div>
          </div>
          <button
            onClick={() => setEnabled(!enabled)}
            className={cn(
              "w-12 h-6 rounded-full transition-colors relative",
              enabled ? "bg-primary" : "bg-secondary"
            )}
          >
            <span
              className={cn(
                "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                enabled ? "translate-x-7" : "translate-x-1"
              )}
            />
          </button>
        </div>

        {/* Embedding Stats */}
        <div className="p-4 bg-secondary/30 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-4 h-4 text-primary" />
            <Label className="text-sm uppercase tracking-wider">Embedding Status</Label>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Analyses Embedded</span>
              <span>{stats.embedded} / {stats.total}</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${embeddingProgress}%` }}
              />
            </div>
            {stats.total > stats.embedded && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleBackfill}
                disabled={backfill.isPending}
                className="mt-2 gap-2"
              >
                {backfill.isPending ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Embedding...
                  </>
                ) : (
                  <>
                    <Database className="w-3 h-3" />
                    Backfill {stats.total - stats.embedded} Analyses
                  </>
                )}
              </Button>
            )}
          </div>
        </div>

        {/* Similarity Threshold */}
        <div>
          <Label className="text-sm uppercase tracking-wider mb-2 block">
            Similarity Threshold: {(threshold * 100).toFixed(0)}%
          </Label>
          <input
            type="range"
            min="0"
            max="100"
            value={threshold * 100}
            onChange={(e) => setThreshold(Number(e.target.value) / 100)}
            className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
          />
          <p className="text-xs text-muted-foreground mt-1">
            Higher values return more relevant but fewer results
          </p>
        </div>

        {/* Max Results */}
        <div>
          <Label className="text-sm uppercase tracking-wider mb-2 block">
            Max Context Results: {maxResults}
          </Label>
          <input
            type="range"
            min="1"
            max="10"
            value={maxResults}
            onChange={(e) => setMaxResults(Number(e.target.value))}
            className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
          />
          <p className="text-xs text-muted-foreground mt-1">
            Number of past analyses to include in chat context
          </p>
        </div>

        {/* Auto-Embed */}
        <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
          <div>
            <p className="font-medium">Auto-Embed New Analyses</p>
            <p className="text-xs text-muted-foreground">
              Automatically generate embeddings for new analyses
            </p>
          </div>
          <button
            onClick={() => setAutoEmbed(!autoEmbed)}
            className={cn(
              "w-12 h-6 rounded-full transition-colors relative",
              autoEmbed ? "bg-primary" : "bg-secondary"
            )}
          >
            <span
              className={cn(
                "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                autoEmbed ? "translate-x-7" : "translate-x-1"
              )}
            />
          </button>
        </div>

        {/* Save Button */}
        <div className="flex justify-end">
          <Button onClick={handleSave} disabled={updateSettings.isPending} className="gap-2">
            {updateSettings.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Save RAG Settings
          </Button>
        </div>
      </div>
    </JarvisPanel>
  );
}

function GpuBenchmarkPanel() {
  const { data: gpuStats, isLoading: gpuLoading } = trpc.gpu.getStats.useQuery();
  const { data: benchmarkHistory, refetch: refetchHistory } = trpc.gpu.getHistory.useQuery({ limit: 10 });
  const { data: benchmarkStats } = trpc.gpu.getBenchmarkStats.useQuery();
  const runBenchmark = trpc.gpu.runBenchmark.useMutation({
    onSuccess: (result: any) => {
      if (result.success) {
        toast.success(`Benchmark complete! Average speedup: ${result.summary?.avg_speedup}x`);
        setBenchmarkResults(result);
        refetchHistory();
      } else {
        toast.error(`Benchmark failed: ${result.error}`);
      }
    },
    onError: (error) => {
      toast.error(`Benchmark error: ${error.message}`);
    },
  });
  const deleteBenchmark = trpc.gpu.deleteBenchmark.useMutation({
    onSuccess: () => {
      toast.success('Benchmark record deleted');
      refetchHistory();
    },
  });
  
  const [benchmarkResults, setBenchmarkResults] = useState<any>(null);
  const [showHistory, setShowHistory] = useState(false);
  
  const gpuData = gpuStats as any;
  const primaryGpu = gpuData?.gpus?.[0];
  const gpuAvailable = gpuData?.success && primaryGpu;
  
  return (
    <JarvisPanel title="GPU Performance" glowPulse>
      <div className="space-y-6">
        {/* GPU Status */}
        <div className="p-4 bg-secondary/30 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <Zap className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium">
                  {gpuAvailable ? primaryGpu.name : "GPU Not Detected"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {gpuAvailable 
                    ? `${(primaryGpu.memory.total_mb / 1024).toFixed(0)} GB VRAM`
                    : "Install NVIDIA drivers for GPU acceleration"
                  }
                </p>
              </div>
            </div>
            <JarvisStatusIndicator 
              status={gpuAvailable ? "online" : "offline"} 
              label={gpuAvailable ? "Ready" : "Unavailable"} 
            />
          </div>
          
          {gpuAvailable && (
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="text-center">
                <Thermometer className="w-4 h-4 mx-auto mb-1 text-orange-400" />
                <p className={cn(
                  "text-lg font-mono",
                  primaryGpu.temperature_c > 80 ? "text-red-400" : 
                  primaryGpu.temperature_c > 70 ? "text-orange-400" : "text-green-400"
                )}>
                  {primaryGpu.temperature_c}°C
                </p>
                <p className="text-xs text-muted-foreground">Temperature</p>
              </div>
              <div className="text-center">
                <Gauge className="w-4 h-4 mx-auto mb-1 text-yellow-400" />
                <p className="text-lg font-mono">
                  {primaryGpu.power.draw_w.toFixed(0)}W
                </p>
                <p className="text-xs text-muted-foreground">Power Draw</p>
              </div>
              <div className="text-center">
                <Cpu className="w-4 h-4 mx-auto mb-1 text-primary" />
                <p className="text-lg font-mono">
                  {primaryGpu.utilization.gpu_percent}%
                </p>
                <p className="text-xs text-muted-foreground">Utilization</p>
              </div>
            </div>
          )}
        </div>
        
        {/* Run Benchmark */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="font-medium">Performance Benchmark</p>
              <p className="text-xs text-muted-foreground">
                Test FFT, correlation, and PSD operations on CPU vs GPU
              </p>
            </div>
            <Button
              onClick={() => runBenchmark.mutate()}
              disabled={runBenchmark.isPending || !gpuAvailable}
              className="gap-2"
            >
              {runBenchmark.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Benchmark
                </>
              )}
            </Button>
          </div>
        </div>
        
        {/* Benchmark Results */}
        {benchmarkResults?.success && (
          <div className="space-y-4">
            {/* Summary */}
            <div className="p-4 bg-primary/10 rounded-lg border border-primary/30">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-2xl font-bold text-primary">
                    {benchmarkResults.summary.avg_speedup}x
                  </p>
                  <p className="text-xs text-muted-foreground">Avg Speedup</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-green-400">
                    {benchmarkResults.summary.max_speedup}x
                  </p>
                  <p className="text-xs text-muted-foreground">Max Speedup</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-orange-400">
                    {benchmarkResults.summary.min_speedup}x
                  </p>
                  <p className="text-xs text-muted-foreground">Min Speedup</p>
                </div>
              </div>
            </div>
            
            {/* Detailed Results */}
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">Detailed Results</p>
              <div className="max-h-60 overflow-y-auto space-y-2">
                {benchmarkResults.benchmarks.map((bench: any, idx: number) => (
                  <div key={idx} className="p-3 bg-secondary/30 rounded-lg text-sm">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium">{bench.operation}</span>
                      <span className="text-xs text-muted-foreground">{bench.size_name}</span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-muted-foreground">CPU: </span>
                        <span>{bench.cpu.avg_ms.toFixed(2)}ms</span>
                      </div>
                      {bench.gpu && (
                        <>
                          <div>
                            <span className="text-muted-foreground">GPU: </span>
                            <span className="text-primary">{bench.gpu.avg_ms.toFixed(2)}ms</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Speedup: </span>
                            <span className="text-green-400">{bench.speedup}x</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* No GPU Message */}
        {!gpuAvailable && (
          <div className="text-center py-4">
            <Zap className="w-8 h-8 text-muted-foreground mx-auto mb-2 opacity-50" />
            <p className="text-sm text-muted-foreground">
              GPU acceleration unavailable
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Install NVIDIA drivers and CuPy for GPU-accelerated analysis
            </p>
          </div>
        )}
        
        {/* Benchmark History Section */}
        {benchmarkStats && benchmarkStats.totalRuns > 0 && (
          <div className="border-t border-border/30 pt-4">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="flex items-center justify-between w-full text-left"
            >
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-primary" />
                <span className="font-medium">Benchmark History</span>
                <span className="text-xs text-muted-foreground">({benchmarkStats.totalRuns} runs)</span>
              </div>
              {showHistory ? (
                <ChevronUp className="w-4 h-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="w-4 h-4 text-muted-foreground" />
              )}
            </button>
            
            {showHistory && (
              <div className="mt-4 space-y-4">
                {/* Stats Summary */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 bg-secondary/30 rounded-lg text-center">
                    <TrendingUp className="w-4 h-4 mx-auto mb-1 text-primary" />
                    <p className="text-lg font-mono text-primary">
                      {benchmarkStats.avgSpeedup?.toFixed(1) || '-'}x
                    </p>
                    <p className="text-xs text-muted-foreground">Avg Speedup</p>
                  </div>
                  <div className="p-3 bg-secondary/30 rounded-lg text-center">
                    <Zap className="w-4 h-4 mx-auto mb-1 text-green-400" />
                    <p className="text-lg font-mono text-green-400">
                      {benchmarkStats.bestSpeedup?.toFixed(1) || '-'}x
                    </p>
                    <p className="text-xs text-muted-foreground">Best Ever</p>
                  </div>
                  <div className="p-3 bg-secondary/30 rounded-lg text-center">
                    <History className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
                    <p className="text-sm font-mono">
                      {benchmarkStats.lastRunDate 
                        ? new Date(benchmarkStats.lastRunDate).toLocaleDateString()
                        : '-'
                      }
                    </p>
                    <p className="text-xs text-muted-foreground">Last Run</p>
                  </div>
                </div>
                
                {/* Performance Trend Chart */}
                {benchmarkHistory && benchmarkHistory.length > 0 && (
                  <div className="p-4 bg-secondary/20 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-xs uppercase tracking-wider text-muted-foreground">Performance Trend</p>
                      <div className="flex gap-2">
                        <BenchmarkCompare history={benchmarkHistory as any} />
                        <BenchmarkExport 
                          history={benchmarkHistory as any} 
                          gpuName={primaryGpu?.name}
                        />
                      </div>
                    </div>
                    <BenchmarkChart history={benchmarkHistory as any} />
                  </div>
                )}
                
                {/* History List */}
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {benchmarkHistory?.map((record: any) => (
                    <div
                      key={record.id}
                      className="p-3 bg-secondary/30 rounded-lg flex items-center justify-between"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">
                            {record.gpuName || 'Unknown GPU'}
                          </span>
                          {record.success === 1 ? (
                            <span className="text-xs px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded">
                              Success
                            </span>
                          ) : (
                            <span className="text-xs px-1.5 py-0.5 bg-red-500/20 text-red-400 rounded">
                              Failed
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                          <span>
                            {new Date(record.createdAt).toLocaleString()}
                          </span>
                          {record.avgSpeedup && (
                            <span className="text-primary">
                              Avg: {record.avgSpeedup.toFixed(1)}x
                            </span>
                          )}
                          {record.maxSpeedup && (
                            <span className="text-green-400">
                              Max: {record.maxSpeedup.toFixed(1)}x
                            </span>
                          )}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteBenchmark.mutate({ id: record.id })}
                        disabled={deleteBenchmark.isPending}
                        className="text-muted-foreground hover:text-red-400"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </JarvisPanel>
  );
}

function VoiceOption({
  name,
  description,
  selected = false,
}: {
  name: string;
  description: string;
  selected?: boolean;
}) {
  return (
    <button
      className={cn(
        "p-4 rounded-lg border text-left transition-all",
        selected
          ? "border-primary bg-primary/10"
          : "border-border/50 hover:border-primary/50"
      )}
    >
      <p className="font-semibold">{name}</p>
      <p className="text-xs text-muted-foreground">{description}</p>
    </button>
  );
}
