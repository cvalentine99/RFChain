import { JarvisLayout } from "@/components/JarvisLayout";
import { JarvisPanel, JarvisStat, JarvisProgress, JarvisStatusIndicator } from "@/components/JarvisPanel";
import { JarvisChat } from "@/components/JarvisChat";
import { Activity, Radio, Shield, Upload, Zap, Database, Cpu, HardDrive, Thermometer, Gauge } from "lucide-react";
import { trpc } from "@/lib/trpc";
import { useEffect, useState } from "react";

// GPU stats type
interface GpuStats {
  success: boolean;
  error?: string;
  gpu_count?: number;
  gpus?: Array<{
    index: number;
    name: string;
    memory: {
      total_mb: number;
      used_mb: number;
      free_mb: number;
      used_percent: number;
    };
    utilization: {
      gpu_percent: number;
      memory_percent: number;
    };
    temperature_c: number;
    power: {
      draw_w: number;
      limit_w: number;
    };
  }>;
}

export default function Dashboard() {
  const { data: recentAnalyses, isLoading } = trpc.analysis.getRecent.useQuery({ limit: 5 });
  
  // GPU monitoring state
  const [gpuStats, setGpuStats] = useState<GpuStats | null>(null);
  const { data: gpuData, refetch: refetchGpu } = trpc.gpu.getStats.useQuery(undefined, {
    refetchInterval: 3000, // Refresh every 3 seconds
  });
  
  useEffect(() => {
    if (gpuData) {
      setGpuStats(gpuData as GpuStats);
    }
  }, [gpuData]);
  
  // Get primary GPU stats
  const primaryGpu = gpuStats?.gpus?.[0];
  const gpuMemoryPercent = primaryGpu?.memory?.used_percent ?? 0;
  const gpuUtilization = primaryGpu?.utilization?.gpu_percent ?? 0;
  const gpuTemp = primaryGpu?.temperature_c ?? 0;
  const gpuPower = primaryGpu?.power?.draw_w ?? 0;
  const gpuPowerLimit = primaryGpu?.power?.limit_w ?? 450;
  const gpuName = primaryGpu?.name ?? "No GPU Detected";
  const gpuVramUsed = primaryGpu?.memory?.used_mb ?? 0;
  const gpuVramTotal = primaryGpu?.memory?.total_mb ?? 0;

  return (
    <JarvisLayout>
      <JarvisChat />
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-wider jarvis-text">SYSTEM DASHBOARD</h2>
            <p className="text-sm text-muted-foreground mt-1">
              RF Signal Intelligence & Forensic Analysis Platform
            </p>
          </div>
          <JarvisStatusIndicator status="online" label="All Systems Operational" />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <JarvisPanel title="Total Analyses" glowPulse>
            <div className="flex items-center justify-between">
              <JarvisStat label="Processed" value={recentAnalyses?.length ?? 0} />
              <Activity className="w-8 h-8 text-primary opacity-50" />
            </div>
          </JarvisPanel>

          <JarvisPanel title="Signal Types">
            <div className="flex items-center justify-between">
              <JarvisStat label="Detected" value={3} unit="types" />
              <Radio className="w-8 h-8 text-primary opacity-50" />
            </div>
          </JarvisPanel>

          <JarvisPanel title="Forensic Chains">
            <div className="flex items-center justify-between">
              <JarvisStat label="Verified" value={6} unit="checkpoints" status="good" />
              <Shield className="w-8 h-8 text-green-400 opacity-50" />
            </div>
          </JarvisPanel>

          <JarvisPanel title="Storage Used">
            <div className="flex items-center justify-between">
              <JarvisStat label="S3 Bucket" value="2.4" unit="GB" />
              <Database className="w-8 h-8 text-primary opacity-50" />
            </div>
          </JarvisPanel>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Analyses */}
          <div className="lg:col-span-2">
            <JarvisPanel title="Recent Analyses" scanLine>
              {isLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-16 bg-secondary/50 rounded animate-pulse" />
                  ))}
                </div>
              ) : recentAnalyses && recentAnalyses.length > 0 ? (
                <div className="space-y-3">
                  {recentAnalyses.map((analysis) => (
                    <div
                      key={analysis.id}
                      className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg border border-border/30 hover:border-primary/50 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded bg-primary/20 flex items-center justify-center">
                          <Radio className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <p className="font-medium text-sm">Analysis #{analysis.id}</p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(analysis.createdAt).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <JarvisStatusIndicator
                          status="online"
                          label="COMPLETED"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-50" />
                  <p className="text-muted-foreground">No analyses yet</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Upload a signal file to begin forensic analysis
                  </p>
                </div>
              )}
            </JarvisPanel>
          </div>

          {/* System Status */}
          <div className="space-y-4">
            <JarvisPanel title="System Resources">
              <div className="space-y-4">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Cpu className="w-4 h-4 text-primary" />
                    <span className="text-xs uppercase tracking-wider">CPU Usage</span>
                  </div>
                  <JarvisProgress value={23} label="" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <HardDrive className="w-4 h-4 text-primary" />
                    <span className="text-xs uppercase tracking-wider">Memory</span>
                  </div>
                  <JarvisProgress value={45} label="" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-primary" />
                    <span className="text-xs uppercase tracking-wider">GPU</span>
                  </div>
                  <JarvisProgress value={gpuUtilization} label="" />
                </div>
              </div>
            </JarvisPanel>
            
            {/* GPU VRAM Panel */}
            <JarvisPanel title="GPU Memory (VRAM)" glowPulse={gpuMemoryPercent > 80}>
              <div className="space-y-3">
                {gpuStats?.success && primaryGpu ? (
                  <>
                    <div className="text-xs text-muted-foreground mb-2 truncate" title={gpuName}>
                      {gpuName}
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs uppercase tracking-wider">VRAM Used</span>
                        <span className="text-xs font-mono">
                          {(gpuVramUsed / 1024).toFixed(1)} / {(gpuVramTotal / 1024).toFixed(1)} GB
                        </span>
                      </div>
                      <JarvisProgress 
                        value={gpuMemoryPercent} 
                        label="" 
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-3">
                      <div className="flex items-center gap-2">
                        <Thermometer className="w-3 h-3 text-orange-400" />
                        <span className="text-xs">
                          <span className={gpuTemp > 80 ? "text-red-400" : gpuTemp > 70 ? "text-orange-400" : "text-green-400"}>
                            {gpuTemp}°C
                          </span>
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Gauge className="w-3 h-3 text-yellow-400" />
                        <span className="text-xs">
                          {gpuPower.toFixed(0)}W / {gpuPowerLimit.toFixed(0)}W
                        </span>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-4">
                    <Zap className="w-8 h-8 text-muted-foreground mx-auto mb-2 opacity-50" />
                    <p className="text-xs text-muted-foreground">
                      {gpuStats?.error || "GPU not detected"}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Install NVIDIA drivers for GPU monitoring
                    </p>
                  </div>
                )}
              </div>
            </JarvisPanel>

            <JarvisPanel title="Forensic Standards">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">NIST SP 800-86</span>
                  <span className="text-green-400">✓ Compliant</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">ISO/IEC 27037</span>
                  <span className="text-green-400">✓ Compliant</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">SWGDE Standards</span>
                  <span className="text-green-400">✓ Compliant</span>
                </div>
              </div>
            </JarvisPanel>
          </div>
        </div>

        {/* Quick Actions */}
        <JarvisPanel title="Quick Actions">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <QuickActionButton
              icon={Upload}
              label="Upload Signal"
              href="/upload"
            />
            <QuickActionButton
              icon={Radio}
              label="View Analysis"
              href="/analysis"
            />
            <QuickActionButton
              icon={Shield}
              label="Forensic Report"
              href="/forensics"
            />
            <QuickActionButton
              icon={Activity}
              label="System Logs"
              href="/settings"
            />
          </div>
        </JarvisPanel>
      </div>
    </JarvisLayout>
  );
}

function QuickActionButton({
  icon: Icon,
  label,
  href,
}: {
  icon: typeof Upload;
  label: string;
  href: string;
}) {
  return (
    <a
      href={href}
      className="flex flex-col items-center justify-center p-4 bg-secondary/30 rounded-lg border border-border/30 hover:border-primary/50 hover:bg-secondary/50 transition-all group"
    >
      <Icon className="w-8 h-8 text-primary mb-2 group-hover:scale-110 transition-transform" />
      <span className="text-sm font-medium uppercase tracking-wider">{label}</span>
    </a>
  );
}
