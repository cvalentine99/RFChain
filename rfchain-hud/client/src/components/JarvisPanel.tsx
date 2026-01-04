import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface JarvisPanelProps {
  children: ReactNode;
  title?: string;
  className?: string;
  glowPulse?: boolean;
  scanLine?: boolean;
  corners?: boolean;
}

export function JarvisPanel({
  children,
  title,
  className,
  glowPulse = false,
  scanLine = false,
  corners = true,
}: JarvisPanelProps) {
  return (
    <div
      className={cn(
        "jarvis-panel rounded-lg p-4 relative",
        glowPulse && "jarvis-glow-pulse",
        scanLine && "jarvis-scan-line",
        className
      )}
    >
      {corners && (
        <>
          {/* Top-left corner */}
          <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-primary" />
          {/* Top-right corner */}
          <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-primary" />
          {/* Bottom-left corner */}
          <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-primary" />
          {/* Bottom-right corner */}
          <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-primary" />
        </>
      )}
      
      {title && (
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-border/50">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <h3 className="text-sm font-semibold uppercase tracking-wider jarvis-text">
            {title}
          </h3>
        </div>
      )}
      
      {children}
    </div>
  );
}

interface JarvisStatProps {
  label: string;
  value: string | number;
  unit?: string;
  status?: "normal" | "warning" | "critical" | "good";
  className?: string;
}

export function JarvisStat({ label, value, unit, status = "normal", className }: JarvisStatProps) {
  const statusColors = {
    normal: "text-foreground",
    good: "text-green-400",
    warning: "text-yellow-400",
    critical: "text-red-400",
  };

  return (
    <div className={cn("flex flex-col", className)}>
      <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className={cn("text-xl font-bold tabular-nums", statusColors[status])}>
          {value}
        </span>
        {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
      </div>
    </div>
  );
}

interface JarvisProgressProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  className?: string;
}

export function JarvisProgress({ value, max = 100, label, showValue = true, className }: JarvisProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div className={cn("space-y-1", className)}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs">
          {label && <span className="text-muted-foreground uppercase tracking-wider">{label}</span>}
          {showValue && <span className="jarvis-text">{percentage.toFixed(1)}%</span>}
        </div>
      )}
      <div className="h-2 bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-primary/80 to-primary transition-all duration-500 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

interface JarvisStatusIndicatorProps {
  status: "online" | "offline" | "processing" | "idle";
  label?: string;
  className?: string;
}

export function JarvisStatusIndicator({ status, label, className }: JarvisStatusIndicatorProps) {
  const statusConfig = {
    online: { color: "bg-green-500", pulse: true, text: "Online" },
    offline: { color: "bg-red-500", pulse: false, text: "Offline" },
    processing: { color: "bg-yellow-500", pulse: true, text: "Processing" },
    idle: { color: "bg-primary", pulse: false, text: "Idle" },
  };

  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className={cn("w-2 h-2 rounded-full", config.color, config.pulse && "animate-pulse")} />
      <span className="text-xs text-muted-foreground uppercase tracking-wider">
        {label || config.text}
      </span>
    </div>
  );
}
