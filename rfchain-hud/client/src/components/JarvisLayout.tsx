import { cn } from "@/lib/utils";
import { ReactNode, useState, useEffect } from "react";
import { Activity, Radio, Upload, MessageSquare, Shield, Settings, Menu, X, Zap, GitCompare } from "lucide-react";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { JarvisStatusIndicator } from "./JarvisPanel";

interface JarvisLayoutProps {
  children: ReactNode;
}

const navItems = [
  { href: "/", icon: Activity, label: "Dashboard" },
  { href: "/upload", icon: Upload, label: "Upload Signal" },
  { href: "/analysis", icon: Radio, label: "Analysis" },
  { href: "/compare", icon: GitCompare, label: "Compare" },
  { href: "/forensics", icon: Shield, label: "Forensics" },
  { href: "/settings", icon: Settings, label: "Settings" },
];

export function JarvisLayout({ children }: JarvisLayoutProps) {
  const [location] = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background jarvis-hex-grid">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="flex items-center justify-between h-full px-4">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
            
            <div className="flex items-center gap-2">
              <div className="relative">
                <Zap className="w-8 h-8 text-primary" />
                <div className="absolute inset-0 w-8 h-8 text-primary blur-sm opacity-50">
                  <Zap className="w-8 h-8" />
                </div>
              </div>
              <div className="flex flex-col">
                <h1 className="text-lg font-bold tracking-wider jarvis-text">RFCHAIN</h1>
                <span className="text-[10px] text-muted-foreground uppercase tracking-widest">
                  Signal Intelligence System
                </span>
              </div>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="hidden md:flex items-center gap-6">
            <JarvisStatusIndicator status="online" label="System Active" />
            <div className="h-4 w-px bg-border" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">JARVIS AI</span>
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            </div>
          </div>

          {/* Time Display */}
          <div className="hidden sm:block">
            <JarvisTime />
          </div>
        </div>
        
        {/* Animated bottom border */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent opacity-50" />
      </header>

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed top-16 left-0 bottom-0 w-64 border-r border-border/50 bg-sidebar/80 backdrop-blur-md z-40 transition-transform duration-300",
          "lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <nav className="p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = location === item.href;
            return (
              <Link 
                key={item.href} 
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200",
                  "hover:bg-sidebar-accent",
                  isActive
                    ? "bg-sidebar-accent border border-primary/50 jarvis-text"
                    : "text-sidebar-foreground"
                )}
                onClick={() => setSidebarOpen(false)}
              >
                <item.icon className={cn("w-5 h-5", isActive && "text-primary")} />
                <span className="text-sm font-medium uppercase tracking-wider">{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="jarvis-panel rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <MessageSquare className="w-4 h-4 text-primary" />
              <span className="text-xs uppercase tracking-wider">AI Assistant</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Press <kbd className="px-1 py-0.5 bg-secondary rounded text-primary">Ctrl+J</kbd> to open chat
            </p>
          </div>
        </div>
      </aside>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-4 lg:p-6">{children}</div>
      </main>
    </div>
  );
}

function JarvisTime() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
      year: "numeric",
    });
  };

  return (
    <div className="text-right">
      <div className="text-lg font-bold tabular-nums jarvis-text">{formatTime(time)}</div>
      <div className="text-xs text-muted-foreground uppercase tracking-wider">{formatDate(time)}</div>
    </div>
  );
}
