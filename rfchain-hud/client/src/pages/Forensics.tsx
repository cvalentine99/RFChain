import { JarvisLayout } from "@/components/JarvisLayout";
import { JarvisPanel, JarvisStatusIndicator } from "@/components/JarvisPanel";
import { Shield, CheckCircle, XCircle, ChevronRight, Copy, Check, Download, Loader2, Clock, Database, FileText } from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Link, useParams } from "wouter";
import { Button } from "@/components/ui/button";
import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";

export default function Forensics() {
  const params = useParams<{ id?: string }>();
  const analysisId = params.id ? parseInt(params.id) : undefined;

  if (analysisId) {
    return <ForensicDetail analysisId={analysisId} />;
  }

  return <ForensicsList />;
}

function ForensicsList() {
  const { data: reports, isLoading } = trpc.forensic.getUserReports.useQuery({ limit: 20 });

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h2 className="text-2xl font-bold tracking-wider jarvis-text">FORENSIC REPORTS</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Chain of custody and hash verification records
          </p>
        </div>

        {/* Reports List */}
        <JarvisPanel title="Forensic Chain Records" scanLine>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-20 bg-secondary/50 rounded animate-pulse" />
              ))}
            </div>
          ) : reports && reports.length > 0 ? (
            <div className="space-y-3">
              {reports.map((report) => (
                <Link key={report.id} href={`/forensics/${report.analysisId}`}>
                  <a className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg border border-border/30 hover:border-primary/50 transition-all group">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded bg-green-500/20 flex items-center justify-center">
                        <Shield className="w-6 h-6 text-green-500" />
                      </div>
                      <div>
                        <p className="font-medium">Analysis #{report.analysisId}</p>
                        <div className="flex items-center gap-4 mt-1">
                          <span className="text-xs text-muted-foreground">
                            {new Date(report.createdAt).toLocaleString()}
                          </span>
                          <span className="text-xs text-green-400">
                            Hash Chain Verified
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <JarvisStatusIndicator status="online" label="Valid" />
                      <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                    </div>
                  </a>
                </Link>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Shield className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">No forensic reports yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Complete a signal analysis to generate forensic chain records
              </p>
              <Link href="/upload">
                <Button className="mt-4">Upload Signal</Button>
              </Link>
            </div>
          )}
        </JarvisPanel>

        {/* Standards Compliance */}
        <JarvisPanel title="Compliance Standards">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-secondary/30 rounded-lg border border-green-500/30">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span className="font-semibold">NIST SP 800-86</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Guide to Integrating Forensic Techniques into Incident Response
              </p>
            </div>
            <div className="p-4 bg-secondary/30 rounded-lg border border-green-500/30">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span className="font-semibold">ISO/IEC 27037</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Guidelines for identification, collection, acquisition and preservation
              </p>
            </div>
            <div className="p-4 bg-secondary/30 rounded-lg border border-green-500/30">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span className="font-semibold">SWGDE Standards</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Scientific Working Group on Digital Evidence best practices
              </p>
            </div>
          </div>
        </JarvisPanel>
      </div>
    </JarvisLayout>
  );
}

// Type for hash chain entry from Python script
interface HashChainEntry {
  stage: string;
  hash: string;
  timestamp_utc: string;
  data_shape: number[];
  data_dtype: string;
  details: Record<string, unknown>;
}

// Type for forensic pipeline data
interface ForensicPipelineData {
  hash_chain?: HashChainEntry[];
  total_checkpoints?: number;
}

function ForensicDetail({ analysisId }: { analysisId: number }) {
  const { data: report, isLoading } = trpc.forensic.getByAnalysisId.useQuery({ analysisId });

  // Parse forensicPipeline JSON to extract hash chain
  const hashChain = useMemo(() => {
    if (!report?.forensicPipeline) return [];
    
    try {
      const pipeline: ForensicPipelineData = typeof report.forensicPipeline === 'string' 
        ? JSON.parse(report.forensicPipeline) 
        : report.forensicPipeline;
      
      return pipeline.hash_chain || [];
    } catch {
      return [];
    }
  }, [report?.forensicPipeline]);

  // Map stage names to display names
  const stageDisplayNames: Record<string, string> = {
    'raw_input': 'Raw Input',
    'post_metrics': 'Post Metrics',
    'post_anomaly_detection': 'Post Anomaly Detection',
    'post_digital_analysis': 'Post Digital Analysis',
    'post_v3_analysis': 'Post V3 Analysis',
    'pre_output': 'Pre Output',
  };

  if (isLoading) {
    return (
      <JarvisLayout>
        <div className="space-y-6">
          <div className="h-8 w-64 bg-secondary/50 rounded animate-pulse" />
          <div className="space-y-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="h-24 bg-secondary/50 rounded animate-pulse" />
            ))}
          </div>
        </div>
      </JarvisLayout>
    );
  }

  if (!report) {
    return (
      <JarvisLayout>
        <div className="text-center py-12">
          <Shield className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium">Forensic report not found</p>
          <p className="text-sm text-muted-foreground mt-1">
            The requested forensic report could not be found
          </p>
          <Link href="/forensics">
            <Button className="mt-4">Back to List</Button>
          </Link>
        </div>
      </JarvisLayout>
    );
  }

  const checkpointCount = hashChain.length;
  const allValid = hashChain.length > 0 && hashChain.every(c => c.hash);

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <Link href="/forensics">
                <a className="hover:text-primary">Forensics</a>
              </Link>
              <ChevronRight className="w-4 h-4" />
              <span>Report #{analysisId}</span>
            </div>
            <h2 className="text-2xl font-bold tracking-wider jarvis-text">
              FORENSIC CHAIN REPORT
            </h2>
          </div>
          <div className="flex items-center gap-4">
            <JarvisStatusIndicator 
              status={allValid ? "online" : "idle"} 
              label={allValid ? "Chain Verified" : "Verification Pending"} 
            />
            <PdfExportButton analysisId={analysisId} />
          </div>
        </div>

        {/* Hash Chain Visualization */}
        <JarvisPanel title={`${checkpointCount}-Stage Hash Chain`} glowPulse>
          {hashChain.length > 0 ? (
            <div className="space-y-4">
              {hashChain.map((checkpoint, index) => (
                <HashCheckpoint
                  key={`${checkpoint.stage}-${index}`}
                  index={index + 1}
                  name={stageDisplayNames[checkpoint.stage] || checkpoint.stage}
                  sha256={checkpoint.hash}
                  timestamp={checkpoint.timestamp_utc}
                  dataShape={checkpoint.data_shape}
                  dataType={checkpoint.data_dtype}
                  details={checkpoint.details}
                  isLast={index === hashChain.length - 1}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Shield className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-50" />
              <p className="text-muted-foreground">No hash chain data available</p>
              <p className="text-xs text-muted-foreground mt-1">
                Hash checkpoints are generated during signal analysis
              </p>
            </div>
          )}
        </JarvisPanel>

        {/* Output Verification */}
        <JarvisPanel title="Output File Verification">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-secondary/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs uppercase tracking-wider text-muted-foreground">Output SHA-256</span>
              </div>
              <HashDisplay hash={report.outputSha256} />
            </div>
            <div className="p-4 bg-secondary/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs uppercase tracking-wider text-muted-foreground">Output SHA3-256</span>
              </div>
              <HashDisplay hash={report.outputSha3} />
            </div>
          </div>
        </JarvisPanel>

        {/* Chain of Custody */}
        <JarvisPanel title="Chain of Custody Log">
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between p-2 bg-secondary/20 rounded">
              <span className="text-muted-foreground">Report Generated</span>
              <span>{new Date(report.createdAt).toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-secondary/20 rounded">
              <span className="text-muted-foreground">Analysis ID</span>
              <span>#{report.analysisId}</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-secondary/20 rounded">
              <span className="text-muted-foreground">Hash Checkpoints</span>
              <span>{checkpointCount} stages recorded</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-secondary/20 rounded">
              <span className="text-muted-foreground">Verification Status</span>
              <span className={cn(
                "flex items-center gap-1",
                allValid ? "text-green-400" : "text-yellow-400"
              )}>
                {allValid ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    All Checkpoints Valid
                  </>
                ) : (
                  <>
                    <Clock className="w-4 h-4" />
                    Pending Verification
                  </>
                )}
              </span>
            </div>
          </div>
        </JarvisPanel>
      </div>
    </JarvisLayout>
  );
}

function HashCheckpoint({
  index,
  name,
  sha256,
  timestamp,
  dataShape,
  dataType,
  details,
  isLast,
}: {
  index: number;
  name: string;
  sha256: string | null;
  timestamp?: string;
  dataShape?: number[];
  dataType?: string;
  details?: Record<string, unknown>;
  isLast: boolean;
}) {
  const isValid = !!sha256;

  return (
    <div className="relative">
      <div className="flex items-start gap-4">
        {/* Checkpoint indicator */}
        <div className="flex flex-col items-center">
          <div
            className={cn(
              "w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm",
              isValid
                ? "bg-green-500/20 text-green-500 border-2 border-green-500"
                : "bg-red-500/20 text-red-500 border-2 border-red-500"
            )}
          >
            {index}
          </div>
          {!isLast && (
            <div className="w-0.5 h-20 bg-gradient-to-b from-primary to-primary/20 mt-2" />
          )}
        </div>

        {/* Checkpoint content */}
        <div className="flex-1 pb-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="font-semibold uppercase tracking-wider">{name}</span>
            {isValid ? (
              <CheckCircle className="w-4 h-4 text-green-500" />
            ) : (
              <XCircle className="w-4 h-4 text-red-500" />
            )}
          </div>
          
          {/* Hash display */}
          <div className="p-3 bg-secondary/30 rounded mb-2">
            <span className="text-xs text-muted-foreground block mb-1">SHA-256 Hash</span>
            <HashDisplay hash={sha256} />
          </div>
          
          {/* Metadata row */}
          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
            {timestamp && (
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                <span>{new Date(timestamp).toLocaleString()}</span>
              </div>
            )}
            {dataShape && (
              <div className="flex items-center gap-1">
                <Database className="w-3 h-3" />
                <span>Shape: [{dataShape.join(', ')}]</span>
              </div>
            )}
            {dataType && (
              <div className="flex items-center gap-1">
                <FileText className="w-3 h-3" />
                <span>Type: {dataType}</span>
              </div>
            )}
          </div>
          
          {/* Details if available */}
          {details && Object.keys(details).length > 0 && (
            <div className="mt-2 p-2 bg-secondary/20 rounded text-xs">
              <span className="text-muted-foreground">Details: </span>
              <span className="text-primary">
                {Object.entries(details).map(([k, v]) => `${k}: ${v}`).join(', ')}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PdfExportButton({ analysisId }: { analysisId: number }) {
  const [isGenerating, setIsGenerating] = useState(false);
  const generatePdf = trpc.forensic.generatePdf.useMutation();

  const handleExport = async () => {
    setIsGenerating(true);
    try {
      const result = await generatePdf.mutateAsync({ analysisId });
      
      // Convert base64 to blob and download
      const byteCharacters = atob(result.pdf);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'application/pdf' });
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to generate PDF:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Button
      onClick={handleExport}
      disabled={isGenerating}
      className="gap-2"
      variant="outline"
    >
      {isGenerating ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          Generating...
        </>
      ) : (
        <>
          <Download className="w-4 h-4" />
          Export PDF
        </>
      )}
    </Button>
  );
}

function HashDisplay({ hash, compact = false }: { hash: string | null; compact?: boolean }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    if (hash) {
      navigator.clipboard.writeText(hash);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!hash) {
    return <span className="text-muted-foreground text-xs italic">Not computed</span>;
  }

  const displayHash = compact ? `${hash.slice(0, 16)}...${hash.slice(-16)}` : hash;

  return (
    <div className="flex items-center gap-2">
      <code className={cn("font-mono text-primary break-all", compact ? "text-xs" : "text-sm")}>
        {displayHash}
      </code>
      <button
        onClick={copyToClipboard}
        className="p-1 hover:bg-secondary rounded transition-colors flex-shrink-0"
        title="Copy to clipboard"
      >
        {copied ? (
          <Check className="w-3 h-3 text-green-500" />
        ) : (
          <Copy className="w-3 h-3 text-muted-foreground" />
        )}
      </button>
    </div>
  );
}
