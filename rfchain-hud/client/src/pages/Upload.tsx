import { JarvisLayout } from "@/components/JarvisLayout";
import { JarvisPanel, JarvisProgress, JarvisStatusIndicator } from "@/components/JarvisPanel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Upload as UploadIcon, FileAudio, X, CheckCircle, AlertCircle, Eye, Settings2, Layers, Play, Pause, RotateCcw } from "lucide-react";
import { useState, useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";
import { Link } from "wouter";

interface UploadedFile {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "analyzing" | "completed" | "error";
  error?: string;
  dbId?: number;
  analysisId?: number;
  localPath?: string;
}

interface AnalysisConfig {
  sampleRate: number;
  centerFreq: number;
  dataFormat: string;
  enableDigital: boolean;
  enableV3: boolean;
}

interface BatchJob {
  id: number;
  name: string;
  totalFiles: number;
  completedFiles: number;
  failedFiles: number;
  status: 'pending' | 'processing' | 'completed' | 'cancelled';
  progress: number;
}

export default function Upload() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [batchMode, setBatchMode] = useState(false);
  const [activeBatchJob, setActiveBatchJob] = useState<BatchJob | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const processingRef = useRef(false);
  
  const [config, setConfig] = useState<AnalysisConfig>({
    sampleRate: 1e6,
    centerFreq: 0,
    dataFormat: "complex64",
    enableDigital: false,
    enableV3: false,
  });

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      (file) => file.name.endsWith(".bin") || file.name.endsWith(".raw") || file.name.endsWith(".iq")
    );
    
    addFiles(droppedFiles);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      addFiles(selectedFiles);
    }
  }, []);

  const addFiles = (newFiles: File[]) => {
    const uploadFiles: UploadedFile[] = newFiles.map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      file,
      progress: 0,
      status: "pending",
    }));
    
    setFiles((prev) => [...prev, ...uploadFiles]);
    
    // Auto-enable batch mode if multiple files
    if (newFiles.length > 1 || files.length + newFiles.length > 1) {
      setBatchMode(true);
    }
  };

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const createUpload = trpc.signal.createUpload.useMutation();
  const completeUpload = trpc.signal.completeUpload.useMutation();
  const saveAnalysis = trpc.analysis.save.useMutation();
  const createBatch = trpc.batch.create.useMutation();
  const startBatch = trpc.batch.start.useMutation();
  const updateBatchItem = trpc.batch.updateItem.useMutation();
  const incrementProgress = trpc.batch.incrementProgress.useMutation();
  const cancelBatch = trpc.batch.cancel.useMutation();

  // Process a single file
  const processFile = async (uploadFile: UploadedFile): Promise<{ success: boolean; analysisId?: number }> => {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === uploadFile.id ? { ...f, status: "uploading", progress: 0 } : f
      )
    );

    try {
      // Step 1: Create upload record in database
      const { uploadId, s3Key } = await createUpload.mutateAsync({
        filename: uploadFile.file.name,
        originalName: uploadFile.file.name,
        fileSize: uploadFile.file.size,
        mimeType: uploadFile.file.type || "application/octet-stream",
      });

      if (!uploadId) {
        throw new Error("Failed to create upload record");
      }

      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, progress: 20, dbId: uploadId } : f
        )
      );

      // Step 2: Upload file to server (saves locally)
      const formData = new FormData();
      formData.append("file", uploadFile.file);
      formData.append("uploadId", uploadId.toString());

      const uploadResponse = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error("Upload failed");
      }

      const { localPath } = await uploadResponse.json();

      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, progress: 40, localPath } : f
        )
      );

      // Step 3: Complete upload record with local path
      await completeUpload.mutateAsync({
        uploadId,
        s3Url: localPath,
      });

      // Update to analyzing
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, status: "analyzing", progress: 50 } : f
        )
      );

      // Step 4: Trigger analysis with local path
      const analysisResponse = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          localPath,
          uploadId,
          sampleRate: config.sampleRate,
          centerFreq: config.centerFreq,
          dataFormat: config.dataFormat,
          enableDigital: config.enableDigital,
          enableV3: config.enableV3,
        }),
      });

      if (!analysisResponse.ok) {
        const error = await analysisResponse.json();
        throw new Error(error.error || "Analysis failed");
      }

      const analysisResult = await analysisResponse.json();

      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, progress: 80 } : f
        )
      );

      // Step 5: Save analysis results to database
      const { analysisId } = await saveAnalysis.mutateAsync({
        signalUploadId: uploadId,
        metricsUrl: analysisResult.metricsUrl,
        plotUrls: analysisResult.plotUrls,
        metrics: analysisResult.metrics,
      });

      // Mark as completed
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id 
            ? { ...f, status: "completed", progress: 100, dbId: uploadId, analysisId } 
            : f
        )
      );

      return { success: true, analysisId };
    } catch (error) {
      console.error("Upload/analysis error:", error);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id
            ? { ...f, status: "error", error: error instanceof Error ? error.message : "Upload failed" }
            : f
        )
      );
      return { success: false };
    }
  };

  // Single file upload (original behavior)
  const startUpload = async (uploadFile: UploadedFile) => {
    const result = await processFile(uploadFile);
    if (result.success) {
      toast.success(`${uploadFile.file.name} analyzed successfully!`);
    } else {
      toast.error(`Failed to process ${uploadFile.file.name}`);
    }
  };

  // Batch processing
  const startBatchProcessing = async () => {
    const pendingFiles = files.filter((f) => f.status === "pending");
    if (pendingFiles.length === 0) {
      toast.error("No pending files to process");
      return;
    }

    setIsProcessing(true);
    processingRef.current = true;

    // First, upload all files to get local paths
    const uploadedFiles: { filename: string; localPath: string; fileId: string }[] = [];
    
    for (const uploadFile of pendingFiles) {
      if (!processingRef.current) break;
      
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, status: "uploading", progress: 10 } : f
        )
      );

      try {
        // Create upload record
        const { uploadId } = await createUpload.mutateAsync({
          filename: uploadFile.file.name,
          originalName: uploadFile.file.name,
          fileSize: uploadFile.file.size,
          mimeType: uploadFile.file.type || "application/octet-stream",
        });

        if (!uploadId) throw new Error("Failed to create upload record");

        // Upload file
        const formData = new FormData();
        formData.append("file", uploadFile.file);
        formData.append("uploadId", uploadId.toString());

        const uploadResponse = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });

        if (!uploadResponse.ok) throw new Error("Upload failed");

        const { localPath } = await uploadResponse.json();

        // Complete upload record
        await completeUpload.mutateAsync({
          uploadId,
          s3Url: localPath,
        });

        uploadedFiles.push({
          filename: uploadFile.file.name,
          localPath,
          fileId: uploadFile.id,
        });

        setFiles((prev) =>
          prev.map((f) =>
            f.id === uploadFile.id ? { ...f, progress: 30, dbId: uploadId, localPath } : f
          )
        );
      } catch (error) {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === uploadFile.id
              ? { ...f, status: "error", error: error instanceof Error ? error.message : "Upload failed" }
              : f
          )
        );
      }
    }

    if (uploadedFiles.length === 0) {
      setIsProcessing(false);
      processingRef.current = false;
      toast.error("No files were uploaded successfully");
      return;
    }

    // Create batch job
    try {
      const { jobId } = await createBatch.mutateAsync({
        name: `Batch ${new Date().toLocaleString()}`,
        files: uploadedFiles.map(f => ({ filename: f.filename, localPath: f.localPath })),
        options: {
          sampleRate: config.sampleRate,
          dataFormat: config.dataFormat,
          digitalAnalysis: config.enableDigital,
          v3Analysis: config.enableV3,
        },
      });

      // Start the batch job
      await startBatch.mutateAsync({ jobId });

      setActiveBatchJob({
        id: jobId,
        name: `Batch ${new Date().toLocaleString()}`,
        totalFiles: uploadedFiles.length,
        completedFiles: 0,
        failedFiles: 0,
        status: 'processing',
        progress: 0,
      });

      toast.info(`Batch processing started: ${uploadedFiles.length} files`);

      // Process files sequentially
      for (let i = 0; i < uploadedFiles.length; i++) {
        if (!processingRef.current) {
          await cancelBatch.mutateAsync({ jobId });
          break;
        }

        const fileInfo = uploadedFiles[i];
        const uploadFile = files.find(f => f.id === fileInfo.fileId);
        if (!uploadFile) continue;

        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileInfo.fileId ? { ...f, status: "analyzing", progress: 50 } : f
          )
        );

        try {
          // Run analysis
          const analysisResponse = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              localPath: fileInfo.localPath,
              uploadId: uploadFile.dbId,
              sampleRate: config.sampleRate,
              centerFreq: config.centerFreq,
              dataFormat: config.dataFormat,
              enableDigital: config.enableDigital,
              enableV3: config.enableV3,
            }),
          });

          if (!analysisResponse.ok) {
            throw new Error("Analysis failed");
          }

          const analysisResult = await analysisResponse.json();

          // Save analysis results
          const { analysisId } = await saveAnalysis.mutateAsync({
            signalUploadId: uploadFile.dbId!,
            metricsUrl: analysisResult.metricsUrl,
            plotUrls: analysisResult.plotUrls,
            metrics: analysisResult.metrics,
          });

          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileInfo.fileId
                ? { ...f, status: "completed", progress: 100, analysisId }
                : f
            )
          );

          await incrementProgress.mutateAsync({ jobId, success: true });

          setActiveBatchJob(prev => prev ? {
            ...prev,
            completedFiles: prev.completedFiles + 1,
            progress: Math.round(((prev.completedFiles + prev.failedFiles + 1) / prev.totalFiles) * 100),
          } : null);

        } catch (error) {
          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileInfo.fileId
                ? { ...f, status: "error", error: error instanceof Error ? error.message : "Analysis failed" }
                : f
            )
          );

          await incrementProgress.mutateAsync({ jobId, success: false });

          setActiveBatchJob(prev => prev ? {
            ...prev,
            failedFiles: prev.failedFiles + 1,
            progress: Math.round(((prev.completedFiles + prev.failedFiles + 1) / prev.totalFiles) * 100),
          } : null);
        }
      }

      // Update final status
      setActiveBatchJob(prev => prev ? { ...prev, status: 'completed' } : null);
      
      const completed = files.filter(f => f.status === 'completed').length;
      const failed = files.filter(f => f.status === 'error').length;
      
      if (failed === 0) {
        toast.success(`Batch complete: ${completed} files analyzed successfully!`);
      } else {
        toast.warning(`Batch complete: ${completed} succeeded, ${failed} failed`);
      }

    } catch (error) {
      toast.error(`Batch processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }

    setIsProcessing(false);
    processingRef.current = false;
  };

  const pauseBatchProcessing = () => {
    processingRef.current = false;
    setIsProcessing(false);
    toast.info("Batch processing paused");
  };

  const uploadAll = async () => {
    if (batchMode) {
      await startBatchProcessing();
    } else {
      const pendingFiles = files.filter((f) => f.status === "pending");
      for (const file of pendingFiles) {
        await startUpload(file);
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const pendingCount = files.filter(f => f.status === 'pending').length;
  const completedCount = files.filter(f => f.status === 'completed').length;
  const failedCount = files.filter(f => f.status === 'error').length;

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-wider jarvis-text">SIGNAL UPLOAD</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Upload RF signal files for forensic analysis
            </p>
          </div>
          
          {/* Batch Mode Toggle */}
          <div className="flex items-center gap-3 bg-secondary/30 rounded-lg px-4 py-2 border border-border/30">
            <Layers className="w-4 h-4 text-primary" />
            <Label htmlFor="batchMode" className="text-sm">Batch Mode</Label>
            <Switch
              id="batchMode"
              checked={batchMode}
              onCheckedChange={setBatchMode}
            />
          </div>
        </div>

        {/* Batch Progress Panel */}
        {activeBatchJob && (
          <JarvisPanel title="Batch Processing Status" glowPulse={activeBatchJob.status === 'processing'}>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{activeBatchJob.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {activeBatchJob.completedFiles + activeBatchJob.failedFiles} / {activeBatchJob.totalFiles} files processed
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {activeBatchJob.status === 'processing' && (
                    <JarvisStatusIndicator status="processing" label="Processing" />
                  )}
                  {activeBatchJob.status === 'completed' && (
                    <JarvisStatusIndicator status="online" label="Completed" />
                  )}
                  {activeBatchJob.status === 'cancelled' && (
                    <JarvisStatusIndicator status="offline" label="Cancelled" />
                  )}
                </div>
              </div>
              
              <JarvisProgress value={activeBatchJob.progress} showValue />
              
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-secondary/30 rounded-lg p-3">
                  <p className="text-2xl font-bold text-primary">{activeBatchJob.totalFiles}</p>
                  <p className="text-xs text-muted-foreground">Total Files</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3">
                  <p className="text-2xl font-bold text-green-500">{activeBatchJob.completedFiles}</p>
                  <p className="text-xs text-muted-foreground">Completed</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3">
                  <p className="text-2xl font-bold text-red-500">{activeBatchJob.failedFiles}</p>
                  <p className="text-xs text-muted-foreground">Failed</p>
                </div>
              </div>
            </div>
          </JarvisPanel>
        )}

        {/* Upload Zone */}
        <JarvisPanel title="Upload Zone" glowPulse={isDragging}>
          <div
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300",
              isDragging
                ? "border-primary bg-primary/10"
                : "border-border hover:border-primary/50"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
                <UploadIcon className="w-8 h-8 text-primary" />
              </div>
              <div>
                <p className="text-lg font-medium">
                  {batchMode ? "Drag & drop multiple signal files here" : "Drag & drop signal files here"}
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  Supported formats: .bin, .raw, .iq {batchMode && "• Batch processing enabled"}
                </p>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="h-px w-12 bg-border" />
                <span className="text-xs uppercase">or</span>
                <div className="h-px w-12 bg-border" />
              </div>
              <label>
                <input
                  type="file"
                  className="hidden"
                  accept=".bin,.raw,.iq"
                  multiple
                  onChange={handleFileSelect}
                />
                <Button variant="outline" className="cursor-pointer" asChild>
                  <span>Browse Files</span>
                </Button>
              </label>
            </div>
          </div>
        </JarvisPanel>

        {/* Analysis Configuration */}
        <JarvisPanel title="Analysis Configuration">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">Advanced Settings</span>
              </div>
              <Switch
                checked={showAdvanced}
                onCheckedChange={setShowAdvanced}
              />
            </div>

            {showAdvanced && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-4 border-t border-border/50">
                <div className="space-y-2">
                  <Label htmlFor="sampleRate">Sample Rate (Hz)</Label>
                  <Input
                    id="sampleRate"
                    type="number"
                    value={config.sampleRate}
                    onChange={(e) => setConfig({ ...config, sampleRate: parseFloat(e.target.value) || 1e6 })}
                    placeholder="1000000"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="centerFreq">Center Frequency (Hz)</Label>
                  <Input
                    id="centerFreq"
                    type="number"
                    value={config.centerFreq}
                    onChange={(e) => setConfig({ ...config, centerFreq: parseFloat(e.target.value) || 0 })}
                    placeholder="0"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="dataFormat">Data Format</Label>
                  <Select
                    value={config.dataFormat}
                    onValueChange={(value) => setConfig({ ...config, dataFormat: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="complex64">Complex64 (I/Q float32)</SelectItem>
                      <SelectItem value="int16">Int16 (I/Q interleaved)</SelectItem>
                      <SelectItem value="int8">Int8 (I/Q interleaved)</SelectItem>
                      <SelectItem value="float32">Float32 (real only)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center gap-3">
                  <Switch
                    id="enableDigital"
                    checked={config.enableDigital}
                    onCheckedChange={(checked) => setConfig({ ...config, enableDigital: checked })}
                  />
                  <Label htmlFor="enableDigital">Enable Digital Analysis</Label>
                </div>

                <div className="flex items-center gap-3">
                  <Switch
                    id="enableV3"
                    checked={config.enableV3}
                    onCheckedChange={(checked) => setConfig({ ...config, enableV3: checked })}
                  />
                  <Label htmlFor="enableV3">Enable V3 Enhanced Analysis</Label>
                </div>
              </div>
            )}
          </div>
        </JarvisPanel>

        {/* File Queue */}
        {files.length > 0 && (
          <JarvisPanel title={`Upload Queue (${files.length} files)`}>
            {/* Queue Summary */}
            {files.length > 1 && (
              <div className="grid grid-cols-4 gap-4 mb-4 text-center">
                <div className="bg-secondary/30 rounded-lg p-2">
                  <p className="text-lg font-bold">{files.length}</p>
                  <p className="text-xs text-muted-foreground">Total</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-2">
                  <p className="text-lg font-bold text-yellow-500">{pendingCount}</p>
                  <p className="text-xs text-muted-foreground">Pending</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-2">
                  <p className="text-lg font-bold text-green-500">{completedCount}</p>
                  <p className="text-xs text-muted-foreground">Done</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-2">
                  <p className="text-lg font-bold text-red-500">{failedCount}</p>
                  <p className="text-xs text-muted-foreground">Failed</p>
                </div>
              </div>
            )}
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {files.map((uploadFile, index) => (
                <div
                  key={uploadFile.id}
                  className="flex items-center gap-4 p-3 bg-secondary/30 rounded-lg border border-border/30"
                >
                  <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center flex-shrink-0 text-xs font-mono">
                    {index + 1}
                  </div>
                  <div className="w-10 h-10 rounded bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <FileAudio className="w-5 h-5 text-primary" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-medium text-sm truncate">{uploadFile.file.name}</p>
                      <span className="text-xs text-muted-foreground ml-2">
                        {formatFileSize(uploadFile.file.size)}
                      </span>
                    </div>
                    
                    {(uploadFile.status === "uploading" || uploadFile.status === "analyzing") && (
                      <JarvisProgress value={uploadFile.progress} showValue />
                    )}
                    
                    {uploadFile.status === "analyzing" && (
                      <div className="flex items-center gap-2 mt-1">
                        <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                        <span className="text-xs text-yellow-500">Running forensic analysis...</span>
                      </div>
                    )}
                    
                    {uploadFile.status === "error" && (
                      <p className="text-xs text-red-500 mt-1">{uploadFile.error}</p>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {uploadFile.status === "pending" && (
                      <JarvisStatusIndicator status="idle" label="Pending" />
                    )}
                    {uploadFile.status === "uploading" && (
                      <JarvisStatusIndicator status="processing" label="Uploading" />
                    )}
                    {uploadFile.status === "analyzing" && (
                      <JarvisStatusIndicator status="processing" label="Analyzing" />
                    )}
                    {uploadFile.status === "completed" && (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    )}
                    {uploadFile.status === "error" && (
                      <AlertCircle className="w-5 h-5 text-red-500" />
                    )}
                    
                    {uploadFile.status === "pending" && !isProcessing && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => removeFile(uploadFile.id)}
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    )}
                    {uploadFile.status === "completed" && uploadFile.dbId && (
                      <Link href={`/analysis/${uploadFile.dbId}`}>
                        <Button variant="ghost" size="icon">
                          <Eye className="w-4 h-4" />
                        </Button>
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
            
            {pendingCount > 0 && (
              <div className="mt-4 flex justify-between items-center">
                <div className="flex items-center gap-2">
                  {isProcessing && (
                    <Button variant="outline" onClick={pauseBatchProcessing} className="gap-2">
                      <Pause className="w-4 h-4" />
                      Pause
                    </Button>
                  )}
                  {!isProcessing && completedCount > 0 && failedCount > 0 && (
                    <Button variant="outline" onClick={() => {
                      // Reset failed files to pending
                      setFiles(prev => prev.map(f => 
                        f.status === 'error' ? { ...f, status: 'pending', error: undefined, progress: 0 } : f
                      ));
                    }} className="gap-2">
                      <RotateCcw className="w-4 h-4" />
                      Retry Failed
                    </Button>
                  )}
                </div>
                <Button 
                  onClick={uploadAll} 
                  disabled={isProcessing}
                  className="gap-2"
                >
                  {isProcessing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      {batchMode ? <Layers className="w-4 h-4" /> : <UploadIcon className="w-4 h-4" />}
                      {batchMode ? `Process Batch (${pendingCount})` : "Upload & Analyze All"}
                    </>
                  )}
                </Button>
              </div>
            )}
          </JarvisPanel>
        )}

        {/* Instructions */}
        <JarvisPanel title="Analysis Information">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-semibold uppercase tracking-wider mb-2 jarvis-text">
                Supported File Types
              </h4>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• <strong>.bin</strong> - Raw binary I/Q samples</li>
                <li>• <strong>.raw</strong> - Unprocessed signal data</li>
                <li>• <strong>.iq</strong> - Interleaved I/Q format</li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold uppercase tracking-wider mb-2 jarvis-text">
                {batchMode ? "Batch Processing" : "Analysis Pipeline"}
              </h4>
              <ul className="space-y-1 text-sm text-muted-foreground">
                {batchMode ? (
                  <>
                    <li>• Sequential processing with progress tracking</li>
                    <li>• Automatic retry for failed files</li>
                    <li>• Shared configuration across all files</li>
                    <li>• Pause/resume capability</li>
                  </>
                ) : (
                  <>
                    <li>• 6-stage forensic hash chain verification</li>
                    <li>• NIST SP 800-86 compliant processing</li>
                    <li>• 15-17 visualization outputs</li>
                    <li>• Dual-algorithm hashing (SHA-256 + SHA3-256)</li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </JarvisPanel>
      </div>
    </JarvisLayout>
  );
}
