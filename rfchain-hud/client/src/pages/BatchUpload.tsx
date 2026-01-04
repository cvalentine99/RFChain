import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { trpc } from '@/lib/trpc';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Upload, FileUp, Trash2, Play, Pause, CheckCircle, XCircle, Clock, Loader2 } from 'lucide-react';
import { Link } from 'wouter';

interface QueuedFile {
  id: string;
  file: File;
  status: 'queued' | 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  analysisId?: number;
}

export default function BatchUpload() {
  const [files, setFiles] = useState<QueuedFile[]>([]);
  const [jobName, setJobName] = useState('');
  const [sampleRate, setSampleRate] = useState('1000000');
  const [dataFormat, setDataFormat] = useState('complex64');
  const [digitalAnalysis, setDigitalAnalysis] = useState(false);
  const [v3Analysis, setV3Analysis] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<number | null>(null);
  
  const createBatch = trpc.batch.create.useMutation();
  const startBatch = trpc.batch.start.useMutation();
  const { data: activeJob, refetch: refetchActiveJob } = trpc.batch.getActive.useQuery();
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file, index) => ({
      id: `${Date.now()}-${index}`,
      file,
      status: 'queued' as const,
      progress: 0,
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.bin', '.raw', '.iq', '.dat', '.cf32', '.cs16', '.cu8'],
    },
    multiple: true,
  });
  
  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };
  
  const clearCompleted = () => {
    setFiles(prev => prev.filter(f => f.status !== 'completed' && f.status !== 'failed'));
  };
  
  const startBatchProcessing = async () => {
    if (files.length === 0) return;
    
    setIsProcessing(true);
    
    try {
      // Upload files first and collect local paths
      const uploadedFiles: { filename: string; localPath: string }[] = [];
      
      for (let i = 0; i < files.length; i++) {
        const queuedFile = files[i];
        
        // Update status to uploading
        setFiles(prev => prev.map(f => 
          f.id === queuedFile.id ? { ...f, status: 'uploading' as const, progress: 0 } : f
        ));
        
        // Upload file
        const formData = new FormData();
        formData.append('file', queuedFile.file);
        formData.append('sampleRate', sampleRate);
        formData.append('dataFormat', dataFormat);
        
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error(`Upload failed for ${queuedFile.file.name}`);
        }
        
        const result = await response.json();
        
        uploadedFiles.push({
          filename: queuedFile.file.name,
          localPath: result.localPath,
        });
        
        // Update progress
        setFiles(prev => prev.map(f => 
          f.id === queuedFile.id ? { ...f, progress: 100 } : f
        ));
      }
      
      // Create batch job
      const batchResult = await createBatch.mutateAsync({
        name: jobName || `Batch ${new Date().toLocaleString()}`,
        files: uploadedFiles,
        options: {
          sampleRate: parseFloat(sampleRate),
          dataFormat,
          digitalAnalysis,
          v3Analysis,
        },
      });
      
      setCurrentJobId(batchResult.jobId);
      
      // Update all files to processing status
      setFiles(prev => prev.map(f => ({ ...f, status: 'processing' as const })));
      
      // Start batch processing
      await startBatch.mutateAsync({ jobId: batchResult.jobId });
      
      // Poll for completion (simplified - in production use WebSocket)
      // For now, just mark as processing and let user check status
      
    } catch (error) {
      console.error('Batch processing error:', error);
      setFiles(prev => prev.map(f => ({
        ...f,
        status: 'failed' as const,
        error: error instanceof Error ? error.message : 'Unknown error',
      })));
    } finally {
      setIsProcessing(false);
    }
  };
  
  const totalSize = files.reduce((acc, f) => acc + f.file.size, 0);
  const completedCount = files.filter(f => f.status === 'completed').length;
  const failedCount = files.filter(f => f.status === 'failed').length;
  const queuedCount = files.filter(f => f.status === 'queued').length;
  
  const getStatusIcon = (status: QueuedFile['status']) => {
    switch (status) {
      case 'queued': return <Clock className="h-4 w-4 text-gray-400" />;
      case 'uploading': return <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />;
      case 'processing': return <Loader2 className="h-4 w-4 text-yellow-400 animate-spin" />;
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-400" />;
    }
  };
  
  const getStatusBadge = (status: QueuedFile['status']) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      queued: 'secondary',
      uploading: 'default',
      processing: 'default',
      completed: 'default',
      failed: 'destructive',
    };
    return <Badge variant={variants[status]}>{status}</Badge>;
  };
  
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-cyan-400">Batch Analysis</h1>
          <p className="text-gray-400 mt-1">Upload and process multiple signal files at once</p>
        </div>
        <Link href="/batch-history">
          <Button variant="outline">View History</Button>
        </Link>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Area */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-cyan-400">Upload Files</CardTitle>
              <CardDescription>
                Drag and drop signal files or click to browse. Supports .bin, .raw, .iq, .dat, .cf32, .cs16, .cu8
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive 
                    ? 'border-cyan-400 bg-cyan-400/10' 
                    : 'border-gray-700 hover:border-gray-600'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="h-12 w-12 mx-auto mb-4 text-gray-500" />
                {isDragActive ? (
                  <p className="text-cyan-400">Drop files here...</p>
                ) : (
                  <>
                    <p className="text-gray-300 mb-2">Drag & drop signal files here</p>
                    <p className="text-gray-500 text-sm">or click to select files</p>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
          
          {/* File Queue */}
          {files.length > 0 && (
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-cyan-400">File Queue</CardTitle>
                  <CardDescription>
                    {files.length} files ({(totalSize / 1024 / 1024).toFixed(1)} MB total)
                  </CardDescription>
                </div>
                <Button variant="ghost" size="sm" onClick={clearCompleted}>
                  Clear Completed
                </Button>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {files.map((queuedFile) => (
                    <div
                      key={queuedFile.id}
                      className="flex items-center justify-between p-3 bg-gray-800 rounded-lg"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        {getStatusIcon(queuedFile.status)}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-200 truncate">
                            {queuedFile.file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {(queuedFile.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {getStatusBadge(queuedFile.status)}
                        {queuedFile.status === 'queued' && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeFile(queuedFile.id)}
                          >
                            <Trash2 className="h-4 w-4 text-red-400" />
                          </Button>
                        )}
                        {queuedFile.analysisId && (
                          <Link href={`/analysis/${queuedFile.analysisId}`}>
                            <Button variant="ghost" size="sm">View</Button>
                          </Link>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Progress Summary */}
                {isProcessing && (
                  <div className="mt-4 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Processing...</span>
                      <span className="text-gray-300">
                        {completedCount + failedCount} / {files.length}
                      </span>
                    </div>
                    <Progress 
                      value={((completedCount + failedCount) / files.length) * 100} 
                      className="h-2"
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
        
        {/* Settings Panel */}
        <div className="space-y-6">
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-cyan-400">Batch Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="jobName">Job Name</Label>
                <Input
                  id="jobName"
                  placeholder="My Batch Analysis"
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                  className="bg-gray-800 border-gray-700"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="sampleRate">Sample Rate (Hz)</Label>
                <Select value={sampleRate} onValueChange={setSampleRate}>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="250000">250 kHz</SelectItem>
                    <SelectItem value="500000">500 kHz</SelectItem>
                    <SelectItem value="1000000">1 MHz</SelectItem>
                    <SelectItem value="2000000">2 MHz</SelectItem>
                    <SelectItem value="2400000">2.4 MHz</SelectItem>
                    <SelectItem value="5000000">5 MHz</SelectItem>
                    <SelectItem value="10000000">10 MHz</SelectItem>
                    <SelectItem value="20000000">20 MHz</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="dataFormat">Data Format</Label>
                <Select value={dataFormat} onValueChange={setDataFormat}>
                  <SelectTrigger className="bg-gray-800 border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="complex64">Complex64 (float32 I/Q)</SelectItem>
                    <SelectItem value="complex128">Complex128 (float64 I/Q)</SelectItem>
                    <SelectItem value="int16">Int16 (signed 16-bit I/Q)</SelectItem>
                    <SelectItem value="uint8">Uint8 (unsigned 8-bit I/Q)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="digitalAnalysis">Digital Analysis</Label>
                <Switch
                  id="digitalAnalysis"
                  checked={digitalAnalysis}
                  onCheckedChange={setDigitalAnalysis}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="v3Analysis">V3 Enhanced Analysis</Label>
                <Switch
                  id="v3Analysis"
                  checked={v3Analysis}
                  onCheckedChange={setV3Analysis}
                />
              </div>
            </CardContent>
          </Card>
          
          {/* Action Buttons */}
          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="pt-6">
              <Button
                className="w-full bg-cyan-600 hover:bg-cyan-700"
                size="lg"
                disabled={files.length === 0 || isProcessing || queuedCount === 0}
                onClick={startBatchProcessing}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Batch ({queuedCount} files)
                  </>
                )}
              </Button>
              
              {/* Stats */}
              <div className="grid grid-cols-3 gap-2 mt-4 text-center">
                <div className="p-2 bg-gray-800 rounded">
                  <p className="text-2xl font-bold text-gray-300">{queuedCount}</p>
                  <p className="text-xs text-gray-500">Queued</p>
                </div>
                <div className="p-2 bg-gray-800 rounded">
                  <p className="text-2xl font-bold text-green-400">{completedCount}</p>
                  <p className="text-xs text-gray-500">Completed</p>
                </div>
                <div className="p-2 bg-gray-800 rounded">
                  <p className="text-2xl font-bold text-red-400">{failedCount}</p>
                  <p className="text-xs text-gray-500">Failed</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Active Job Info */}
          {activeJob && (
            <Card className="bg-gray-900 border-yellow-600">
              <CardHeader>
                <CardTitle className="text-yellow-400 text-sm">Active Job</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300 text-sm">{activeJob.name}</p>
                <Progress 
                  value={(activeJob.completedFiles / activeJob.totalFiles) * 100}
                  className="mt-2 h-2"
                />
                <p className="text-xs text-gray-500 mt-1">
                  {activeJob.completedFiles} / {activeJob.totalFiles} files
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
