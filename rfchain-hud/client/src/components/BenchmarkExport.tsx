import { Button } from '@/components/ui/button';
import { Download, FileText, FileSpreadsheet } from 'lucide-react';
import { toast } from 'sonner';

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
  errorMessage: string | null;
}

interface BenchmarkExportProps {
  history: BenchmarkRecord[];
  gpuName?: string;
}

export function BenchmarkExport({ history, gpuName }: BenchmarkExportProps) {
  const exportToCSV = () => {
    if (!history || history.length === 0) {
      toast.error('No benchmark data to export');
      return;
    }

    const headers = [
      'Date',
      'Time',
      'GPU Name',
      'GPU Memory (MB)',
      'Avg Speedup',
      'Max Speedup',
      'Min Speedup',
      'Status',
      'Error Message',
    ];

    const rows = history.map(record => [
      new Date(record.createdAt).toLocaleDateString(),
      new Date(record.createdAt).toLocaleTimeString(),
      record.gpuName || 'Unknown',
      record.gpuMemoryMb || '',
      record.avgSpeedup?.toFixed(2) || '',
      record.maxSpeedup?.toFixed(2) || '',
      record.minSpeedup?.toFixed(2) || '',
      record.success === 1 ? 'Success' : 'Failed',
      record.errorMessage || '',
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `benchmark-history-${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast.success('CSV exported successfully');
  };

  const exportToPDF = () => {
    if (!history || history.length === 0) {
      toast.error('No benchmark data to export');
      return;
    }

    // Generate HTML content for PDF
    const successfulRuns = history.filter(h => h.success === 1);
    const avgSpeedups = successfulRuns.filter(h => h.avgSpeedup).map(h => h.avgSpeedup as number);
    const maxSpeedups = successfulRuns.filter(h => h.maxSpeedup).map(h => h.maxSpeedup as number);
    
    const overallAvg = avgSpeedups.length > 0 
      ? (avgSpeedups.reduce((a, b) => a + b, 0) / avgSpeedups.length).toFixed(2)
      : 'N/A';
    const bestSpeedup = maxSpeedups.length > 0 
      ? Math.max(...maxSpeedups).toFixed(2)
      : 'N/A';

    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>GPU Benchmark Report</title>
        <style>
          body { font-family: 'Segoe UI', Arial, sans-serif; padding: 40px; color: #333; }
          h1 { color: #0891b2; border-bottom: 2px solid #0891b2; padding-bottom: 10px; }
          h2 { color: #0e7490; margin-top: 30px; }
          .summary { background: #f0f9ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
          .summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
          .summary-item { text-align: center; }
          .summary-value { font-size: 24px; font-weight: bold; color: #0891b2; }
          .summary-label { font-size: 12px; color: #666; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; }
          th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
          th { background: #f8fafc; font-weight: 600; color: #374151; }
          tr:hover { background: #f9fafb; }
          .success { color: #059669; }
          .failed { color: #dc2626; }
          .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <h1>🚀 GPU Benchmark Report</h1>
        <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
        <p><strong>GPU:</strong> ${gpuName || history[0]?.gpuName || 'Unknown'}</p>
        
        <div class="summary">
          <h2 style="margin-top: 0;">Performance Summary</h2>
          <div class="summary-grid">
            <div class="summary-item">
              <div class="summary-value">${successfulRuns.length}</div>
              <div class="summary-label">Total Runs</div>
            </div>
            <div class="summary-item">
              <div class="summary-value">${overallAvg}x</div>
              <div class="summary-label">Average Speedup</div>
            </div>
            <div class="summary-item">
              <div class="summary-value">${bestSpeedup}x</div>
              <div class="summary-label">Best Speedup</div>
            </div>
          </div>
        </div>
        
        <h2>Benchmark History</h2>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>GPU</th>
              <th>Avg Speedup</th>
              <th>Max Speedup</th>
              <th>Min Speedup</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${history.map(record => `
              <tr>
                <td>${new Date(record.createdAt).toLocaleString()}</td>
                <td>${record.gpuName || 'Unknown'}</td>
                <td>${record.avgSpeedup?.toFixed(2) || '-'}x</td>
                <td>${record.maxSpeedup?.toFixed(2) || '-'}x</td>
                <td>${record.minSpeedup?.toFixed(2) || '-'}x</td>
                <td class="${record.success === 1 ? 'success' : 'failed'}">
                  ${record.success === 1 ? '✓ Success' : '✗ Failed'}
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
        
        <div class="footer">
          <p>RFChain HUD - GPU Benchmark Report</p>
          <p>This report was automatically generated by the RFChain Signal Intelligence System.</p>
        </div>
      </body>
      </html>
    `;

    // Open in new window for printing/saving as PDF
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(htmlContent);
      printWindow.document.close();
      printWindow.onload = () => {
        printWindow.print();
      };
      toast.success('PDF report opened - use Print to save as PDF');
    } else {
      toast.error('Please allow popups to generate PDF');
    }
  };

  return (
    <div className="flex gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={exportToCSV}
        className="gap-2"
      >
        <FileSpreadsheet className="w-4 h-4" />
        CSV
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={exportToPDF}
        className="gap-2"
      >
        <FileText className="w-4 h-4" />
        PDF
      </Button>
    </div>
  );
}
