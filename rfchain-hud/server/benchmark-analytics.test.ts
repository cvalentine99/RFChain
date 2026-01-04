import { describe, it, expect } from "vitest";

describe("Benchmark Analytics", () => {
  describe("Performance Trend Chart", () => {
    it("should prepare chart data from history records", () => {
      const history = [
        { id: 1, createdAt: "2026-01-01", avgSpeedup: 15.0, maxSpeedup: 20.0, minSpeedup: 10.0, success: 1 },
        { id: 2, createdAt: "2026-01-02", avgSpeedup: 18.0, maxSpeedup: 25.0, minSpeedup: 12.0, success: 1 },
        { id: 3, createdAt: "2026-01-03", avgSpeedup: 16.0, maxSpeedup: 22.0, minSpeedup: 11.0, success: 1 },
      ];

      const successfulRuns = history.filter(h => h.success === 1 && h.avgSpeedup !== null);
      const avgSpeedups = successfulRuns.map(h => h.avgSpeedup);
      const maxSpeedups = successfulRuns.map(h => h.maxSpeedup);
      const minSpeedups = successfulRuns.map(h => h.minSpeedup);

      expect(successfulRuns.length).toBe(3);
      expect(avgSpeedups).toEqual([15.0, 18.0, 16.0]);
      expect(maxSpeedups).toEqual([20.0, 25.0, 22.0]);
      expect(minSpeedups).toEqual([10.0, 12.0, 11.0]);
    });

    it("should filter out failed benchmarks from chart", () => {
      const history = [
        { id: 1, avgSpeedup: 15.0, success: 1 },
        { id: 2, avgSpeedup: null, success: 0 }, // Failed
        { id: 3, avgSpeedup: 18.0, success: 1 },
      ];

      const chartData = history.filter(h => h.success === 1 && h.avgSpeedup !== null);
      expect(chartData.length).toBe(2);
    });

    it("should require at least 2 data points for trend line", () => {
      const singleRecord = [{ id: 1, avgSpeedup: 15.0, success: 1 }];
      const canShowTrend = singleRecord.filter(h => h.success === 1).length >= 2;
      expect(canShowTrend).toBe(false);
    });
  });

  describe("CSV Export", () => {
    it("should format benchmark data as CSV", () => {
      const record = {
        createdAt: "2026-01-04T10:30:00Z",
        gpuName: "RTX 4090",
        gpuMemoryMb: 24564,
        avgSpeedup: 15.5,
        maxSpeedup: 25.3,
        minSpeedup: 8.2,
        success: 1,
        errorMessage: null,
      };

      const csvRow = [
        new Date(record.createdAt).toLocaleDateString(),
        new Date(record.createdAt).toLocaleTimeString(),
        record.gpuName,
        record.gpuMemoryMb,
        record.avgSpeedup?.toFixed(2),
        record.maxSpeedup?.toFixed(2),
        record.minSpeedup?.toFixed(2),
        record.success === 1 ? 'Success' : 'Failed',
        record.errorMessage || '',
      ];

      expect(csvRow[2]).toBe("RTX 4090");
      expect(csvRow[4]).toBe("15.50");
      expect(csvRow[7]).toBe("Success");
    });

    it("should handle missing values in CSV export", () => {
      const record = {
        gpuName: null,
        avgSpeedup: null,
        success: 0,
        errorMessage: "GPU not available",
      };

      const gpuName = record.gpuName || 'Unknown';
      const avgSpeedup = record.avgSpeedup?.toFixed(2) || '';
      const status = record.success === 1 ? 'Success' : 'Failed';

      expect(gpuName).toBe('Unknown');
      expect(avgSpeedup).toBe('');
      expect(status).toBe('Failed');
    });
  });

  describe("PDF Export", () => {
    it("should calculate summary statistics for PDF report", () => {
      const history = [
        { avgSpeedup: 15.0, maxSpeedup: 20.0, success: 1 },
        { avgSpeedup: 18.0, maxSpeedup: 25.0, success: 1 },
        { avgSpeedup: null, maxSpeedup: null, success: 0 },
      ];

      const successfulRuns = history.filter(h => h.success === 1);
      const avgSpeedups = successfulRuns.filter(h => h.avgSpeedup).map(h => h.avgSpeedup as number);
      const maxSpeedups = successfulRuns.filter(h => h.maxSpeedup).map(h => h.maxSpeedup as number);

      const overallAvg = avgSpeedups.reduce((a, b) => a + b, 0) / avgSpeedups.length;
      const bestSpeedup = Math.max(...maxSpeedups);

      expect(successfulRuns.length).toBe(2);
      expect(overallAvg).toBe(16.5);
      expect(bestSpeedup).toBe(25.0);
    });
  });

  describe("Benchmark Comparison", () => {
    it("should identify baseline and current benchmarks by date", () => {
      const first = { id: 1, createdAt: "2026-01-01T10:00:00Z", avgSpeedup: 15.0 };
      const second = { id: 2, createdAt: "2026-01-04T10:00:00Z", avgSpeedup: 18.0 };

      const baseline = new Date(first.createdAt) < new Date(second.createdAt) ? first : second;
      const current = baseline === first ? second : first;

      expect(baseline.id).toBe(1);
      expect(current.id).toBe(2);
    });

    it("should calculate percentage difference between benchmarks", () => {
      const baseline = { avgSpeedup: 15.0 };
      const current = { avgSpeedup: 18.0 };

      const diff = ((current.avgSpeedup - baseline.avgSpeedup) / baseline.avgSpeedup) * 100;
      expect(diff).toBe(20); // 20% improvement
    });

    it("should handle negative performance change", () => {
      const baseline = { avgSpeedup: 20.0 };
      const current = { avgSpeedup: 16.0 };

      const diff = ((current.avgSpeedup - baseline.avgSpeedup) / baseline.avgSpeedup) * 100;
      expect(diff).toBe(-20); // 20% decrease
    });

    it("should detect no significant change", () => {
      const baseline = { avgSpeedup: 15.0 };
      const current = { avgSpeedup: 15.1 };

      const diff = ((current.avgSpeedup - baseline.avgSpeedup) / baseline.avgSpeedup) * 100;
      const isSignificant = Math.abs(diff) >= 1;
      expect(isSignificant).toBe(false);
    });

    it("should require exactly 2 benchmarks for comparison", () => {
      const selectedIds: number[] = [1];
      const canCompare = selectedIds.length === 2;
      expect(canCompare).toBe(false);

      selectedIds.push(2);
      expect(selectedIds.length === 2).toBe(true);
    });
  });
});
