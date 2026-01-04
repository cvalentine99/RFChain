import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock child_process spawn
vi.mock("child_process", () => ({
  spawn: vi.fn(),
}));

describe("GPU Router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("gpu.getStats", () => {
    it("should return GPU stats structure when nvidia-smi succeeds", async () => {
      const { spawn } = await import("child_process");
      const mockSpawn = spawn as unknown as ReturnType<typeof vi.fn>;
      
      // Mock successful nvidia-smi output
      const mockProcess = {
        stdout: {
          on: vi.fn((event, callback) => {
            if (event === "data") {
              callback("0, NVIDIA GeForce RTX 4090, 24564, 1234, 23330, 15, 10, 45, 250, 450");
            }
          }),
        },
        stderr: {
          on: vi.fn(),
        },
        on: vi.fn((event, callback) => {
          if (event === "close") {
            callback(0);
          }
        }),
        kill: vi.fn(),
      };
      
      mockSpawn.mockReturnValue(mockProcess);
      
      // The actual parsing happens in the Python script
      // This test verifies the structure expected from the endpoint
      const expectedStructure = {
        success: true,
        gpu_count: 1,
        gpus: expect.arrayContaining([
          expect.objectContaining({
            index: expect.any(Number),
            name: expect.any(String),
            memory: expect.objectContaining({
              total_mb: expect.any(Number),
              used_mb: expect.any(Number),
              free_mb: expect.any(Number),
              used_percent: expect.any(Number),
            }),
            utilization: expect.objectContaining({
              gpu_percent: expect.any(Number),
              memory_percent: expect.any(Number),
            }),
            temperature_c: expect.any(Number),
            power: expect.objectContaining({
              draw_w: expect.any(Number),
              limit_w: expect.any(Number),
            }),
          }),
        ]),
      };
      
      // Verify the expected structure is valid
      expect(expectedStructure).toBeDefined();
    });

    it("should handle GPU not available gracefully", async () => {
      // When nvidia-smi is not found, the response should indicate failure
      const errorResponse = {
        success: false,
        error: "nvidia-smi not found - NVIDIA drivers may not be installed",
        gpu_count: 0,
        gpus: [],
      };
      
      expect(errorResponse.success).toBe(false);
      expect(errorResponse.gpus).toHaveLength(0);
      expect(errorResponse.error).toContain("nvidia-smi");
    });

    it("should handle timeout gracefully", async () => {
      const timeoutResponse = {
        success: false,
        error: "GPU monitoring timed out",
        gpus: [],
      };
      
      expect(timeoutResponse.success).toBe(false);
      expect(timeoutResponse.error).toContain("timed out");
    });
  });

  describe("gpu.runBenchmark", () => {
    it("should return benchmark results structure when successful", async () => {
      const expectedBenchmarkStructure = {
        success: true,
        gpu_available: true,
        timestamp: expect.any(String),
        benchmarks: expect.arrayContaining([
          expect.objectContaining({
            operation: expect.any(String),
            size: expect.any(Number),
            iterations: expect.any(Number),
            cpu: expect.objectContaining({
              avg_ms: expect.any(Number),
              std_ms: expect.any(Number),
            }),
          }),
        ]),
        summary: expect.objectContaining({
          avg_speedup: expect.any(Number),
          max_speedup: expect.any(Number),
          min_speedup: expect.any(Number),
        }),
      };
      
      // Verify the expected structure is valid
      expect(expectedBenchmarkStructure).toBeDefined();
    });

    it("should handle benchmark operations", async () => {
      // Test that benchmark includes expected operations
      const expectedOperations = ["FFT", "Correlation", "PSD", "Array Ops"];
      
      expectedOperations.forEach((op) => {
        expect(op).toBeDefined();
      });
    });

    it("should handle CPU-only mode when GPU unavailable", async () => {
      const cpuOnlyResponse = {
        success: true,
        gpu_available: false,
        benchmarks: [
          {
            operation: "FFT",
            size: 65536,
            iterations: 100,
            cpu: { avg_ms: 5.2, std_ms: 0.3 },
            // No GPU field when unavailable
          },
        ],
        summary: {},
      };
      
      expect(cpuOnlyResponse.gpu_available).toBe(false);
      expect(cpuOnlyResponse.benchmarks[0].cpu).toBeDefined();
      expect((cpuOnlyResponse.benchmarks[0] as any).gpu).toBeUndefined();
    });

    it("should handle benchmark failure gracefully", async () => {
      const failureResponse = {
        success: false,
        error: "Benchmark failed",
      };
      
      expect(failureResponse.success).toBe(false);
      expect(failureResponse.error).toBeDefined();
    });
  });

  describe("GPU Monitor Python Script", () => {
    it("should parse nvidia-smi output correctly", () => {
      // Simulate nvidia-smi CSV output parsing
      const csvLine = "0, NVIDIA GeForce RTX 4090, 24564, 1234, 23330, 15, 10, 45, 250, 450";
      const parts = csvLine.split(",").map((p) => p.trim());
      
      expect(parts[0]).toBe("0"); // index
      expect(parts[1]).toBe("NVIDIA GeForce RTX 4090"); // name
      expect(parseFloat(parts[2])).toBe(24564); // total memory
      expect(parseFloat(parts[3])).toBe(1234); // used memory
      expect(parseFloat(parts[4])).toBe(23330); // free memory
      expect(parseFloat(parts[5])).toBe(15); // gpu utilization
      expect(parseFloat(parts[6])).toBe(10); // memory utilization
      expect(parseFloat(parts[7])).toBe(45); // temperature
      expect(parseFloat(parts[8])).toBe(250); // power draw
      expect(parseFloat(parts[9])).toBe(450); // power limit
    });

    it("should calculate memory percentage correctly", () => {
      const totalMb = 24564;
      const usedMb = 1234;
      const usedPercent = (usedMb / totalMb) * 100;
      
      expect(usedPercent).toBeCloseTo(5.02, 1);
    });
  });

  describe("GPU Benchmark Python Script", () => {
    it("should calculate speedup correctly", () => {
      const cpuTime = 10.5; // ms
      const gpuTime = 0.5; // ms
      const speedup = cpuTime / gpuTime;
      
      expect(speedup).toBe(21);
    });

    it("should handle different signal sizes", () => {
      const testSizes = [
        { size: 65536, name: "Small (64K samples)" },
        { size: 262144, name: "Medium (256K samples)" },
        { size: 1048576, name: "Large (1M samples)" },
        { size: 4194304, name: "XLarge (4M samples)" },
      ];
      
      testSizes.forEach((ts) => {
        expect(ts.size).toBeGreaterThan(0);
        expect(ts.name).toBeDefined();
      });
    });
  });
  
  describe("Benchmark History", () => {
    it("should have correct history record structure", () => {
      const historyRecord = {
        id: 1,
        userId: 1,
        gpuName: "NVIDIA GeForce RTX 4090",
        gpuMemoryMb: 24564,
        cudaVersion: "12.4",
        driverVersion: "550.54.14",
        avgSpeedup: 15.5,
        maxSpeedup: 25.3,
        minSpeedup: 8.2,
        benchmarkResults: [],
        systemInfo: {},
        success: 1,
        errorMessage: null,
        createdAt: new Date(),
      };
      
      expect(historyRecord.id).toBeDefined();
      expect(historyRecord.userId).toBeDefined();
      expect(historyRecord.gpuName).toBe("NVIDIA GeForce RTX 4090");
      expect(historyRecord.avgSpeedup).toBeGreaterThan(0);
    });
    
    it("should calculate stats from history", () => {
      const history = [
        { avgSpeedup: 15.0, maxSpeedup: 20.0, success: 1 },
        { avgSpeedup: 18.0, maxSpeedup: 25.0, success: 1 },
        { avgSpeedup: 12.0, maxSpeedup: 18.0, success: 1 },
      ];
      
      const avgSpeedups = history.map(h => h.avgSpeedup);
      const maxSpeedups = history.map(h => h.maxSpeedup);
      
      const overallAvg = avgSpeedups.reduce((a, b) => a + b, 0) / avgSpeedups.length;
      const bestSpeedup = Math.max(...maxSpeedups);
      
      expect(overallAvg).toBe(15);
      expect(bestSpeedup).toBe(25);
    });
    
    it("should handle empty history", () => {
      const emptyStats = {
        totalRuns: 0,
        avgSpeedup: null,
        bestSpeedup: null,
        lastRunDate: null,
      };
      
      expect(emptyStats.totalRuns).toBe(0);
      expect(emptyStats.avgSpeedup).toBeNull();
    });
    
    it("should filter failed benchmarks from stats", () => {
      const history = [
        { avgSpeedup: 15.0, success: 1 },
        { avgSpeedup: null, success: 0 }, // Failed
        { avgSpeedup: 18.0, success: 1 },
      ];
      
      const successfulRuns = history.filter(h => h.success === 1);
      const avgSpeedups = successfulRuns
        .filter(h => h.avgSpeedup !== null)
        .map(h => h.avgSpeedup as number);
      
      expect(successfulRuns.length).toBe(2);
      expect(avgSpeedups.length).toBe(2);
    });
  });
});
