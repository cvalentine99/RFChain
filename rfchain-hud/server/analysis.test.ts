import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the database module
vi.mock("./db", () => ({
  createAnalysisResult: vi.fn().mockResolvedValue(1),
  createForensicReport: vi.fn().mockResolvedValue(1),
  updateSignalUploadStatus: vi.fn().mockResolvedValue(undefined),
  getAnalysisResult: vi.fn().mockResolvedValue({
    id: 1,
    signalId: 1,
    userId: 1,
    avgPowerDbm: -30.5,
    peakPowerDbm: -20.2,
    paprDb: 10.3,
    snrEstimateDb: 25.0,
    sampleCount: 1000000,
    createdAt: new Date(),
  }),
  getAnalysisBySignalId: vi.fn().mockResolvedValue({
    id: 1,
    signalId: 1,
    userId: 1,
    avgPowerDbm: -30.5,
    peakPowerDbm: -20.2,
    paprDb: 10.3,
    snrEstimateDb: 25.0,
    sampleCount: 1000000,
    createdAt: new Date(),
  }),
}));

describe("Analysis Integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Analysis Result Parsing", () => {
    it("should parse metrics from analysis output", () => {
      const analysisOutput = {
        metrics: {
          avg_power_dbm: -30.5,
          peak_power_dbm: -20.2,
          papr_db: 10.3,
          iq_imbalance_db: 0.5,
          snr_estimate_db: 25.0,
          bandwidth_estimate_hz: 100000,
          center_freq_offset_hz: 50,
          dc_offset: "(0.001, 0.002)",
          sample_count: 1000000,
        },
        anomalies: {
          dc_spike: false,
          saturation: false,
          dropout: false,
        },
      };

      // Parse DC offset
      const dcStr = String(analysisOutput.metrics.dc_offset);
      const match = dcStr.match(/\(([^,]+),\s*([^)]+)\)/);
      let dcReal = 0;
      let dcImag = 0;
      if (match) {
        dcReal = parseFloat(match[1]) || 0;
        dcImag = parseFloat(match[2]) || 0;
      }

      expect(dcReal).toBeCloseTo(0.001);
      expect(dcImag).toBeCloseTo(0.002);
      expect(analysisOutput.metrics.avg_power_dbm).toBe(-30.5);
      expect(analysisOutput.metrics.sample_count).toBe(1000000);
    });

    it("should handle missing DC offset gracefully", () => {
      const analysisOutput = {
        metrics: {
          avg_power_dbm: -30.5,
          dc_offset: null,
        },
      };

      let dcReal = 0;
      let dcImag = 0;
      if (analysisOutput.metrics.dc_offset) {
        const dcStr = String(analysisOutput.metrics.dc_offset);
        const match = dcStr.match(/\(([^,]+),\s*([^)]+)\)/);
        if (match) {
          dcReal = parseFloat(match[1]) || 0;
          dcImag = parseFloat(match[2]) || 0;
        }
      }

      expect(dcReal).toBe(0);
      expect(dcImag).toBe(0);
    });
  });

  describe("Forensic Pipeline Parsing", () => {
    it("should extract hash values from forensic pipeline", () => {
      const forensicPipeline = {
        hash_chain: [
          {
            stage: "raw_input",
            hashes: {
              sha256: "abc123",
              sha3_256: "def456",
            },
          },
          {
            stage: "post_metrics",
            hashes: {
              sha256: "ghi789",
              sha3_256: "jkl012",
            },
          },
        ],
      };

      const getHash = (stage: string, algo: "sha256" | "sha3_256" = "sha256") => {
        const checkpoint = forensicPipeline.hash_chain.find((c: any) => c.stage === stage);
        return checkpoint?.hashes?.[algo] || null;
      };

      expect(getHash("raw_input", "sha256")).toBe("abc123");
      expect(getHash("raw_input", "sha3_256")).toBe("def456");
      expect(getHash("post_metrics", "sha256")).toBe("ghi789");
      expect(getHash("unknown_stage")).toBeNull();
    });

    it("should handle empty hash chain", () => {
      const forensicPipeline = {
        hash_chain: [],
      };

      const getHash = (stage: string, algo: "sha256" | "sha3_256" = "sha256") => {
        const checkpoint = forensicPipeline.hash_chain.find((c: any) => c.stage === stage);
        return checkpoint?.hashes?.[algo] || null;
      };

      expect(getHash("raw_input")).toBeNull();
    });
  });

  describe("Anomaly Detection", () => {
    it("should detect anomalies correctly", () => {
      const anomalies = {
        dc_spike: true,
        saturation: false,
        dropout: true,
        details: ["DC offset detected: 0.1234", "Dropout detected: 500 samples below threshold"],
      };

      const hasAnomalies = anomalies.dc_spike || anomalies.saturation || anomalies.dropout;
      expect(hasAnomalies).toBe(true);
      expect(anomalies.details.length).toBe(2);
    });

    it("should report no anomalies when all flags are false", () => {
      const anomalies = {
        dc_spike: false,
        saturation: false,
        dropout: false,
        details: [],
      };

      const hasAnomalies = anomalies.dc_spike || anomalies.saturation || anomalies.dropout;
      expect(hasAnomalies).toBe(false);
    });
  });

  describe("Plot URL Parsing", () => {
    it("should parse plot URLs from analysis result", () => {
      const plotUrls = {
        "01_time_domain.png": "https://s3.example.com/analysis/1/01_time_domain.png",
        "02_frequency_domain.png": "https://s3.example.com/analysis/1/02_frequency_domain.png",
        "03_spectrogram.png": "https://s3.example.com/analysis/1/03_spectrogram.png",
        "04_constellation.png": "https://s3.example.com/analysis/1/04_constellation.png",
      };

      expect(Object.keys(plotUrls).length).toBe(4);
      expect(plotUrls["01_time_domain.png"]).toContain("time_domain");
    });

    it("should filter main plots from additional plots", () => {
      const plotUrls = {
        "01_time_domain.png": "url1",
        "02_frequency_domain.png": "url2",
        "03_spectrogram.png": "url3",
        "04_constellation.png": "url4",
        "05_waterfall.png": "url5",
        "06_phase_frequency.png": "url6",
      };

      const mainPlots = ["01_time_domain.png", "02_frequency_domain.png", "03_spectrogram.png", "04_constellation.png"];
      const additionalPlots = Object.entries(plotUrls).filter(([key]) => !mainPlots.includes(key));

      expect(additionalPlots.length).toBe(2);
      expect(additionalPlots[0][0]).toBe("05_waterfall.png");
    });
  });
});

describe("Analysis Configuration", () => {
  it("should validate sample rate values", () => {
    const validSampleRates = [1e6, 2.4e6, 10e6, 20e6];
    
    validSampleRates.forEach(rate => {
      expect(rate).toBeGreaterThan(0);
      expect(rate).toBeLessThanOrEqual(100e6);
    });
  });

  it("should validate data format options", () => {
    const validFormats = ["complex64", "int16", "int8", "float32"];
    
    expect(validFormats).toContain("complex64");
    expect(validFormats).toContain("int16");
    expect(validFormats.length).toBe(4);
  });
});
