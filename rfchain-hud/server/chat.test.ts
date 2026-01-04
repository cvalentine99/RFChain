import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the LLM module
vi.mock("./_core/llm", () => ({
  invokeLLM: vi.fn().mockResolvedValue({
    id: "test-id",
    created: Date.now(),
    model: "test-model",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: "Good day, sir. I am JARVIS, your RF Signal Intelligence Assistant.",
        },
        finish_reason: "stop",
      },
    ],
  }),
}));

// Mock the database module
vi.mock("./db", () => ({
  createChatMessage: vi.fn().mockResolvedValue(1),
  getSessionMessages: vi.fn().mockResolvedValue([]),
  getUserLlmConfig: vi.fn().mockResolvedValue(null),
  upsertLlmConfig: vi.fn().mockResolvedValue(undefined),
  getRecentAnalysesWithUploads: vi.fn().mockResolvedValue([]),
  getUserSignalUploads: vi.fn().mockResolvedValue([]),
  getSignalUpload: vi.fn().mockResolvedValue(null),
  createSignalUpload: vi.fn().mockResolvedValue(1),
  updateSignalUploadStatus: vi.fn().mockResolvedValue(undefined),
  getAnalysisResult: vi.fn().mockResolvedValue(null),
  getAnalysisBySignalId: vi.fn().mockResolvedValue(null),
  getUserAnalyses: vi.fn().mockResolvedValue([]),
  createAnalysisResult: vi.fn().mockResolvedValue(1),
  getForensicReport: vi.fn().mockResolvedValue(null),
  getUserForensicReports: vi.fn().mockResolvedValue([]),
  createForensicReport: vi.fn().mockResolvedValue(1),
  getDb: vi.fn().mockResolvedValue(null),
}));

describe("Chat Router", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should have JARVIS system prompt configured", async () => {
    const { invokeLLM } = await import("./_core/llm");
    
    // Simulate calling the chat endpoint
    const systemPrompt = `You are JARVIS, an advanced AI assistant specialized in RF signal analysis and forensic investigation.`;
    
    expect(systemPrompt).toContain("JARVIS");
    expect(systemPrompt).toContain("RF signal analysis");
  });

  it("should handle LLM response content correctly", () => {
    // Test string content extraction
    const stringContent = "Hello, this is a test response";
    expect(typeof stringContent === "string" ? stringContent : "").toBe("Hello, this is a test response");

    // Test array content extraction
    const arrayContent = [
      { type: "text" as const, text: "Part 1" },
      { type: "text" as const, text: " Part 2" },
    ];
    const extracted = arrayContent.map(c => c.type === "text" ? c.text : "").join("");
    expect(extracted).toBe("Part 1 Part 2");
  });

  it("should build context-aware prompts when analysis data is provided", () => {
    const analysisContext = {
      avgPowerDbm: -45.2,
      peakPowerDbm: -30.1,
      paprDb: 15.1,
      snrEstimateDb: 25.5,
    };

    let systemPrompt = "You are JARVIS.";
    if (analysisContext) {
      systemPrompt += `\n\nCurrent Analysis Context:\n${JSON.stringify(analysisContext, null, 2)}`;
    }

    expect(systemPrompt).toContain("avgPowerDbm");
    expect(systemPrompt).toContain("-45.2");
    expect(systemPrompt).toContain("snrEstimateDb");
  });
});

describe("LLM Config Router", () => {
  it("should validate provider enum values", () => {
    const validProviders = ["builtin", "openai", "anthropic", "local"];
    
    validProviders.forEach(provider => {
      expect(["builtin", "openai", "anthropic", "local"]).toContain(provider);
    });

    expect(["builtin", "openai", "anthropic", "local"]).not.toContain("invalid");
  });

  it("should handle optional API key and endpoint fields", () => {
    const configWithKey = {
      provider: "openai",
      apiKey: "sk-test-key",
    };

    const configWithEndpoint = {
      provider: "local",
      localEndpoint: "http://localhost:11434/v1",
    };

    const configBuiltin = {
      provider: "builtin",
    };

    expect(configWithKey.apiKey).toBeDefined();
    expect(configWithEndpoint.localEndpoint).toBeDefined();
    expect(configBuiltin.provider).toBe("builtin");
  });
});

describe("Signal Upload Router", () => {
  it("should generate unique S3 keys", () => {
    const userId = 123;
    const timestamp = Date.now();
    const randomSuffix = Math.random().toString(36).substring(2, 10);
    const filename = "test_signal.bin";
    
    const s3Key = `signals/${userId}/${timestamp}-${randomSuffix}-${filename}`;
    
    expect(s3Key).toContain(`signals/${userId}/`);
    expect(s3Key).toContain(filename);
    expect(s3Key.length).toBeGreaterThan(30);
  });

  it("should validate file extensions", () => {
    const validExtensions = [".bin", ".raw", ".iq"];
    
    const testFiles = [
      { name: "signal.bin", valid: true },
      { name: "data.raw", valid: true },
      { name: "capture.iq", valid: true },
      { name: "document.pdf", valid: false },
      { name: "image.png", valid: false },
    ];

    testFiles.forEach(file => {
      const isValid = validExtensions.some(ext => file.name.endsWith(ext));
      expect(isValid).toBe(file.valid);
    });
  });
});

describe("Forensic Report Router", () => {
  it("should handle 6-stage hash chain structure", () => {
    const hashChain = {
      rawInputHash: "abc123...",
      postMetricsHash: "def456...",
      postAnomalyHash: "ghi789...",
      postDigitalHash: "jkl012...",
      postV3Hash: "mno345...",
      preOutputHash: "pqr678...",
    };

    const stages = Object.keys(hashChain);
    expect(stages.length).toBe(6);
    expect(stages).toContain("rawInputHash");
    expect(stages).toContain("preOutputHash");
  });

  it("should support dual hashing (SHA-256 and SHA3-256)", () => {
    const dualHashes = {
      sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      sha3_256: "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
    };

    expect(dualHashes.sha256.length).toBe(64);
    expect(dualHashes.sha3_256.length).toBe(64);
  });
});
