import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, json, bigint, float } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 */
export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).unique(), // Optional for local auth
  username: varchar("username", { length: 64 }).unique(), // For local auth
  passwordHash: varchar("passwordHash", { length: 255 }), // bcrypt hash for local auth
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }), // 'local' or 'oauth'
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

/**
 * Sessions table for local auth
 */
export const sessions = mysqlTable("sessions", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  token: varchar("token", { length: 128 }).notNull().unique(),
  expiresAt: timestamp("expiresAt").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Session = typeof sessions.$inferSelect;
export type InsertSession = typeof sessions.$inferInsert;

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Signal uploads - stores metadata about uploaded signal files
 */
export const signalUploads = mysqlTable("signal_uploads", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  filename: varchar("filename", { length: 255 }).notNull(),
  originalName: varchar("originalName", { length: 255 }).notNull(),
  fileSize: bigint("fileSize", { mode: "number" }).notNull(),
  mimeType: varchar("mimeType", { length: 100 }),
  s3Key: varchar("s3Key", { length: 512 }).notNull(),
  s3Url: varchar("s3Url", { length: 1024 }).notNull(),
  status: mysqlEnum("status", ["pending", "processing", "completed", "failed"]).default("pending").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type SignalUpload = typeof signalUploads.$inferSelect;
export type InsertSignalUpload = typeof signalUploads.$inferInsert;

/**
 * Analysis results - stores computed metrics from RF analysis
 */
export const analysisResults = mysqlTable("analysis_results", {
  id: int("id").autoincrement().primaryKey(),
  signalId: int("signalId").notNull(),
  userId: int("userId").notNull(),
  
  // Core metrics
  sampleCount: bigint("sampleCount", { mode: "number" }),
  durationMs: float("durationMs"),
  avgPowerDbm: float("avgPowerDbm"),
  peakPowerDbm: float("peakPowerDbm"),
  paprDb: float("paprDb"),
  iqImbalanceDb: float("iqImbalanceDb"),
  snrEstimateDb: float("snrEstimateDb"),
  bandwidthHz: float("bandwidthHz"),
  freqOffsetHz: float("freqOffsetHz"),
  dcOffsetReal: float("dcOffsetReal"),
  dcOffsetImag: float("dcOffsetImag"),
  
  // Anomaly detection
  anomalies: json("anomalies"),
  
  // Full metrics JSON
  fullMetrics: json("fullMetrics"),
  
  // Visualization URLs (S3)
  timeDomainUrl: varchar("timeDomainUrl", { length: 1024 }),
  frequencyDomainUrl: varchar("frequencyDomainUrl", { length: 1024 }),
  spectrogramUrl: varchar("spectrogramUrl", { length: 1024 }),
  waterfallUrl: varchar("waterfallUrl", { length: 1024 }),
  constellationUrl: varchar("constellationUrl", { length: 1024 }),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type AnalysisResult = typeof analysisResults.$inferSelect;
export type InsertAnalysisResult = typeof analysisResults.$inferInsert;

/**
 * Forensic reports - stores chain of custody and hash verification data
 */
export const forensicReports = mysqlTable("forensic_reports", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull(),
  userId: int("userId").notNull(),
  
  // Hash checkpoints (6 stages)
  rawInputHash: varchar("rawInputHash", { length: 128 }),
  postMetricsHash: varchar("postMetricsHash", { length: 128 }),
  postAnomalyHash: varchar("postAnomalyHash", { length: 128 }),
  postDigitalHash: varchar("postDigitalHash", { length: 128 }),
  postV3Hash: varchar("postV3Hash", { length: 128 }),
  preOutputHash: varchar("preOutputHash", { length: 128 }),
  
  // SHA3-256 hashes
  rawInputHashSha3: varchar("rawInputHashSha3", { length: 128 }),
  postMetricsHashSha3: varchar("postMetricsHashSha3", { length: 128 }),
  postAnomalyHashSha3: varchar("postAnomalyHashSha3", { length: 128 }),
  postDigitalHashSha3: varchar("postDigitalHashSha3", { length: 128 }),
  postV3HashSha3: varchar("postV3HashSha3", { length: 128 }),
  preOutputHashSha3: varchar("preOutputHashSha3", { length: 128 }),
  
  // Full forensic pipeline data
  forensicPipeline: json("forensicPipeline"),
  chainOfCustody: json("chainOfCustody"),
  
  // Output file hash
  outputSha256: varchar("outputSha256", { length: 128 }),
  outputSha3: varchar("outputSha3", { length: 128 }),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ForensicReport = typeof forensicReports.$inferSelect;
export type InsertForensicReport = typeof forensicReports.$inferInsert;

/**
 * Chat messages - stores RAG assistant conversation history
 */
export const chatMessages = mysqlTable("chat_messages", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  sessionId: varchar("sessionId", { length: 64 }).notNull(),
  role: mysqlEnum("role", ["user", "assistant", "system"]).notNull(),
  content: text("content").notNull(),
  
  // Context for RAG
  analysisContext: json("analysisContext"),
  
  // Voice-related
  wasVoiceInput: int("wasVoiceInput").default(0),
  audioUrl: varchar("audioUrl", { length: 1024 }),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ChatMessage = typeof chatMessages.$inferSelect;
export type InsertChatMessage = typeof chatMessages.$inferInsert;

/**
 * LLM configurations - stores user's LLM provider settings
 */
export const llmConfigs = mysqlTable("llm_configs", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().unique(),
  provider: mysqlEnum("provider", ["builtin", "openai", "anthropic", "local"]).default("builtin").notNull(),
  apiKey: varchar("apiKey", { length: 512 }),
  localEndpoint: varchar("localEndpoint", { length: 512 }),
  model: varchar("model", { length: 128 }),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type LlmConfig = typeof llmConfigs.$inferSelect;
export type InsertLlmConfig = typeof llmConfigs.$inferInsert;

/**
 * Analysis embeddings - stores vector embeddings for RAG semantic search
 */
export const analysisEmbeddings = mysqlTable("analysis_embeddings", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull().unique(),
  userId: int("userId").notNull(),
  
  // Text content that was embedded
  contentText: text("contentText").notNull(),
  
  // Vector embedding (stored as JSON array of floats)
  // Using 1536 dimensions for OpenAI text-embedding-3-small compatibility
  embedding: json("embedding").notNull(),
  
  // Metadata for filtering
  signalFilename: varchar("signalFilename", { length: 255 }),
  analysisDate: timestamp("analysisDate"),
  
  // Summary metrics for quick reference
  avgPowerDbm: float("avgPowerDbm"),
  bandwidthHz: float("bandwidthHz"),
  hasAnomalies: int("hasAnomalies").default(0),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type AnalysisEmbedding = typeof analysisEmbeddings.$inferSelect;
export type InsertAnalysisEmbedding = typeof analysisEmbeddings.$inferInsert;

/**
 * RAG settings - stores user preferences for RAG-enhanced chat
 */
export const ragSettings = mysqlTable("rag_settings", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().unique(),
  
  // RAG configuration
  enabled: int("enabled").default(1).notNull(), // 1 = enabled, 0 = disabled
  similarityThreshold: float("similarityThreshold").default(0.7).notNull(), // 0.0 to 1.0
  maxResults: int("maxResults").default(5).notNull(), // Max relevant analyses to include
  
  // Auto-embed settings
  autoEmbed: int("autoEmbed").default(1).notNull(), // Auto-generate embeddings for new analyses
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type RagSettings = typeof ragSettings.$inferSelect;
export type InsertRagSettings = typeof ragSettings.$inferInsert;


/**
 * Batch jobs - stores batch processing job metadata
 */
export const batchJobs = mysqlTable("batch_jobs", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  
  // Job metadata
  name: varchar("name", { length: 255 }),
  totalFiles: int("totalFiles").notNull(),
  completedFiles: int("completedFiles").default(0).notNull(),
  failedFiles: int("failedFiles").default(0).notNull(),
  
  // Status
  status: mysqlEnum("status", ["pending", "processing", "completed", "failed", "cancelled"]).default("pending").notNull(),
  
  // Processing options
  sampleRate: float("sampleRate").default(1000000),
  dataFormat: varchar("dataFormat", { length: 32 }).default("complex64"),
  digitalAnalysis: int("digitalAnalysis").default(0),
  v3Analysis: int("v3Analysis").default(0),
  
  // Timing
  startedAt: timestamp("startedAt"),
  completedAt: timestamp("completedAt"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type BatchJob = typeof batchJobs.$inferSelect;
export type InsertBatchJob = typeof batchJobs.$inferInsert;

/**
 * Batch queue items - individual files in a batch job queue
 */
export const batchQueueItems = mysqlTable("batch_queue_items", {
  id: int("id").autoincrement().primaryKey(),
  batchJobId: int("batchJobId").notNull(),
  signalUploadId: int("signalUploadId"),
  
  // File info
  filename: varchar("filename", { length: 255 }).notNull(),
  localPath: varchar("localPath", { length: 1024 }),
  
  // Processing status
  status: mysqlEnum("status", ["queued", "processing", "completed", "failed"]).default("queued").notNull(),
  position: int("position").notNull(), // Queue position
  
  // Result reference
  analysisResultId: int("analysisResultId"),
  
  // Error tracking
  errorMessage: text("errorMessage"),
  
  // Timing
  startedAt: timestamp("startedAt"),
  completedAt: timestamp("completedAt"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type BatchQueueItem = typeof batchQueueItems.$inferSelect;
export type InsertBatchQueueItem = typeof batchQueueItems.$inferInsert;


/**
 * GPU benchmark history - stores benchmark results for performance tracking
 */
export const gpuBenchmarkHistory = mysqlTable("gpu_benchmark_history", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  
  // GPU info
  gpuName: varchar("gpuName", { length: 255 }),
  gpuMemoryMb: int("gpuMemoryMb"),
  cudaVersion: varchar("cudaVersion", { length: 32 }),
  driverVersion: varchar("driverVersion", { length: 32 }),
  
  // Summary metrics
  avgSpeedup: float("avgSpeedup"),
  maxSpeedup: float("maxSpeedup"),
  minSpeedup: float("minSpeedup"),
  
  // Individual benchmark results (JSON array)
  benchmarkResults: json("benchmarkResults"),
  
  // System info at time of benchmark
  systemInfo: json("systemInfo"),
  
  // Status
  success: int("success").default(1).notNull(),
  errorMessage: text("errorMessage"),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type GpuBenchmarkHistory = typeof gpuBenchmarkHistory.$inferSelect;
export type InsertGpuBenchmarkHistory = typeof gpuBenchmarkHistory.$inferInsert;


/**
 * Signal signatures - known signal profiles for automatic classification
 */
export const signalSignatures = mysqlTable("signal_signatures", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId"), // null = system/built-in signature
  
  // Signature identification
  name: varchar("name", { length: 255 }).notNull(),
  category: varchar("category", { length: 64 }).notNull(), // WiFi, LTE, Bluetooth, Amateur, etc.
  subcategory: varchar("subcategory", { length: 64 }), // 802.11n, Band 7, etc.
  description: text("description"),
  
  // Spectral characteristics
  bandwidthMinHz: float("bandwidthMinHz"),
  bandwidthMaxHz: float("bandwidthMaxHz"),
  centerFreqHz: float("centerFreqHz"), // If known
  
  // Modulation characteristics
  modulationType: varchar("modulationType", { length: 64 }), // OFDM, QAM, FSK, etc.
  symbolRateMin: float("symbolRateMin"),
  symbolRateMax: float("symbolRateMax"),
  
  // Power characteristics
  typicalPaprDb: float("typicalPaprDb"),
  paprToleranceDb: float("paprToleranceDb"),
  
  // Spectral fingerprint (normalized PSD shape)
  spectralFingerprint: json("spectralFingerprint"), // Array of normalized magnitude values
  fingerprintResolution: int("fingerprintResolution"), // Number of bins in fingerprint
  
  // Cyclostationary features
  cyclicFrequencies: json("cyclicFrequencies"), // Array of expected cyclic frequencies
  
  // OFDM-specific
  ofdmFftSize: int("ofdmFftSize"),
  ofdmCyclicPrefix: float("ofdmCyclicPrefix"),
  ofdmSubcarrierSpacing: float("ofdmSubcarrierSpacing"),
  
  // Matching parameters
  matchThreshold: float("matchThreshold").default(0.7), // Minimum similarity for match
  priority: int("priority").default(0), // Higher = checked first
  
  // Metadata
  isBuiltIn: int("isBuiltIn").default(0).notNull(), // 1 = system signature, 0 = user-defined
  enabled: int("enabled").default(1).notNull(),
  
  // Reference info
  referenceUrl: varchar("referenceUrl", { length: 1024 }),
  referenceNotes: text("referenceNotes"),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type SignalSignature = typeof signalSignatures.$inferSelect;
export type InsertSignalSignature = typeof signalSignatures.$inferInsert;

/**
 * Signature matches - records of signals matched to signatures
 */
export const signatureMatches = mysqlTable("signature_matches", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull(),
  signatureId: int("signatureId").notNull(),
  
  // Match quality
  matchScore: float("matchScore").notNull(), // 0.0 to 1.0
  confidence: varchar("confidence", { length: 16 }).notNull(), // high, medium, low
  
  // Match details
  matchDetails: json("matchDetails"), // Which features matched and how well
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type SignatureMatch = typeof signatureMatches.$inferSelect;
export type InsertSignatureMatch = typeof signatureMatches.$inferInsert;
