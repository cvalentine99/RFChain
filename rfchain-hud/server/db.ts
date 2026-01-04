import { eq, desc, and, sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { 
  InsertUser, users, User,
  signalUploads, InsertSignalUpload, SignalUpload,
  analysisResults, InsertAnalysisResult, AnalysisResult,
  forensicReports, InsertForensicReport, ForensicReport,
  chatMessages, InsertChatMessage,
  llmConfigs, InsertLlmConfig, LlmConfig,
  analysisEmbeddings, InsertAnalysisEmbedding, AnalysisEmbedding,
  ragSettings, InsertRagSettings, RagSettings,
  batchJobs, InsertBatchJob, BatchJob,
  batchQueueItems, InsertBatchQueueItem, BatchQueueItem,
  gpuBenchmarkHistory, InsertGpuBenchmarkHistory, GpuBenchmarkHistory,
  sessions, InsertSession
} from "../drizzle/schema";
import * as bcrypt from "bcrypt";
import * as crypto from "crypto";
import { cosineSimilarity } from "./_core/embeddings";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// Health check for database connectivity
export async function healthCheck(): Promise<{ connected: boolean; version?: string }> {
  try {
    const db = await getDb();
    if (!db) {
      return { connected: false };
    }
    // Simple query to verify connection
    const result = await db.execute(sql`SELECT VERSION() as version`);
    // Safely extract version from result
    let version = 'unknown';
    if (Array.isArray(result) && result.length > 0) {
      const firstRow = Array.isArray(result[0]) ? result[0][0] : result[0];
      if (firstRow && typeof firstRow === 'object' && 'version' in firstRow) {
        version = String(firstRow.version);
      }
    }
    return { connected: true, version };
  } catch (error) {
    console.error('[Database] Health check failed:', error);
    return { connected: false };
  }
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

// ============= Signal Upload Queries =============

export async function createSignalUpload(upload: InsertSignalUpload): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(signalUploads).values(upload);
  return result[0].insertId;
}

export async function getSignalUpload(id: number): Promise<SignalUpload | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(signalUploads).where(eq(signalUploads.id, id)).limit(1);
  return result[0];
}

export async function getUserSignalUploads(userId: number, limit = 20): Promise<SignalUpload[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(signalUploads)
    .where(eq(signalUploads.userId, userId))
    .orderBy(desc(signalUploads.createdAt))
    .limit(limit);
}

export async function updateSignalUploadStatus(id: number, status: SignalUpload['status']): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.update(signalUploads).set({ status }).where(eq(signalUploads.id, id));
}

// ============= Analysis Results Queries =============

export async function createAnalysisResult(result: InsertAnalysisResult): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const insertResult = await db.insert(analysisResults).values(result);
  return insertResult[0].insertId;
}

export async function getAnalysisResult(id: number): Promise<AnalysisResult | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(analysisResults).where(eq(analysisResults.id, id)).limit(1);
  return result[0];
}

export async function getAnalysisBySignalId(signalId: number): Promise<AnalysisResult | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(analysisResults).where(eq(analysisResults.signalId, signalId)).limit(1);
  return result[0];
}

export async function getUserAnalyses(userId: number, limit = 20): Promise<AnalysisResult[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(analysisResults)
    .where(eq(analysisResults.userId, userId))
    .orderBy(desc(analysisResults.createdAt))
    .limit(limit);
}

export async function getRecentAnalysesWithUploads(userId: number, limit = 10) {
  const db = await getDb();
  if (!db) return [];
  
  // Get analyses with their signal uploads
  const results = await db.select({
    id: analysisResults.id,
    signalId: analysisResults.signalId,
    avgPowerDbm: analysisResults.avgPowerDbm,
    peakPowerDbm: analysisResults.peakPowerDbm,
    paprDb: analysisResults.paprDb,
    snrEstimateDb: analysisResults.snrEstimateDb,
    sampleCount: analysisResults.sampleCount,
    bandwidthHz: analysisResults.bandwidthHz,
    freqOffsetHz: analysisResults.freqOffsetHz,
    iqImbalanceDb: analysisResults.iqImbalanceDb,
    dcOffsetReal: analysisResults.dcOffsetReal,
    dcOffsetImag: analysisResults.dcOffsetImag,
    anomalies: analysisResults.anomalies,
    fullMetrics: analysisResults.fullMetrics,
    createdAt: analysisResults.createdAt,
  }).from(analysisResults)
    .where(eq(analysisResults.userId, userId))
    .orderBy(desc(analysisResults.createdAt))
    .limit(limit);
  
  return results;
}

// ============= Forensic Reports Queries =============

export async function createForensicReport(report: InsertForensicReport): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(forensicReports).values(report);
  return result[0].insertId;
}

export async function getForensicReport(analysisId: number): Promise<ForensicReport | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(forensicReports).where(eq(forensicReports.analysisId, analysisId)).limit(1);
  return result[0];
}

export async function getUserForensicReports(userId: number, limit = 20): Promise<ForensicReport[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(forensicReports)
    .where(eq(forensicReports.userId, userId))
    .orderBy(desc(forensicReports.createdAt))
    .limit(limit);
}

// ============= Chat Messages Queries =============

export async function createChatMessage(message: InsertChatMessage): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(chatMessages).values(message);
  return result[0].insertId;
}

export async function getSessionMessages(sessionId: string, limit = 50) {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(chatMessages)
    .where(eq(chatMessages.sessionId, sessionId))
    .orderBy(desc(chatMessages.createdAt))
    .limit(limit);
}

// ============= LLM Config Queries =============

export async function getUserLlmConfig(userId: number): Promise<LlmConfig | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(llmConfigs).where(eq(llmConfigs.userId, userId)).limit(1);
  return result[0];
}

export async function upsertLlmConfig(config: InsertLlmConfig): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.insert(llmConfigs).values(config).onDuplicateKeyUpdate({
    set: {
      provider: config.provider,
      apiKey: config.apiKey,
      localEndpoint: config.localEndpoint,
      model: config.model,
    },
  });
}

// ============= Analysis Embeddings Queries =============

export async function createAnalysisEmbedding(embedding: InsertAnalysisEmbedding): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(analysisEmbeddings).values(embedding);
  return result[0].insertId;
}

export async function getAnalysisEmbedding(analysisId: number): Promise<AnalysisEmbedding | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(analysisEmbeddings)
    .where(eq(analysisEmbeddings.analysisId, analysisId))
    .limit(1);
  return result[0];
}

export async function getAllUserEmbeddings(userId: number): Promise<AnalysisEmbedding[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(analysisEmbeddings)
    .where(eq(analysisEmbeddings.userId, userId))
    .orderBy(desc(analysisEmbeddings.createdAt));
}

export async function getAnalysesWithoutEmbeddings(userId: number, limit = 50): Promise<AnalysisResult[]> {
  const db = await getDb();
  if (!db) return [];
  
  // Get analyses that don't have embeddings yet
  const analyses = await db.select().from(analysisResults)
    .where(eq(analysisResults.userId, userId))
    .orderBy(desc(analysisResults.createdAt))
    .limit(limit);
  
  // Filter out those that already have embeddings
  const embeddedIds = await db.select({ analysisId: analysisEmbeddings.analysisId })
    .from(analysisEmbeddings)
    .where(eq(analysisEmbeddings.userId, userId));
  
  const embeddedIdSet = new Set(embeddedIds.map(e => e.analysisId));
  return analyses.filter(a => !embeddedIdSet.has(a.id));
}

export type SimilarAnalysis = {
  analysisId: number;
  similarity: number;
  contentText: string;
  signalFilename: string | null;
  avgPowerDbm: number | null;
  bandwidthHz: number | null;
  hasAnomalies: boolean;
};

/**
 * Find similar analyses using cosine similarity
 * This performs in-memory similarity calculation since MySQL doesn't have native vector support
 */
export async function findSimilarAnalyses(
  userId: number,
  queryEmbedding: number[],
  limit = 5,
  minSimilarity = 0.3
): Promise<SimilarAnalysis[]> {
  const db = await getDb();
  if (!db) return [];
  
  // Get all embeddings for the user
  const allEmbeddings = await getAllUserEmbeddings(userId);
  
  // Calculate similarity scores
  const scored = allEmbeddings.map(emb => {
    const embedding = emb.embedding as number[];
    const similarity = cosineSimilarity(queryEmbedding, embedding);
    return {
      analysisId: emb.analysisId,
      similarity,
      contentText: emb.contentText,
      signalFilename: emb.signalFilename,
      avgPowerDbm: emb.avgPowerDbm,
      bandwidthHz: emb.bandwidthHz,
      hasAnomalies: emb.hasAnomalies === 1,
    };
  });
  
  // Sort by similarity and filter
  return scored
    .filter(s => s.similarity >= minSimilarity)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}

/**
 * Update or insert an embedding for an analysis
 */
export async function upsertAnalysisEmbedding(embedding: InsertAnalysisEmbedding): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.insert(analysisEmbeddings).values(embedding).onDuplicateKeyUpdate({
    set: {
      contentText: embedding.contentText,
      embedding: embedding.embedding,
      signalFilename: embedding.signalFilename,
      avgPowerDbm: embedding.avgPowerDbm,
      bandwidthHz: embedding.bandwidthHz,
      hasAnomalies: embedding.hasAnomalies,
    },
  });
}


// ============ RAG Settings ============

export async function getRagSettings(userId: number): Promise<RagSettings | null> {
  const db = await getDb();
  if (!db) return null;
  
  const results = await db.select().from(ragSettings)
    .where(eq(ragSettings.userId, userId))
    .limit(1);
  
  return results[0] || null;
}

export async function upsertRagSettings(userId: number, settings: Partial<InsertRagSettings>): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  const existing = await getRagSettings(userId);
  
  if (existing) {
    await db.update(ragSettings)
      .set({
        enabled: settings.enabled ?? existing.enabled,
        similarityThreshold: settings.similarityThreshold ?? existing.similarityThreshold,
        maxResults: settings.maxResults ?? existing.maxResults,
        autoEmbed: settings.autoEmbed ?? existing.autoEmbed,
      })
      .where(eq(ragSettings.userId, userId));
  } else {
    await db.insert(ragSettings).values({
      userId,
      enabled: settings.enabled ?? 1,
      similarityThreshold: settings.similarityThreshold ?? 0.7,
      maxResults: settings.maxResults ?? 5,
      autoEmbed: settings.autoEmbed ?? 1,
    });
  }
}

export async function getEmbeddingStats(userId: number): Promise<{ total: number; embedded: number }> {
  const db = await getDb();
  if (!db) return { total: 0, embedded: 0 };
  
  // Count total analyses
  const totalResult = await db.select({ count: sql<number>`count(*)` })
    .from(analysisResults)
    .where(eq(analysisResults.userId, userId));
  
  // Count embedded analyses
  const embeddedResult = await db.select({ count: sql<number>`count(*)` })
    .from(analysisEmbeddings)
    .where(eq(analysisEmbeddings.userId, userId));
  
  return {
    total: totalResult[0]?.count || 0,
    embedded: embeddedResult[0]?.count || 0,
  };
}


// ============ Batch Processing ============

export async function createBatchJob(job: InsertBatchJob): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(batchJobs).values(job);
  return result[0].insertId;
}

export async function getBatchJob(id: number): Promise<BatchJob | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(batchJobs).where(eq(batchJobs.id, id)).limit(1);
  return result[0];
}

export async function getUserBatchJobs(userId: number, limit = 20): Promise<BatchJob[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(batchJobs)
    .where(eq(batchJobs.userId, userId))
    .orderBy(desc(batchJobs.createdAt))
    .limit(limit);
}

export async function updateBatchJobStatus(
  id: number, 
  status: BatchJob['status'],
  updates?: { completedFiles?: number; failedFiles?: number; startedAt?: Date; completedAt?: Date }
): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.update(batchJobs).set({ 
    status,
    ...updates
  }).where(eq(batchJobs.id, id));
}

export async function incrementBatchJobProgress(id: number, success: boolean): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  if (success) {
    await db.update(batchJobs)
      .set({ completedFiles: sql`${batchJobs.completedFiles} + 1` })
      .where(eq(batchJobs.id, id));
  } else {
    await db.update(batchJobs)
      .set({ failedFiles: sql`${batchJobs.failedFiles} + 1` })
      .where(eq(batchJobs.id, id));
  }
}

// Queue Items
export async function createBatchQueueItem(item: InsertBatchQueueItem): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(batchQueueItems).values(item);
  return result[0].insertId;
}

export async function getBatchQueueItems(batchJobId: number): Promise<BatchQueueItem[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(batchQueueItems)
    .where(eq(batchQueueItems.batchJobId, batchJobId))
    .orderBy(batchQueueItems.position);
}

export async function getNextQueueItem(batchJobId: number): Promise<BatchQueueItem | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(batchQueueItems)
    .where(and(
      eq(batchQueueItems.batchJobId, batchJobId),
      eq(batchQueueItems.status, 'queued')
    ))
    .orderBy(batchQueueItems.position)
    .limit(1);
  
  return result[0];
}

export async function updateQueueItemStatus(
  id: number,
  status: BatchQueueItem['status'],
  updates?: { signalUploadId?: number; analysisResultId?: number; errorMessage?: string; startedAt?: Date; completedAt?: Date }
): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.update(batchQueueItems).set({
    status,
    ...updates
  }).where(eq(batchQueueItems.id, id));
}

export async function getActiveBatchJob(userId: number): Promise<BatchJob | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(batchJobs)
    .where(and(
      eq(batchJobs.userId, userId),
      eq(batchJobs.status, 'processing')
    ))
    .limit(1);
  
  return result[0];
}


// ============================================================================
// GPU Benchmark History
// ============================================================================

export async function saveBenchmarkResult(data: InsertGpuBenchmarkHistory): Promise<number | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.insert(gpuBenchmarkHistory).values(data);
  return result[0].insertId;
}

export async function getBenchmarkHistory(userId: number, limit: number = 20): Promise<GpuBenchmarkHistory[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(gpuBenchmarkHistory)
    .where(eq(gpuBenchmarkHistory.userId, userId))
    .orderBy(desc(gpuBenchmarkHistory.createdAt))
    .limit(limit);
}

export async function getBenchmarkById(id: number): Promise<GpuBenchmarkHistory | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(gpuBenchmarkHistory)
    .where(eq(gpuBenchmarkHistory.id, id))
    .limit(1);
  
  return result[0];
}

export async function getLatestBenchmark(userId: number): Promise<GpuBenchmarkHistory | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(gpuBenchmarkHistory)
    .where(and(
      eq(gpuBenchmarkHistory.userId, userId),
      eq(gpuBenchmarkHistory.success, 1)
    ))
    .orderBy(desc(gpuBenchmarkHistory.createdAt))
    .limit(1);
  
  return result[0];
}

export async function deleteBenchmark(id: number, userId: number): Promise<boolean> {
  const db = await getDb();
  if (!db) return false;
  
  await db.delete(gpuBenchmarkHistory)
    .where(and(
      eq(gpuBenchmarkHistory.id, id),
      eq(gpuBenchmarkHistory.userId, userId)
    ));
  
  return true;
}

export async function getBenchmarkStats(userId: number): Promise<{
  totalRuns: number;
  avgSpeedup: number | null;
  bestSpeedup: number | null;
  lastRunDate: Date | null;
}> {
  const db = await getDb();
  if (!db) return { totalRuns: 0, avgSpeedup: null, bestSpeedup: null, lastRunDate: null };
  
  const history = await db.select().from(gpuBenchmarkHistory)
    .where(and(
      eq(gpuBenchmarkHistory.userId, userId),
      eq(gpuBenchmarkHistory.success, 1)
    ))
    .orderBy(desc(gpuBenchmarkHistory.createdAt));
  
  if (history.length === 0) {
    return { totalRuns: 0, avgSpeedup: null, bestSpeedup: null, lastRunDate: null };
  }
  
  const avgSpeedups = history.filter(h => h.avgSpeedup !== null).map(h => h.avgSpeedup as number);
  const maxSpeedups = history.filter(h => h.maxSpeedup !== null).map(h => h.maxSpeedup as number);
  
  return {
    totalRuns: history.length,
    avgSpeedup: avgSpeedups.length > 0 ? avgSpeedups.reduce((a, b) => a + b, 0) / avgSpeedups.length : null,
    bestSpeedup: maxSpeedups.length > 0 ? Math.max(...maxSpeedups) : null,
    lastRunDate: history[0].createdAt,
  };
}


// ============================================================================
// Local Authentication
// ============================================================================

const SALT_ROUNDS = 12;
const SESSION_DURATION_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

export async function createLocalUser(
  username: string, 
  password: string, 
  name?: string, 
  email?: string,
  role: 'user' | 'admin' = 'user'
): Promise<User | null> {
  const db = await getDb();
  if (!db) return null;
  
  // Check if username already exists
  const existing = await db.select().from(users)
    .where(eq(users.username, username))
    .limit(1);
  
  if (existing.length > 0) {
    throw new Error('Username already exists');
  }
  
  const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
  
  const result = await db.insert(users).values({
    username,
    passwordHash,
    name: name || username,
    email,
    loginMethod: 'local',
    role,
    lastSignedIn: new Date(),
  });
  
  // Fetch and return the created user
  const newUser = await db.select().from(users)
    .where(eq(users.id, result[0].insertId))
    .limit(1);
  
  return newUser[0] || null;
}

export async function verifyLocalUser(username: string, password: string): Promise<User | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.select().from(users)
    .where(eq(users.username, username))
    .limit(1);
  
  if (result.length === 0) return null;
  
  const user = result[0];
  if (!user.passwordHash) return null;
  
  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) return null;
  
  // Update last signed in
  await db.update(users)
    .set({ lastSignedIn: new Date() })
    .where(eq(users.id, user.id));
  
  return user;
}

export async function getUserByUsername(username: string): Promise<User | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.select().from(users)
    .where(eq(users.username, username))
    .limit(1);
  
  return result[0] || null;
}

export async function createSession(userId: number): Promise<string | null> {
  const db = await getDb();
  if (!db) return null;
  
  const token = crypto.randomBytes(64).toString('hex');
  const expiresAt = new Date(Date.now() + SESSION_DURATION_MS);
  
  await db.insert(sessions).values({
    userId,
    token,
    expiresAt,
  });
  
  return token;
}

export async function validateSession(token: string): Promise<User | null> {
  const db = await getDb();
  if (!db) return null;
  
  const result = await db.select().from(sessions)
    .where(eq(sessions.token, token))
    .limit(1);
  
  if (result.length === 0) return null;
  
  const session = result[0];
  
  // Check if session expired
  if (new Date() > session.expiresAt) {
    await db.delete(sessions).where(eq(sessions.id, session.id));
    return null;
  }
  
  // Get user
  const userResult = await db.select().from(users)
    .where(eq(users.id, session.userId))
    .limit(1);
  
  return userResult[0] || null;
}

export async function deleteSession(token: string): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.delete(sessions).where(eq(sessions.token, token));
}

export async function deleteUserSessions(userId: number): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.delete(sessions).where(eq(sessions.userId, userId));
}

export async function changePassword(userId: number, newPassword: string): Promise<boolean> {
  const db = await getDb();
  if (!db) return false;
  
  const passwordHash = await bcrypt.hash(newPassword, SALT_ROUNDS);
  
  await db.update(users)
    .set({ passwordHash })
    .where(eq(users.id, userId));
  
  // Invalidate all sessions
  await deleteUserSessions(userId);
  
  return true;
}

export async function getUserCount(): Promise<number> {
  const db = await getDb();
  if (!db) return 0;
  
  const result = await db.select({ count: sql<number>`count(*)` }).from(users);
  return result[0]?.count || 0;
}
