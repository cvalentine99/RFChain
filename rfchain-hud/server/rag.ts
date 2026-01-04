/**
 * RAG (Retrieval-Augmented Generation) Service for RFChain HUD
 * 
 * This module provides:
 * - Document creation from signal analysis data
 * - Embedding generation via LLM API or local models
 * - Vector storage and retrieval via FAISS
 * - Context augmentation for JARVIS chat
 */

import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { ENV } from './_core/env';
import { generateEmbedding, isEmbeddingError, cosineSimilarity } from './_core/embeddings';
import { db } from './db';
import { analysisResults, analysisEmbeddings, forensicReports, signalUploads } from '../drizzle/schema';
import { eq, desc, and, isNotNull } from 'drizzle-orm';

// Types
export interface SignalDocument {
  analysisId: number;
  filename: string;
  content: string;
  metrics: {
    avgPowerDbm?: number | null;
    peakPowerDbm?: number | null;
    paprDb?: number | null;
    snrEstimateDb?: number | null;
    bandwidthHz?: number | null;
    freqOffsetHz?: number | null;
    iqImbalanceDb?: number | null;
    sampleCount?: number | null;
    durationMs?: number | null;
  };
  anomalies?: Record<string, unknown> | null;
  spectralFeatures?: Record<string, unknown> | null;
  forensicData?: {
    rawInputHash?: string;
    outputHash?: string;
    analysisTimestamp?: string;
  };
}

export interface RAGSearchResult {
  analysisId: number;
  filename: string;
  similarity: number;
  content: string;
  metrics: Record<string, unknown>;
}

export interface RAGContext {
  relevantAnalyses: RAGSearchResult[];
  contextText: string;
  totalMatches: number;
}

// FAISS index path
const FAISS_DATA_DIR = path.join(__dirname, '..', 'data');
const FAISS_INDEX_PATH = path.join(FAISS_DATA_DIR, 'faiss_index');
const FAISS_METADATA_PATH = path.join(FAISS_DATA_DIR, 'faiss_metadata.json');

// In-memory cache for embeddings (for fast retrieval without FAISS)
let embeddingCache: Map<number, { embedding: number[]; metadata: SignalDocument }> = new Map();
let cacheLoaded = false;

/**
 * Create a document from analysis data for embedding
 */
export function createSignalDocument(
  analysis: typeof analysisResults.$inferSelect,
  upload: typeof signalUploads.$inferSelect | null,
  forensic?: typeof forensicReports.$inferSelect | null
): SignalDocument {
  const sections: string[] = [];
  
  // Header
  const filename = upload?.originalName || upload?.filename || `Analysis ${analysis.id}`;
  sections.push(`Signal Analysis Report: ${filename}`);
  sections.push(`Analysis ID: ${analysis.id}`);
  sections.push('');
  
  // Core Metrics
  sections.push('=== SIGNAL METRICS ===');
  
  if (analysis.avgPowerDbm !== null) {
    sections.push(`Average Power: ${analysis.avgPowerDbm.toFixed(2)} dBm`);
  }
  if (analysis.peakPowerDbm !== null) {
    sections.push(`Peak Power: ${analysis.peakPowerDbm.toFixed(2)} dBm`);
  }
  if (analysis.paprDb !== null) {
    sections.push(`PAPR: ${analysis.paprDb.toFixed(2)} dB`);
  }
  if (analysis.snrEstimateDb !== null) {
    sections.push(`SNR Estimate: ${analysis.snrEstimateDb.toFixed(2)} dB`);
  }
  if (analysis.bandwidthHz !== null) {
    const bwKHz = analysis.bandwidthHz / 1000;
    sections.push(`Bandwidth: ${bwKHz.toFixed(2)} kHz`);
  }
  if (analysis.freqOffsetHz !== null) {
    sections.push(`Frequency Offset: ${analysis.freqOffsetHz.toFixed(2)} Hz`);
  }
  if (analysis.iqImbalanceDb !== null) {
    sections.push(`I/Q Imbalance: ${analysis.iqImbalanceDb.toFixed(2)} dB`);
  }
  if (analysis.sampleCount !== null) {
    sections.push(`Sample Count: ${analysis.sampleCount.toLocaleString()}`);
  }
  if (analysis.durationMs !== null) {
    sections.push(`Duration: ${analysis.durationMs.toFixed(2)} ms`);
  }
  
  sections.push('');
  
  // Anomalies
  const anomalies = analysis.anomalies as Record<string, unknown> | null;
  if (anomalies) {
    sections.push('=== ANOMALY DETECTION ===');
    
    const detected: string[] = [];
    if (anomalies.dc_spike) detected.push('DC Spike detected');
    if (anomalies.saturation) detected.push('Signal Saturation/Clipping detected');
    if (anomalies.dropout) detected.push('Signal Dropout detected');
    if (anomalies.periodic_interference) detected.push('Periodic Interference detected');
    
    if (detected.length > 0) {
      detected.forEach(d => sections.push(`- ${d}`));
    } else {
      sections.push('No anomalies detected');
    }
    
    const details = anomalies.details as string[] | undefined;
    if (details && Array.isArray(details)) {
      details.forEach(detail => sections.push(`  Detail: ${detail}`));
    }
    
    sections.push('');
  }
  
  // Full metrics (spectral features)
  const fullMetrics = analysis.fullMetrics as Record<string, unknown> | null;
  if (fullMetrics) {
    sections.push('=== SPECTRAL CHARACTERISTICS ===');
    
    if (fullMetrics.modulation_type) {
      sections.push(`Modulation Type: ${fullMetrics.modulation_type}`);
    }
    if (fullMetrics.symbol_rate) {
      sections.push(`Symbol Rate: ${fullMetrics.symbol_rate} sps`);
    }
    if (fullMetrics.ofdm_detected) {
      sections.push('OFDM Signal Detected');
      if (fullMetrics.ofdm_fft_size) {
        sections.push(`  FFT Size: ${fullMetrics.ofdm_fft_size}`);
      }
      if (fullMetrics.ofdm_cp_length) {
        sections.push(`  Cyclic Prefix: ${fullMetrics.ofdm_cp_length}`);
      }
    }
    if (fullMetrics.noise_floor_dB) {
      sections.push(`Noise Floor: ${fullMetrics.noise_floor_dB} dB`);
    }
    
    sections.push('');
  }
  
  // Forensic data
  if (forensic) {
    sections.push('=== FORENSIC CHAIN OF CUSTODY ===');
    
    if (forensic.rawInputHash) {
      sections.push(`Input Hash (SHA-256): ${forensic.rawInputHash.substring(0, 16)}...`);
    }
    if (forensic.outputSha256) {
      sections.push(`Output Hash (SHA-256): ${forensic.outputSha256.substring(0, 16)}...`);
    }
    sections.push(`Analysis Timestamp: ${forensic.createdAt.toISOString()}`);
    
    sections.push('');
  }
  
  // Signal classification summary
  sections.push('=== SIGNAL CLASSIFICATION ===');
  
  let signalType = 'Unknown';
  if (analysis.bandwidthHz !== null) {
    const bw = analysis.bandwidthHz;
    if (bw < 10000) signalType = 'Narrowband';
    else if (bw < 100000) signalType = 'Standard bandwidth';
    else signalType = 'Wideband';
  }
  sections.push(`Signal Type: ${signalType}`);
  
  let quality = 'Unknown';
  if (analysis.snrEstimateDb !== null) {
    const snr = analysis.snrEstimateDb;
    if (snr > 20) quality = 'High quality (SNR > 20 dB)';
    else if (snr > 10) quality = 'Medium quality (SNR 10-20 dB)';
    else quality = 'Low quality (SNR < 10 dB)';
  }
  sections.push(`Signal Quality: ${quality}`);
  
  return {
    analysisId: analysis.id,
    filename,
    content: sections.join('\n'),
    metrics: {
      avgPowerDbm: analysis.avgPowerDbm,
      peakPowerDbm: analysis.peakPowerDbm,
      paprDb: analysis.paprDb,
      snrEstimateDb: analysis.snrEstimateDb,
      bandwidthHz: analysis.bandwidthHz,
      freqOffsetHz: analysis.freqOffsetHz,
      iqImbalanceDb: analysis.iqImbalanceDb,
      sampleCount: analysis.sampleCount,
      durationMs: analysis.durationMs,
    },
    anomalies,
    spectralFeatures: fullMetrics,
    forensicData: forensic ? {
      rawInputHash: forensic.rawInputHash || undefined,
      outputHash: forensic.outputSha256 || undefined,
      analysisTimestamp: forensic.createdAt.toISOString(),
    } : undefined,
  };
}

/**
 * Index a signal analysis for RAG retrieval
 */
export async function indexAnalysis(analysisId: number): Promise<boolean> {
  try {
    // Get analysis data
    const [analysis] = await db
      .select()
      .from(analysisResults)
      .where(eq(analysisResults.id, analysisId))
      .limit(1);
    
    if (!analysis) {
      console.error(`Analysis ${analysisId} not found`);
      return false;
    }
    
    // Get upload info
    const [upload] = await db
      .select()
      .from(signalUploads)
      .where(eq(signalUploads.id, analysis.signalId))
      .limit(1);
    
    // Get forensic data if available
    const [forensic] = await db
      .select()
      .from(forensicReports)
      .where(eq(forensicReports.analysisId, analysisId))
      .limit(1);
    
    // Create document
    const doc = createSignalDocument(analysis, upload || null, forensic);
    
    // Generate embedding
    const embeddingResult = await generateEmbedding(doc.content);
    
    if (isEmbeddingError(embeddingResult)) {
      console.error(`Failed to generate embedding: ${embeddingResult.error}`);
      return false;
    }
    
    // Check if embedding already exists
    const [existing] = await db
      .select()
      .from(analysisEmbeddings)
      .where(eq(analysisEmbeddings.analysisId, analysisId))
      .limit(1);
    
    const hasAnomalies = doc.anomalies ? 
      Object.values(doc.anomalies).some(v => v === true) ? 1 : 0 : 0;
    
    if (existing) {
      // Update existing
      await db
        .update(analysisEmbeddings)
        .set({
          contentText: doc.content,
          embedding: embeddingResult.embedding,
          signalFilename: doc.filename,
          avgPowerDbm: doc.metrics.avgPowerDbm,
          bandwidthHz: doc.metrics.bandwidthHz,
          hasAnomalies,
        })
        .where(eq(analysisEmbeddings.analysisId, analysisId));
    } else {
      // Insert new
      await db.insert(analysisEmbeddings).values({
        analysisId,
        userId: analysis.userId,
        contentText: doc.content,
        embedding: embeddingResult.embedding,
        signalFilename: doc.filename,
        analysisDate: analysis.createdAt,
        avgPowerDbm: doc.metrics.avgPowerDbm,
        bandwidthHz: doc.metrics.bandwidthHz,
        hasAnomalies,
      });
    }
    
    // Update in-memory cache
    embeddingCache.set(analysisId, {
      embedding: embeddingResult.embedding,
      metadata: doc,
    });
    
    console.log(`Indexed analysis ${analysisId} for RAG`);
    return true;
  } catch (error) {
    console.error(`Error indexing analysis ${analysisId}:`, error);
    return false;
  }
}

/**
 * Load all embeddings into memory cache for fast retrieval
 */
export async function loadEmbeddingCache(): Promise<void> {
  if (cacheLoaded) return;
  
  try {
    const embeddings = await db
      .select()
      .from(analysisEmbeddings)
      .orderBy(desc(analysisEmbeddings.createdAt));
    
    embeddingCache.clear();
    
    for (const emb of embeddings) {
      const embedding = emb.embedding as number[];
      embeddingCache.set(emb.analysisId, {
        embedding,
        metadata: {
          analysisId: emb.analysisId,
          filename: emb.signalFilename || `Analysis ${emb.analysisId}`,
          content: emb.contentText,
          metrics: {
            avgPowerDbm: emb.avgPowerDbm,
            bandwidthHz: emb.bandwidthHz,
          },
          anomalies: emb.hasAnomalies ? { detected: true } : null,
        },
      });
    }
    
    cacheLoaded = true;
    console.log(`Loaded ${embeddingCache.size} embeddings into cache`);
  } catch (error) {
    console.error('Error loading embedding cache:', error);
  }
}

/**
 * Search for similar signal analyses using vector similarity
 */
export async function searchSimilarAnalyses(
  query: string,
  options: {
    k?: number;
    threshold?: number;
    userId?: number;
  } = {}
): Promise<RAGSearchResult[]> {
  const { k = 5, threshold = 0.5, userId } = options;
  
  // Generate query embedding
  const queryEmbeddingResult = await generateEmbedding(query);
  
  if (isEmbeddingError(queryEmbeddingResult)) {
    console.error(`Failed to generate query embedding: ${queryEmbeddingResult.error}`);
    return [];
  }
  
  const queryEmbedding = queryEmbeddingResult.embedding;
  
  // Load cache if needed
  await loadEmbeddingCache();
  
  // Calculate similarities
  const results: RAGSearchResult[] = [];
  
  for (const [analysisId, cached] of embeddingCache) {
    const similarity = cosineSimilarity(queryEmbedding, cached.embedding);
    
    if (similarity >= threshold) {
      results.push({
        analysisId,
        filename: cached.metadata.filename,
        similarity,
        content: cached.metadata.content,
        metrics: cached.metadata.metrics as Record<string, unknown>,
      });
    }
  }
  
  // Sort by similarity and take top k
  results.sort((a, b) => b.similarity - a.similarity);
  return results.slice(0, k);
}

/**
 * Get RAG context for a user query
 */
export async function getRAGContext(
  query: string,
  options: {
    k?: number;
    threshold?: number;
    userId?: number;
    maxContextLength?: number;
  } = {}
): Promise<RAGContext> {
  const { k = 5, threshold = 0.5, userId, maxContextLength = 4000 } = options;
  
  const results = await searchSimilarAnalyses(query, { k, threshold, userId });
  
  // Build context text
  const contextParts: string[] = [];
  let currentLength = 0;
  
  for (const result of results) {
    const entry = `\n--- Signal: ${result.filename} (Similarity: ${(result.similarity * 100).toFixed(1)}%) ---\n${result.content}\n`;
    
    if (currentLength + entry.length > maxContextLength) {
      break;
    }
    
    contextParts.push(entry);
    currentLength += entry.length;
  }
  
  const contextText = contextParts.length > 0
    ? `The following signal analyses are relevant to your query:\n${contextParts.join('\n')}`
    : '';
  
  return {
    relevantAnalyses: results,
    contextText,
    totalMatches: results.length,
  };
}

/**
 * Augment a prompt with RAG context
 */
export async function augmentPromptWithRAG(
  userQuery: string,
  systemPrompt: string,
  options: {
    k?: number;
    threshold?: number;
    userId?: number;
  } = {}
): Promise<{ augmentedSystemPrompt: string; context: RAGContext }> {
  const context = await getRAGContext(userQuery, options);
  
  let augmentedSystemPrompt = systemPrompt;
  
  if (context.contextText) {
    augmentedSystemPrompt = `${systemPrompt}

RELEVANT SIGNAL ANALYSIS DATA:
${context.contextText}

Use the above signal analysis data to provide informed, specific answers about the user's signals. Reference specific metrics, anomalies, and characteristics when relevant.`;
  }
  
  return { augmentedSystemPrompt, context };
}

/**
 * Backfill embeddings for all analyses that don't have them
 */
export async function backfillEmbeddings(
  options: { 
    batchSize?: number;
    onProgress?: (current: number, total: number) => void;
  } = {}
): Promise<{ indexed: number; failed: number; total: number }> {
  const { batchSize = 10, onProgress } = options;
  
  // Get all analyses without embeddings
  const allAnalyses = await db
    .select({ id: analysisResults.id })
    .from(analysisResults)
    .orderBy(desc(analysisResults.createdAt));
  
  const existingEmbeddings = await db
    .select({ analysisId: analysisEmbeddings.analysisId })
    .from(analysisEmbeddings);
  
  const existingIds = new Set(existingEmbeddings.map(e => e.analysisId));
  const toIndex = allAnalyses.filter(a => !existingIds.has(a.id));
  
  let indexed = 0;
  let failed = 0;
  
  for (let i = 0; i < toIndex.length; i += batchSize) {
    const batch = toIndex.slice(i, i + batchSize);
    
    for (const { id } of batch) {
      const success = await indexAnalysis(id);
      if (success) {
        indexed++;
      } else {
        failed++;
      }
    }
    
    if (onProgress) {
      onProgress(Math.min(i + batchSize, toIndex.length), toIndex.length);
    }
    
    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  return { indexed, failed, total: toIndex.length };
}

/**
 * Get RAG statistics
 */
export async function getRAGStats(): Promise<{
  totalEmbeddings: number;
  totalAnalyses: number;
  pendingIndexing: number;
  cacheSize: number;
}> {
  const [embeddingCount] = await db
    .select({ count: analysisEmbeddings.id })
    .from(analysisEmbeddings);
  
  const [analysisCount] = await db
    .select({ count: analysisResults.id })
    .from(analysisResults);
  
  return {
    totalEmbeddings: embeddingCount?.count || 0,
    totalAnalyses: analysisCount?.count || 0,
    pendingIndexing: (analysisCount?.count || 0) - (embeddingCount?.count || 0),
    cacheSize: embeddingCache.size,
  };
}

/**
 * Clear the embedding cache (force reload on next search)
 */
export function clearEmbeddingCache(): void {
  embeddingCache.clear();
  cacheLoaded = false;
}
