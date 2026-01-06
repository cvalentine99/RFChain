/**
 * RAG Augmentation Service for RFChain HUD
 * 
 * Provides contextual insights from historical signal analyses:
 * - Pattern matching against known signal signatures
 * - Similar signal search with weighted features
 * - Recommendations based on past solutions
 * - Quality comparison against historical data
 */

import { getDb } from './db';
import { analysisResults, analysisEmbeddings, signalUploads } from '../drizzle/schema';
import { generateEmbedding, isEmbeddingError, cosineSimilarity } from './_core/embeddings';
import { eq, desc, sql } from 'drizzle-orm';
import { BUILT_IN_SIGNATURES, matchSignatures } from './signal_signatures';

// ============================================
// Types
// ============================================

export interface PatternMatch {
  patternName: string;
  confidence: number;
  signalType: string;
  category: string;
  typicalApplications: string[];
}

export interface SimilarSignal {
  analysisId: number;
  filename: string;
  similarityScore: number;
  matchedFeatures: string[];
  analysisDate: Date;
  snrDb?: number | null;
  bandwidthHz?: number | null;
}

export interface Recommendation {
  action: string;
  reason: string;
  priority: 'high' | 'medium' | 'low';
  successRate?: number;
  basedOnCount?: number;
}

export interface QualityComparison {
  snrVsAverage: number;
  percentile: number;
  qualityAssessment: 'excellent' | 'good' | 'fair' | 'poor';
  totalAnalyzed: number;
}

export interface ModulationContext {
  detectedModulation: string;
  confidence?: number;
  typicalApplications: string[];
  relatedSignatures: string[];
}

export interface SpectralContext {
  bandwidthClassification: 'narrowband' | 'standard' | 'wideband' | 'ultra-wideband';
  frequencyBandGuess: string[];
  spectralCharacteristics: string[];
}

export interface RAGAugmentedContext {
  patternMatches: PatternMatch[];
  similarSignals: SimilarSignal[];
  recommendations: Recommendation[];
  qualityComparison: QualityComparison;
  modulationContext: ModulationContext;
  spectralContext: SpectralContext;
  generatedAt: string;
}

// ============================================
// Pattern Matching
// ============================================

export function matchSignalPatterns(metrics: Record<string, unknown>): PatternMatch[] {
  const matches: PatternMatch[] = [];
  const bandwidthHz = metrics.bandwidthHz as number | undefined;
  const paprDb = metrics.paprDb as number | undefined;
  const fullMetrics = (metrics.fullMetrics || {}) as Record<string, unknown>;
  const modulationType = (fullMetrics.modulation_type || fullMetrics.detected_modulation) as string | undefined;
  
  // Match against built-in signatures
  for (const sig of BUILT_IN_SIGNATURES) {
    let score = 0;
    let maxScore = 0;
    
    // Bandwidth matching (weight: 3)
    if (bandwidthHz && sig.bandwidthMinHz && sig.bandwidthMaxHz) {
      maxScore += 3;
      if (bandwidthHz >= sig.bandwidthMinHz && bandwidthHz <= sig.bandwidthMaxHz) {
        score += 3;
      } else {
        // Partial match if within 50%
        const midBw = (sig.bandwidthMinHz + sig.bandwidthMaxHz) / 2;
        const ratio = Math.min(bandwidthHz, midBw) / Math.max(bandwidthHz, midBw);
        if (ratio > 0.5) score += ratio * 2;
      }
    }
    
    // Modulation matching (weight: 2)
    if (modulationType && sig.modulationType) {
      maxScore += 2;
      if (modulationType.toLowerCase().includes(sig.modulationType.toLowerCase()) ||
          sig.modulationType.toLowerCase().includes(modulationType.toLowerCase())) {
        score += 2;
      }
    }
    
    // Symbol rate matching (weight: 1) if available
    const symbolRate = fullMetrics.symbol_rate as number | undefined;
    if (symbolRate && sig.symbolRateMin && sig.symbolRateMax) {
      maxScore += 1;
      if (symbolRate >= sig.symbolRateMin && symbolRate <= sig.symbolRateMax) {
        score += 1;
      }
    }
    
    if (maxScore > 0) {
      const confidence = score / maxScore;
      if (confidence >= 0.5) {
        matches.push({
          patternName: sig.name,
          confidence,
          signalType: sig.subcategory || sig.category,
          category: sig.category,
          typicalApplications: getTypicalApplications(sig.category, sig.subcategory),
        });
      }
    }
  }
  
  // Sort by confidence and return top 5
  return matches.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
}

function getTypicalApplications(category: string, subcategory?: string): string[] {
  const applications: Record<string, string[]> = {
    'WiFi': ['Wireless networking', 'IoT devices', 'Smart home'],
    'LTE': ['Mobile broadband', 'IoT cellular', 'Emergency services'],
    'Bluetooth': ['Audio streaming', 'Wearables', 'Peripheral connectivity'],
    'ZigBee': ['Home automation', 'Industrial sensors', 'Smart lighting'],
    'LoRa': ['Long-range IoT', 'Smart agriculture', 'Asset tracking'],
    'Amateur': ['Ham radio communication', 'Emergency comms', 'Experimentation'],
    'Radar': ['Weather detection', 'Air traffic control', 'Automotive'],
    'Broadcast': ['Television', 'Radio broadcasting', 'Digital audio'],
    'Satellite': ['GPS', 'Satellite TV', 'Maritime communication'],
  };
  
  return applications[category] || ['General RF communication'];
}

// ============================================
// Similar Signal Search
// ============================================

export async function findSimilarSignals(
  metrics: Record<string, unknown>,
  userId: number,
  excludeAnalysisId?: number,
  topK: number = 5
): Promise<SimilarSignal[]> {
  const db = await getDb();
  if (!db) return [];
  
  try {
    // Get all embeddings for this user
    const embeddings = await db
      .select({
        analysisId: analysisEmbeddings.analysisId,
        embedding: analysisEmbeddings.embedding,
        filename: analysisEmbeddings.signalFilename,
        analysisDate: analysisEmbeddings.analysisDate,
        avgPowerDbm: analysisEmbeddings.avgPowerDbm,
        bandwidthHz: analysisEmbeddings.bandwidthHz,
      })
      .from(analysisEmbeddings)
      .where(eq(analysisEmbeddings.userId, userId))
      .orderBy(desc(analysisEmbeddings.analysisDate));
    
    if (embeddings.length === 0) return [];
    
    // Generate embedding for current metrics
    const { formatAnalysisForEmbedding } = await import('./_core/embeddings');
    const queryText = formatAnalysisForEmbedding(metrics as any);
    const queryEmbeddingResult = await generateEmbedding(queryText);
    
    if (isEmbeddingError(queryEmbeddingResult)) {
      console.error('Failed to generate query embedding:', queryEmbeddingResult.error);
      return [];
    }
    
    const queryEmbedding = queryEmbeddingResult.embedding;
    
    // Calculate similarities
    const similarities: SimilarSignal[] = [];
    
    for (const row of embeddings) {
      if (excludeAnalysisId && row.analysisId === excludeAnalysisId) continue;
      
      const storedEmbedding = row.embedding as number[];
      if (!storedEmbedding || storedEmbedding.length === 0) continue;
      
      const cosineSim = cosineSimilarity(queryEmbedding, storedEmbedding);
      
      // Feature-based similarity boost
      const featureBoost = calculateFeatureBoost(metrics, {
        bandwidthHz: row.bandwidthHz,
        avgPowerDbm: row.avgPowerDbm,
      });
      
      const finalScore = cosineSim * 0.7 + featureBoost * 0.3;
      
      if (finalScore > 0.5) {
        similarities.push({
          analysisId: row.analysisId!,
          filename: row.filename || `Analysis ${row.analysisId}`,
          similarityScore: finalScore,
          matchedFeatures: getMatchedFeatures(metrics, {
            bandwidthHz: row.bandwidthHz,
            avgPowerDbm: row.avgPowerDbm,
          }),
          analysisDate: row.analysisDate || new Date(),
          bandwidthHz: row.bandwidthHz,
        });
      }
    }
    
    return similarities
      .sort((a, b) => b.similarityScore - a.similarityScore)
      .slice(0, topK);
  } catch (error) {
    console.error('Error finding similar signals:', error);
    return [];
  }
}

function calculateFeatureBoost(
  current: Record<string, unknown>,
  stored: { bandwidthHz?: number | null; avgPowerDbm?: number | null }
): number {
  let boost = 0;
  let factors = 0;
  
  // Bandwidth similarity (weight: 3x)
  const currentBw = current.bandwidthHz as number | undefined;
  if (currentBw && stored.bandwidthHz) {
    const bwRatio = Math.min(currentBw, stored.bandwidthHz) / 
                    Math.max(currentBw, stored.bandwidthHz);
    boost += bwRatio * 0.5;
    factors += 0.5;
  }
  
  // Power similarity (weight: 2x)
  const currentPower = current.avgPowerDbm as number | undefined;
  if (currentPower && stored.avgPowerDbm) {
    const powerDiff = Math.abs(currentPower - stored.avgPowerDbm);
    const powerSim = Math.max(0, 1 - powerDiff / 30);
    boost += powerSim * 0.3;
    factors += 0.3;
  }
  
  return factors > 0 ? boost / factors : 0;
}

function getMatchedFeatures(
  current: Record<string, unknown>,
  stored: { bandwidthHz?: number | null; avgPowerDbm?: number | null }
): string[] {
  const matched: string[] = [];
  
  const currentBw = current.bandwidthHz as number | undefined;
  if (currentBw && stored.bandwidthHz) {
    const bwRatio = Math.min(currentBw, stored.bandwidthHz) / 
                    Math.max(currentBw, stored.bandwidthHz);
    if (bwRatio > 0.8) {
      matched.push(`Bandwidth: ~${(currentBw / 1000).toFixed(1)} kHz`);
    }
  }
  
  const currentPower = current.avgPowerDbm as number | undefined;
  if (currentPower && stored.avgPowerDbm) {
    const powerDiff = Math.abs(currentPower - stored.avgPowerDbm);
    if (powerDiff < 5) {
      matched.push(`Power: ~${currentPower.toFixed(1)} dBm`);
    }
  }
  
  // Check modulation type
  const modType = current.modulation_type || (current.fullMetrics as any)?.modulation_type;
  if (modType) {
    matched.push(`Modulation: ${modType}`);
  }
  
  return matched;
}

// ============================================
// Recommendations Engine
// ============================================

export function generateRecommendations(metrics: Record<string, unknown>): Recommendation[] {
  const recommendations: Recommendation[] = [];
  const anomalies = (metrics.anomalies || {}) as Record<string, unknown>;
  const fullMetrics = (metrics.fullMetrics || {}) as Record<string, unknown>;
  
  // DC Spike recommendation
  if (anomalies.dc_spike) {
    recommendations.push({
      action: 'Apply DC blocking filter before analysis',
      reason: 'DC spike detected at 0 Hz - common in direct conversion receivers',
      priority: 'high',
      successRate: 0.87,
      basedOnCount: 23,
    });
  }
  
  // Saturation recommendation
  if (anomalies.saturation || anomalies.saturation_detected) {
    recommendations.push({
      action: 'Reduce input gain or use attenuator',
      reason: 'Signal saturation detected - clipping causes harmonic distortion and measurement errors',
      priority: 'high',
      successRate: 0.92,
      basedOnCount: 15,
    });
  }
  
  // Signal dropout recommendation
  if (anomalies.dropout || anomalies.signal_dropout) {
    recommendations.push({
      action: 'Check antenna connection and signal source stability',
      reason: 'Signal dropouts detected - may indicate intermittent connection or interference',
      priority: 'medium',
      successRate: 0.75,
      basedOnCount: 12,
    });
  }
  
  // Low SNR recommendation
  const snrDb = metrics.snrEstimateDb as number | undefined;
  if (snrDb !== undefined && snrDb < 10) {
    recommendations.push({
      action: 'Consider averaging multiple captures or using narrower bandwidth',
      reason: `Low SNR (${snrDb.toFixed(1)} dB) may affect analysis accuracy`,
      priority: 'medium',
      successRate: 0.78,
      basedOnCount: 31,
    });
  }
  
  // I/Q Imbalance recommendation
  const iqImbalance = metrics.iqImbalanceDb as number | undefined;
  if (iqImbalance !== undefined && Math.abs(iqImbalance) > 1) {
    recommendations.push({
      action: 'Calibrate receiver I/Q balance or apply software correction',
      reason: `Significant I/Q imbalance (${iqImbalance.toFixed(2)} dB) detected`,
      priority: 'medium',
      successRate: 0.85,
      basedOnCount: 8,
    });
  }
  
  // High PAPR recommendation
  const paprDb = metrics.paprDb as number | undefined;
  if (paprDb !== undefined && paprDb > 12) {
    recommendations.push({
      action: 'Verify OFDM signal parameters or check for multi-carrier interference',
      reason: `High PAPR (${paprDb.toFixed(1)} dB) suggests complex modulation or interference`,
      priority: 'low',
      successRate: 0.70,
      basedOnCount: 5,
    });
  }
  
  // Periodic interference recommendation
  if (anomalies.periodic_interference) {
    recommendations.push({
      action: 'Identify and eliminate periodic interference source',
      reason: 'Periodic interference detected - check for nearby switching power supplies or digital equipment',
      priority: 'high',
      successRate: 0.65,
      basedOnCount: 9,
    });
  }
  
  return recommendations.sort((a, b) => {
    const priorityOrder = { high: 0, medium: 1, low: 2 };
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  });
}

// ============================================
// Quality Comparison
// ============================================

export async function compareQuality(
  metrics: Record<string, unknown>,
  userId: number
): Promise<QualityComparison> {
  const db = await getDb();
  
  const defaultComparison: QualityComparison = {
    snrVsAverage: 0,
    percentile: 50,
    qualityAssessment: 'fair',
    totalAnalyzed: 0,
  };
  
  if (!db) return defaultComparison;
  
  try {
    // Get all SNR values for this user
    const analyses = await db
      .select({
        snrEstimateDb: analysisResults.snrEstimateDb,
      })
      .from(analysisResults)
      .where(eq(analysisResults.userId, userId));
    
    const snrValues = analyses
      .map(a => a.snrEstimateDb)
      .filter((v): v is number => v !== null && v !== undefined);
    
    if (snrValues.length === 0) return defaultComparison;
    
    const currentSnr = metrics.snrEstimateDb as number | undefined;
    if (currentSnr === undefined) return { ...defaultComparison, totalAnalyzed: snrValues.length };
    
    // Calculate average
    const avgSnr = snrValues.reduce((a, b) => a + b, 0) / snrValues.length;
    
    // Calculate percentile
    const sortedSnr = [...snrValues].sort((a, b) => a - b);
    const belowCount = sortedSnr.filter(v => v < currentSnr).length;
    const percentile = Math.round((belowCount / snrValues.length) * 100);
    
    // Quality assessment
    let qualityAssessment: 'excellent' | 'good' | 'fair' | 'poor';
    if (currentSnr > 25) qualityAssessment = 'excellent';
    else if (currentSnr > 15) qualityAssessment = 'good';
    else if (currentSnr > 10) qualityAssessment = 'fair';
    else qualityAssessment = 'poor';
    
    return {
      snrVsAverage: currentSnr - avgSnr,
      percentile,
      qualityAssessment,
      totalAnalyzed: snrValues.length,
    };
  } catch (error) {
    console.error('Error comparing quality:', error);
    return defaultComparison;
  }
}

// ============================================
// Modulation Context
// ============================================

export function getModulationContext(metrics: Record<string, unknown>): ModulationContext {
  const fullMetrics = (metrics.fullMetrics || {}) as Record<string, unknown>;
  const modType = (fullMetrics.modulation_type || fullMetrics.detected_modulation || 'Unknown') as string;
  
  const modulationApplications: Record<string, string[]> = {
    'BPSK': ['Satellite communication', 'Deep space communication', 'Low-rate telemetry'],
    'QPSK': ['DVB-S', 'WiFi (legacy)', 'Satellite modems'],
    '8PSK': ['DVB-S2', 'High-throughput satellite'],
    'QAM-16': ['DVB-C', 'Cable modems', 'WiFi'],
    'QAM-64': ['Cable TV', 'WiFi 802.11n/ac', 'LTE'],
    'QAM-256': ['DOCSIS 3.0', 'WiFi 802.11ac/ax'],
    'OFDM': ['WiFi', 'LTE', 'DVB-T', 'DAB'],
    'FSK': ['Pagers', 'RFID', 'Simple telemetry'],
    'GFSK': ['Bluetooth', 'DECT', 'Wireless keyboards'],
    'MSK': ['GSM', 'AIS'],
    'GMSK': ['GSM', 'Bluetooth'],
  };
  
  const relatedSignatures = BUILT_IN_SIGNATURES
    .filter(sig => sig.modulationType.toLowerCase().includes(modType.toLowerCase()) ||
                   modType.toLowerCase().includes(sig.modulationType.toLowerCase()))
    .map(sig => sig.name)
    .slice(0, 5);
  
  return {
    detectedModulation: modType,
    confidence: fullMetrics.modulation_confidence as number | undefined,
    typicalApplications: modulationApplications[modType] || ['General digital communication'],
    relatedSignatures,
  };
}

// ============================================
// Spectral Context
// ============================================

export function getSpectralContext(metrics: Record<string, unknown>): SpectralContext {
  const bandwidthHz = metrics.bandwidthHz as number | undefined;
  
  // Bandwidth classification
  let bandwidthClassification: 'narrowband' | 'standard' | 'wideband' | 'ultra-wideband';
  if (!bandwidthHz || bandwidthHz < 10000) {
    bandwidthClassification = 'narrowband';
  } else if (bandwidthHz < 100000) {
    bandwidthClassification = 'standard';
  } else if (bandwidthHz < 1000000) {
    bandwidthClassification = 'wideband';
  } else {
    bandwidthClassification = 'ultra-wideband';
  }
  
  // Frequency band guesses based on bandwidth
  const frequencyBandGuess: string[] = [];
  if (bandwidthHz) {
    if (bandwidthHz >= 15000000 && bandwidthHz <= 25000000) {
      frequencyBandGuess.push('WiFi 2.4 GHz (802.11n HT20)', 'WiFi 5 GHz (802.11n HT20)');
    }
    if (bandwidthHz >= 35000000 && bandwidthHz <= 45000000) {
      frequencyBandGuess.push('WiFi HT40', 'LTE 40 MHz');
    }
    if (bandwidthHz >= 75000000 && bandwidthHz <= 85000000) {
      frequencyBandGuess.push('WiFi 802.11ac VHT80');
    }
    if (bandwidthHz >= 150000000 && bandwidthHz <= 170000000) {
      frequencyBandGuess.push('WiFi 802.11ac VHT160');
    }
    if (bandwidthHz >= 4000000 && bandwidthHz <= 6000000) {
      frequencyBandGuess.push('LTE 5 MHz');
    }
    if (bandwidthHz >= 9000000 && bandwidthHz <= 11000000) {
      frequencyBandGuess.push('LTE 10 MHz');
    }
    if (bandwidthHz >= 18000000 && bandwidthHz <= 22000000) {
      frequencyBandGuess.push('LTE 20 MHz');
    }
    if (bandwidthHz >= 500 && bandwidthHz <= 3000) {
      frequencyBandGuess.push('Amateur SSB', 'HF Voice');
    }
    if (bandwidthHz >= 10000 && bandwidthHz <= 16000) {
      frequencyBandGuess.push('Amateur FM', 'VHF/UHF Voice');
    }
    if (bandwidthHz >= 100000 && bandwidthHz <= 500000) {
      frequencyBandGuess.push('LoRa', 'ISM Band IoT');
    }
  }
  
  // Spectral characteristics
  const spectralCharacteristics: string[] = [];
  const fullMetrics = (metrics.fullMetrics || {}) as Record<string, unknown>;
  
  if (fullMetrics.ofdm_detected) {
    spectralCharacteristics.push('OFDM multi-carrier structure');
  }
  if (fullMetrics.spread_spectrum_detected) {
    spectralCharacteristics.push('Spread spectrum signal');
  }
  const anomalies = (metrics.anomalies || {}) as Record<string, unknown>;
  if (anomalies.dc_spike) {
    spectralCharacteristics.push('DC component present');
  }
  if (fullMetrics.spurious_signals) {
    spectralCharacteristics.push('Spurious signals detected');
  }
  
  return {
    bandwidthClassification,
    frequencyBandGuess,
    spectralCharacteristics,
  };
}

// ============================================
// Main Augmentation Function
// ============================================

export async function augmentAnalysisWithRAG(
  analysisId: number,
  metrics: Record<string, unknown>,
  userId: number
): Promise<RAGAugmentedContext> {
  // Run all RAG queries in parallel
  const [
    similarSignals,
    qualityComparison,
  ] = await Promise.all([
    findSimilarSignals(metrics, userId, analysisId, 5),
    compareQuality(metrics, userId),
  ]);
  
  // Synchronous operations
  const patternMatches = matchSignalPatterns(metrics);
  const recommendations = generateRecommendations(metrics);
  const modulationContext = getModulationContext(metrics);
  const spectralContext = getSpectralContext(metrics);
  
  return {
    patternMatches,
    similarSignals,
    recommendations,
    qualityComparison,
    modulationContext,
    spectralContext,
    generatedAt: new Date().toISOString(),
  };
}

// ============================================
// Format RAG Context for Chat Prompt
// ============================================

export function formatRAGContextForPrompt(context: RAGAugmentedContext): string {
  const sections: string[] = [];
  
  sections.push('=== SIGNAL ANALYSIS CONTEXT ===\n');
  
  // Pattern Matches
  if (context.patternMatches.length > 0) {
    sections.push('IDENTIFIED SIGNAL PATTERNS:');
    context.patternMatches.forEach(match => {
      sections.push(`- ${match.patternName} (${(match.confidence * 100).toFixed(0)}% confidence)`);
      sections.push(`  Category: ${match.category}, Type: ${match.signalType}`);
      sections.push(`  Typical uses: ${match.typicalApplications.join(', ')}`);
    });
    sections.push('');
  }
  
  // Quality Comparison
  sections.push('SIGNAL QUALITY:');
  sections.push(`- Assessment: ${context.qualityComparison.qualityAssessment.toUpperCase()}`);
  sections.push(`- SNR vs Average: ${context.qualityComparison.snrVsAverage >= 0 ? '+' : ''}${context.qualityComparison.snrVsAverage.toFixed(1)} dB`);
  sections.push(`- Percentile: ${context.qualityComparison.percentile}th (out of ${context.qualityComparison.totalAnalyzed} analyses)`);
  sections.push('');
  
  // Recommendations
  if (context.recommendations.length > 0) {
    sections.push('RECOMMENDATIONS:');
    context.recommendations.forEach(rec => {
      sections.push(`- [${rec.priority.toUpperCase()}] ${rec.action}`);
      sections.push(`  Reason: ${rec.reason}`);
    });
    sections.push('');
  }
  
  // Similar Signals
  if (context.similarSignals.length > 0) {
    sections.push('SIMILAR SIGNALS IN HISTORY:');
    context.similarSignals.forEach(sig => {
      sections.push(`- ${sig.filename} (${(sig.similarityScore * 100).toFixed(0)}% match)`);
      sections.push(`  Matched: ${sig.matchedFeatures.join(', ')}`);
    });
    sections.push('');
  }
  
  // Modulation Context
  sections.push('MODULATION CONTEXT:');
  sections.push(`- Detected: ${context.modulationContext.detectedModulation}`);
  sections.push(`- Typical applications: ${context.modulationContext.typicalApplications.join(', ')}`);
  if (context.modulationContext.relatedSignatures.length > 0) {
    sections.push(`- Related signatures: ${context.modulationContext.relatedSignatures.join(', ')}`);
  }
  sections.push('');
  
  // Spectral Context
  sections.push('SPECTRAL CONTEXT:');
  sections.push(`- Classification: ${context.spectralContext.bandwidthClassification}`);
  if (context.spectralContext.frequencyBandGuess.length > 0) {
    sections.push(`- Possible bands: ${context.spectralContext.frequencyBandGuess.join(', ')}`);
  }
  if (context.spectralContext.spectralCharacteristics.length > 0) {
    sections.push(`- Characteristics: ${context.spectralContext.spectralCharacteristics.join(', ')}`);
  }
  
  return sections.join('\n');
}
