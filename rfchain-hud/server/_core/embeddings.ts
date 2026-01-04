/**
 * Embedding service for RAG vector generation
 * Uses the Forge API to generate text embeddings for semantic search
 */
import { ENV } from "./env";

export type EmbeddingResult = {
  embedding: number[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
};

export type EmbeddingError = {
  error: string;
  code: "SERVICE_ERROR" | "INVALID_INPUT" | "RATE_LIMITED";
  details?: string;
};

/**
 * Generate an embedding vector for the given text
 */
export async function generateEmbedding(
  text: string
): Promise<EmbeddingResult | EmbeddingError> {
  try {
    if (!ENV.forgeApiUrl || !ENV.forgeApiKey) {
      return {
        error: "Embedding service is not configured",
        code: "SERVICE_ERROR",
        details: "BUILT_IN_FORGE_API_URL or BUILT_IN_FORGE_API_KEY is not set",
      };
    }

    if (!text || text.trim().length === 0) {
      return {
        error: "Empty text provided",
        code: "INVALID_INPUT",
        details: "Text to embed cannot be empty",
      };
    }

    // Truncate text if too long (max ~8000 tokens ≈ 32000 chars for safety)
    const truncatedText = text.length > 32000 ? text.substring(0, 32000) : text;

    const baseUrl = ENV.forgeApiUrl.replace(/\/$/, "");
    const response = await fetch(`${baseUrl}/v1/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${ENV.forgeApiKey}`,
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: truncatedText,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      
      if (response.status === 429) {
        return {
          error: "Rate limited by embedding service",
          code: "RATE_LIMITED",
          details: errorText,
        };
      }
      
      return {
        error: "Embedding generation failed",
        code: "SERVICE_ERROR",
        details: `${response.status} ${response.statusText}: ${errorText}`,
      };
    }

    const result = await response.json();
    
    return {
      embedding: result.data[0].embedding,
      model: result.model,
      usage: result.usage,
    };
  } catch (error) {
    return {
      error: "Embedding generation failed",
      code: "SERVICE_ERROR",
      details: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

/**
 * Generate embeddings for multiple texts in batch
 */
export async function generateEmbeddingsBatch(
  texts: string[]
): Promise<(EmbeddingResult | EmbeddingError)[]> {
  // Process in parallel with concurrency limit
  const results: (EmbeddingResult | EmbeddingError)[] = [];
  const concurrencyLimit = 5;
  
  for (let i = 0; i < texts.length; i += concurrencyLimit) {
    const batch = texts.slice(i, i + concurrencyLimit);
    const batchResults = await Promise.all(
      batch.map((text) => generateEmbedding(text))
    );
    results.push(...batchResults);
  }
  
  return results;
}

/**
 * Calculate cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  
  if (normA === 0 || normB === 0) {
    return 0;
  }
  
  return dotProduct / (normA * normB);
}

/**
 * Format analysis data into comprehensive text suitable for embedding.
 * 
 * This creates a rich document that captures all signal characteristics
 * for semantic search and RAG retrieval, including:
 * - Core power and frequency metrics
 * - Spectral characteristics (bandwidth, noise floor, spurs)
 * - Modulation analysis (type, symbol rate, constellation)
 * - Anomaly detection results
 * - Signal quality assessment
 * - Cyclostationary features
 */
export function formatAnalysisForEmbedding(analysis: {
  filename?: string;
  avgPowerDbm?: number | null;
  peakPowerDbm?: number | null;
  paprDb?: number | null;
  bandwidthHz?: number | null;
  freqOffsetHz?: number | null;
  iqImbalanceDb?: number | null;
  snrEstimateDb?: number | null;
  sampleCount?: number | null;
  durationMs?: number | null;
  sampleRate?: number | null;
  anomalies?: Record<string, unknown> | null;
  fullMetrics?: Record<string, unknown> | null;
}): string {
  const parts: string[] = [];
  
  // Header with filename
  if (analysis.filename) {
    parts.push(`=== Signal Analysis: ${analysis.filename} ===`);
  } else {
    parts.push(`=== Signal Analysis ===`);
  }
  parts.push("");
  
  // Core Power Metrics
  parts.push("POWER CHARACTERISTICS:");
  if (analysis.avgPowerDbm !== null && analysis.avgPowerDbm !== undefined) {
    parts.push(`- Average Power: ${analysis.avgPowerDbm.toFixed(2)} dBm`);
  }
  if (analysis.peakPowerDbm !== null && analysis.peakPowerDbm !== undefined) {
    parts.push(`- Peak Power: ${analysis.peakPowerDbm.toFixed(2)} dBm`);
  }
  if (analysis.paprDb !== null && analysis.paprDb !== undefined) {
    parts.push(`- PAPR (Peak-to-Average Power Ratio): ${analysis.paprDb.toFixed(2)} dB`);
    // Add interpretation
    if (analysis.paprDb > 10) {
      parts.push(`  (High PAPR suggests OFDM or multi-carrier signal)`);
    } else if (analysis.paprDb < 3) {
      parts.push(`  (Low PAPR suggests constant envelope modulation like FSK/PSK)`);
    }
  }
  parts.push("");
  
  // Frequency Characteristics
  parts.push("FREQUENCY CHARACTERISTICS:");
  if (analysis.bandwidthHz !== null && analysis.bandwidthHz !== undefined) {
    const bwKHz = analysis.bandwidthHz / 1000;
    parts.push(`- Occupied Bandwidth: ${bwKHz.toFixed(2)} kHz`);
    // Classify bandwidth
    if (analysis.bandwidthHz < 10000) {
      parts.push(`  (Narrowband signal)`);
    } else if (analysis.bandwidthHz < 100000) {
      parts.push(`  (Standard bandwidth signal)`);
    } else if (analysis.bandwidthHz < 1000000) {
      parts.push(`  (Wideband signal)`);
    } else {
      parts.push(`  (Ultra-wideband signal)`);
    }
  }
  if (analysis.freqOffsetHz !== null && analysis.freqOffsetHz !== undefined) {
    parts.push(`- Center Frequency Offset: ${analysis.freqOffsetHz.toFixed(2)} Hz`);
  }
  parts.push("");
  
  // Signal Quality
  parts.push("SIGNAL QUALITY:");
  if (analysis.snrEstimateDb !== null && analysis.snrEstimateDb !== undefined) {
    parts.push(`- SNR Estimate: ${analysis.snrEstimateDb.toFixed(2)} dB`);
    // Quality classification
    if (analysis.snrEstimateDb > 25) {
      parts.push(`  (Excellent signal quality)`);
    } else if (analysis.snrEstimateDb > 15) {
      parts.push(`  (Good signal quality)`);
    } else if (analysis.snrEstimateDb > 10) {
      parts.push(`  (Fair signal quality)`);
    } else {
      parts.push(`  (Poor signal quality, high noise)`);
    }
  }
  if (analysis.iqImbalanceDb !== null && analysis.iqImbalanceDb !== undefined) {
    parts.push(`- I/Q Imbalance: ${analysis.iqImbalanceDb.toFixed(2)} dB`);
    if (Math.abs(analysis.iqImbalanceDb) > 1) {
      parts.push(`  (Significant I/Q imbalance detected)`);
    }
  }
  parts.push("");
  
  // Sample Information
  parts.push("SAMPLE INFORMATION:");
  if (analysis.sampleCount !== null && analysis.sampleCount !== undefined) {
    parts.push(`- Sample Count: ${analysis.sampleCount.toLocaleString()}`);
  }
  if (analysis.durationMs !== null && analysis.durationMs !== undefined) {
    parts.push(`- Duration: ${analysis.durationMs.toFixed(2)} ms`);
  }
  if (analysis.sampleRate !== null && analysis.sampleRate !== undefined) {
    const srMHz = analysis.sampleRate / 1000000;
    parts.push(`- Sample Rate: ${srMHz.toFixed(3)} MHz`);
  }
  parts.push("");
  
  // Anomaly Detection
  if (analysis.anomalies && typeof analysis.anomalies === "object") {
    parts.push("ANOMALY DETECTION:");
    const anomalyEntries = Object.entries(analysis.anomalies);
    const detectedAnomalies = anomalyEntries.filter(([k, v]) => v === true && k !== 'details');
    
    if (detectedAnomalies.length > 0) {
      detectedAnomalies.forEach(([key]) => {
        const readableKey = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        parts.push(`- DETECTED: ${readableKey}`);
      });
      
      // Add details if available
      const details = analysis.anomalies.details as string[] | undefined;
      if (details && Array.isArray(details)) {
        details.forEach(detail => parts.push(`  Detail: ${detail}`));
      }
    } else {
      parts.push("- No anomalies detected");
    }
    parts.push("");
  }
  
  // Extended Metrics from fullMetrics
  if (analysis.fullMetrics && typeof analysis.fullMetrics === "object") {
    const fm = analysis.fullMetrics as Record<string, unknown>;
    
    // Modulation Analysis
    if (fm.modulation_info || fm.modulation_type || fm.detected_modulation) {
      parts.push("MODULATION ANALYSIS:");
      const modInfo = (fm.modulation_info || {}) as Record<string, unknown>;
      
      const modType = modInfo.detected_type || fm.modulation_type || fm.detected_modulation;
      if (modType) {
        parts.push(`- Detected Modulation: ${modType}`);
      }
      if (modInfo.symbol_rate || fm.symbol_rate) {
        parts.push(`- Symbol Rate: ${modInfo.symbol_rate || fm.symbol_rate} symbols/sec`);
      }
      if (modInfo.bits_per_symbol || fm.bits_per_symbol) {
        parts.push(`- Bits per Symbol: ${modInfo.bits_per_symbol || fm.bits_per_symbol}`);
      }
      if (modInfo.constellation_points || fm.constellation_points) {
        parts.push(`- Constellation Points: ${modInfo.constellation_points || fm.constellation_points}`);
      }
      parts.push("");
    }
    
    // OFDM Detection
    if (fm.ofdm_detected || fm.ofdm_info) {
      parts.push("OFDM CHARACTERISTICS:");
      parts.push("- OFDM Signal Detected: Yes");
      if (fm.ofdm_fft_size) {
        parts.push(`- FFT Size: ${fm.ofdm_fft_size}`);
      }
      if (fm.ofdm_cp_length) {
        parts.push(`- Cyclic Prefix Length: ${fm.ofdm_cp_length}`);
      }
      if (fm.ofdm_subcarrier_spacing) {
        parts.push(`- Subcarrier Spacing: ${fm.ofdm_subcarrier_spacing} Hz`);
      }
      if (fm.ofdm_num_subcarriers) {
        parts.push(`- Number of Subcarriers: ${fm.ofdm_num_subcarriers}`);
      }
      parts.push("");
    }
    
    // Spectral Features
    if (fm.noise_floor_dB || fm.spurious_signals || fm.spectral_peaks) {
      parts.push("SPECTRAL FEATURES:");
      if (fm.noise_floor_dB) {
        parts.push(`- Noise Floor: ${fm.noise_floor_dB} dB`);
      }
      if (fm.dc_spike_detected) {
        parts.push(`- DC Spike: Detected`);
      }
      if (fm.spurious_signals && Array.isArray(fm.spurious_signals)) {
        parts.push(`- Spurious Signals: ${(fm.spurious_signals as unknown[]).length} detected`);
      }
      if (fm.spectral_peaks && Array.isArray(fm.spectral_peaks)) {
        parts.push(`- Spectral Peaks: ${(fm.spectral_peaks as unknown[]).length} identified`);
      }
      parts.push("");
    }
    
    // Cyclostationary Features
    if (fm.cyclostationary_features || fm.cyclic_frequencies) {
      parts.push("CYCLOSTATIONARY ANALYSIS:");
      if (fm.cyclic_frequencies && Array.isArray(fm.cyclic_frequencies)) {
        parts.push(`- Cyclic Frequencies Detected: ${(fm.cyclic_frequencies as unknown[]).length}`);
      }
      if (fm.symbol_rate_estimate) {
        parts.push(`- Symbol Rate Estimate: ${fm.symbol_rate_estimate} Hz`);
      }
      parts.push("");
    }
    
    // Spreading Code Analysis
    if (fm.spreading_codes || fm.chip_rate) {
      parts.push("SPREAD SPECTRUM ANALYSIS:");
      if (fm.chip_rate) {
        parts.push(`- Chip Rate: ${fm.chip_rate} chips/sec`);
      }
      if (fm.spreading_factor) {
        parts.push(`- Spreading Factor: ${fm.spreading_factor}`);
      }
      if (fm.code_type) {
        parts.push(`- Code Type: ${fm.code_type}`);
      }
      parts.push("");
    }
    
    // Digital Analysis Results
    if (fm.ber_estimate || fm.evm_percent) {
      parts.push("DIGITAL SIGNAL QUALITY:");
      if (fm.ber_estimate) {
        parts.push(`- BER Estimate: ${fm.ber_estimate}`);
      }
      if (fm.evm_percent) {
        parts.push(`- EVM: ${fm.evm_percent}%`);
      }
      if (fm.mer_db) {
        parts.push(`- MER: ${fm.mer_db} dB`);
      }
      parts.push("");
    }
  }
  
  // Signal Classification Summary
  parts.push("SIGNAL CLASSIFICATION SUMMARY:");
  
  // Determine signal type
  let signalType = "Unknown";
  if (analysis.bandwidthHz) {
    if (analysis.bandwidthHz < 10000) signalType = "Narrowband";
    else if (analysis.bandwidthHz < 100000) signalType = "Standard Bandwidth";
    else if (analysis.bandwidthHz < 1000000) signalType = "Wideband";
    else signalType = "Ultra-Wideband";
  }
  parts.push(`- Signal Type: ${signalType}`);
  
  // Quality assessment
  let quality = "Unknown";
  if (analysis.snrEstimateDb !== null && analysis.snrEstimateDb !== undefined) {
    if (analysis.snrEstimateDb > 25) quality = "Excellent";
    else if (analysis.snrEstimateDb > 15) quality = "Good";
    else if (analysis.snrEstimateDb > 10) quality = "Fair";
    else quality = "Poor";
  }
  parts.push(`- Signal Quality: ${quality}`);
  
  // Anomaly summary
  const hasAnomalies = analysis.anomalies && 
    Object.entries(analysis.anomalies).some(([k, v]) => v === true && k !== 'details');
  parts.push(`- Anomalies Present: ${hasAnomalies ? 'Yes' : 'No'}`);
  
  return parts.join("\n");
}

/**
 * Check if a result is an error
 */
export function isEmbeddingError(
  result: EmbeddingResult | EmbeddingError
): result is EmbeddingError {
  return "error" in result;
}
