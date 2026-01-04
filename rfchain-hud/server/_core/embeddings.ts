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
 * Format analysis data into text suitable for embedding
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
  anomalies?: Record<string, unknown> | null;
  fullMetrics?: Record<string, unknown> | null;
}): string {
  const parts: string[] = [];
  
  // Add filename context
  if (analysis.filename) {
    parts.push(`Signal file: ${analysis.filename}`);
  }
  
  // Add core metrics
  parts.push("Signal Analysis Metrics:");
  
  if (analysis.avgPowerDbm !== null && analysis.avgPowerDbm !== undefined) {
    parts.push(`- Average Power: ${analysis.avgPowerDbm.toFixed(2)} dBm`);
  }
  if (analysis.peakPowerDbm !== null && analysis.peakPowerDbm !== undefined) {
    parts.push(`- Peak Power: ${analysis.peakPowerDbm.toFixed(2)} dBm`);
  }
  if (analysis.paprDb !== null && analysis.paprDb !== undefined) {
    parts.push(`- PAPR (Peak-to-Average Power Ratio): ${analysis.paprDb.toFixed(2)} dB`);
  }
  if (analysis.bandwidthHz !== null && analysis.bandwidthHz !== undefined) {
    const bwKHz = analysis.bandwidthHz / 1000;
    parts.push(`- Estimated Bandwidth: ${bwKHz.toFixed(2)} kHz`);
  }
  if (analysis.freqOffsetHz !== null && analysis.freqOffsetHz !== undefined) {
    parts.push(`- Frequency Offset: ${analysis.freqOffsetHz.toFixed(2)} Hz`);
  }
  if (analysis.iqImbalanceDb !== null && analysis.iqImbalanceDb !== undefined) {
    parts.push(`- I/Q Imbalance: ${analysis.iqImbalanceDb.toFixed(2)} dB`);
  }
  if (analysis.snrEstimateDb !== null && analysis.snrEstimateDb !== undefined) {
    parts.push(`- SNR Estimate: ${analysis.snrEstimateDb.toFixed(2)} dB`);
  }
  if (analysis.sampleCount !== null && analysis.sampleCount !== undefined) {
    parts.push(`- Sample Count: ${analysis.sampleCount.toLocaleString()}`);
  }
  
  // Add anomaly information
  if (analysis.anomalies && typeof analysis.anomalies === "object") {
    const anomalyEntries = Object.entries(analysis.anomalies);
    const detectedAnomalies = anomalyEntries.filter(([_, v]) => v === true);
    
    if (detectedAnomalies.length > 0) {
      parts.push("Detected Anomalies:");
      detectedAnomalies.forEach(([key, _]) => {
        const readableKey = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        parts.push(`- ${readableKey}`);
      });
    } else {
      parts.push("No anomalies detected in signal.");
    }
  }
  
  // Add extended metrics if available
  if (analysis.fullMetrics && typeof analysis.fullMetrics === "object") {
    const fm = analysis.fullMetrics as Record<string, unknown>;
    
    // Add modulation info if present
    if (fm.modulation_info) {
      const modInfo = fm.modulation_info as Record<string, unknown>;
      parts.push("Modulation Analysis:");
      if (modInfo.detected_type) {
        parts.push(`- Detected Type: ${modInfo.detected_type}`);
      }
      if (modInfo.symbol_rate) {
        parts.push(`- Symbol Rate: ${modInfo.symbol_rate} sps`);
      }
    }
    
    // Add OFDM info if present
    if (fm.ofdm_detected) {
      parts.push("OFDM Signal Detected:");
      if (fm.ofdm_fft_size) {
        parts.push(`- FFT Size: ${fm.ofdm_fft_size}`);
      }
      if (fm.ofdm_cp_length) {
        parts.push(`- Cyclic Prefix Length: ${fm.ofdm_cp_length}`);
      }
    }
  }
  
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
