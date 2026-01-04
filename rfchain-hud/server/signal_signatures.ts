/**
 * Signal Signature Library
 * 
 * Built-in signal profiles for automatic classification of RF signals.
 * Includes WiFi, LTE, Bluetooth, ZigBee, LoRa, amateur radio, and more.
 */

import { getDb } from './db';
import { signalSignatures, signatureMatches, analysisResults } from '../drizzle/schema';
import { eq, and, desc, isNull } from 'drizzle-orm';

// Built-in signal signatures
export const BUILT_IN_SIGNATURES = [
  // WiFi 802.11 Family
  {
    name: 'WiFi 802.11b',
    category: 'WiFi',
    subcategory: '802.11b',
    description: 'Legacy 2.4 GHz WiFi using DSSS/CCK modulation',
    bandwidthMinHz: 20e6,
    bandwidthMaxHz: 22e6,
    centerFreqHz: 2437e6, // Channel 6
    modulationType: 'DSSS/CCK',
    symbolRateMin: 1e6,
    symbolRateMax: 11e6,
    typicalPaprDb: 3.5,
    paprToleranceDb: 2.0,
    matchThreshold: 0.7,
    priority: 10,
    isBuiltIn: 1,
    referenceUrl: 'https://standards.ieee.org/standard/802_11b-1999.html',
  },
  {
    name: 'WiFi 802.11g',
    category: 'WiFi',
    subcategory: '802.11g',
    description: '2.4 GHz WiFi using OFDM, up to 54 Mbps',
    bandwidthMinHz: 16e6,
    bandwidthMaxHz: 20e6,
    modulationType: 'OFDM',
    ofdmFftSize: 64,
    ofdmCyclicPrefix: 0.25,
    ofdmSubcarrierSpacing: 312500,
    typicalPaprDb: 8.5,
    paprToleranceDb: 2.0,
    matchThreshold: 0.75,
    priority: 15,
    isBuiltIn: 1,
  },
  {
    name: 'WiFi 802.11n (HT20)',
    category: 'WiFi',
    subcategory: '802.11n',
    description: '2.4/5 GHz WiFi with MIMO, 20 MHz channel',
    bandwidthMinHz: 18e6,
    bandwidthMaxHz: 20e6,
    modulationType: 'OFDM',
    ofdmFftSize: 64,
    ofdmCyclicPrefix: 0.25,
    ofdmSubcarrierSpacing: 312500,
    typicalPaprDb: 9.0,
    paprToleranceDb: 2.5,
    matchThreshold: 0.75,
    priority: 20,
    isBuiltIn: 1,
  },
  {
    name: 'WiFi 802.11n (HT40)',
    category: 'WiFi',
    subcategory: '802.11n',
    description: '2.4/5 GHz WiFi with MIMO, 40 MHz channel',
    bandwidthMinHz: 36e6,
    bandwidthMaxHz: 40e6,
    modulationType: 'OFDM',
    ofdmFftSize: 128,
    ofdmCyclicPrefix: 0.25,
    typicalPaprDb: 9.5,
    paprToleranceDb: 2.5,
    matchThreshold: 0.75,
    priority: 20,
    isBuiltIn: 1,
  },
  {
    name: 'WiFi 802.11ac (VHT80)',
    category: 'WiFi',
    subcategory: '802.11ac',
    description: '5 GHz WiFi, 80 MHz channel, up to 433 Mbps per stream',
    bandwidthMinHz: 75e6,
    bandwidthMaxHz: 80e6,
    modulationType: 'OFDM',
    ofdmFftSize: 256,
    ofdmCyclicPrefix: 0.25,
    typicalPaprDb: 10.0,
    paprToleranceDb: 2.5,
    matchThreshold: 0.75,
    priority: 25,
    isBuiltIn: 1,
  },
  
  // LTE
  {
    name: 'LTE 5 MHz',
    category: 'LTE',
    subcategory: '5MHz',
    description: 'LTE with 5 MHz channel bandwidth, 25 resource blocks',
    bandwidthMinHz: 4.5e6,
    bandwidthMaxHz: 5e6,
    modulationType: 'OFDM/SC-FDMA',
    ofdmFftSize: 512,
    ofdmCyclicPrefix: 0.07,
    ofdmSubcarrierSpacing: 15000,
    typicalPaprDb: 8.0,
    paprToleranceDb: 3.0,
    matchThreshold: 0.7,
    priority: 30,
    isBuiltIn: 1,
  },
  {
    name: 'LTE 10 MHz',
    category: 'LTE',
    subcategory: '10MHz',
    description: 'LTE with 10 MHz channel bandwidth, 50 resource blocks',
    bandwidthMinHz: 9e6,
    bandwidthMaxHz: 10e6,
    modulationType: 'OFDM/SC-FDMA',
    ofdmFftSize: 1024,
    ofdmCyclicPrefix: 0.07,
    ofdmSubcarrierSpacing: 15000,
    typicalPaprDb: 8.5,
    paprToleranceDb: 3.0,
    matchThreshold: 0.7,
    priority: 30,
    isBuiltIn: 1,
  },
  {
    name: 'LTE 20 MHz',
    category: 'LTE',
    subcategory: '20MHz',
    description: 'LTE with 20 MHz channel bandwidth, 100 resource blocks',
    bandwidthMinHz: 18e6,
    bandwidthMaxHz: 20e6,
    modulationType: 'OFDM/SC-FDMA',
    ofdmFftSize: 2048,
    ofdmCyclicPrefix: 0.07,
    ofdmSubcarrierSpacing: 15000,
    typicalPaprDb: 9.0,
    paprToleranceDb: 3.0,
    matchThreshold: 0.7,
    priority: 30,
    isBuiltIn: 1,
  },
  
  // Bluetooth
  {
    name: 'Bluetooth Classic',
    category: 'Bluetooth',
    subcategory: 'BR/EDR',
    description: 'Bluetooth Basic Rate/Enhanced Data Rate, GFSK modulation',

    bandwidthMinHz: 1e6,
    bandwidthMaxHz: 1e6,
    modulationType: 'GFSK',
    symbolRateMin: 1e6,
    symbolRateMax: 3e6,
    typicalPaprDb: 0,
    paprToleranceDb: 1.0,
    matchThreshold: 0.7,
    priority: 25,
    isBuiltIn: 1,
  },
  {
    name: 'Bluetooth Low Energy (BLE)',
    category: 'Bluetooth',
    subcategory: 'BLE',
    description: 'Bluetooth Low Energy, optimized for IoT devices',
    bandwidthMinHz: 1e6,
    bandwidthMaxHz: 2e6,
    modulationType: 'GFSK',
    symbolRateMin: 1e6,
    symbolRateMax: 2e6,
    typicalPaprDb: 0,
    paprToleranceDb: 1.0,
    matchThreshold: 0.7,
    priority: 25,
    isBuiltIn: 1,
  },
  
  // ZigBee / 802.15.4
  {
    name: 'ZigBee 2.4 GHz',
    category: 'ZigBee',
    subcategory: '802.15.4',
    description: 'IEEE 802.15.4 at 2.4 GHz, O-QPSK modulation',
    bandwidthMinHz: 2e6,
    bandwidthMaxHz: 5e6,
    modulationType: 'O-QPSK',
    symbolRateMin: 62500,
    symbolRateMax: 62500,
    typicalPaprDb: 3.0,
    paprToleranceDb: 1.5,
    matchThreshold: 0.7,
    priority: 20,
    isBuiltIn: 1,
  },
  
  // LoRa
  {
    name: 'LoRa SF7',
    category: 'LoRa',
    subcategory: 'SF7',
    description: 'LoRa with Spreading Factor 7, highest data rate',
    bandwidthMinHz: 125e3,
    bandwidthMaxHz: 500e3,
    modulationType: 'CSS', // Chirp Spread Spectrum
    typicalPaprDb: 1.0,
    paprToleranceDb: 1.0,
    matchThreshold: 0.65,
    priority: 20,
    isBuiltIn: 1,
  },
  {
    name: 'LoRa SF12',
    category: 'LoRa',
    subcategory: 'SF12',
    description: 'LoRa with Spreading Factor 12, longest range',
    bandwidthMinHz: 125e3,
    bandwidthMaxHz: 500e3,
    modulationType: 'CSS',
    typicalPaprDb: 1.0,
    paprToleranceDb: 1.0,
    matchThreshold: 0.65,
    priority: 20,
    isBuiltIn: 1,
  },
  
  // Amateur Radio
  {
    name: 'Amateur SSB Voice',
    category: 'Amateur',
    subcategory: 'SSB',
    description: 'Single Sideband voice, typical amateur HF/VHF',
    bandwidthMinHz: 2400,
    bandwidthMaxHz: 3000,
    modulationType: 'SSB',
    typicalPaprDb: 6.0,
    paprToleranceDb: 3.0,
    matchThreshold: 0.6,
    priority: 10,
    isBuiltIn: 1,
  },
  {
    name: 'Amateur FM Voice',
    category: 'Amateur',
    subcategory: 'FM',
    description: 'Narrowband FM voice, VHF/UHF amateur',
    bandwidthMinHz: 10e3,
    bandwidthMaxHz: 16e3,
    modulationType: 'NBFM',
    typicalPaprDb: 0,
    paprToleranceDb: 1.0,
    matchThreshold: 0.65,
    priority: 10,
    isBuiltIn: 1,
  },
  {
    name: 'Amateur FT8',
    category: 'Amateur',
    subcategory: 'FT8',
    description: 'FT8 digital mode, 8-FSK with 15-second transmissions',
    bandwidthMinHz: 50,
    bandwidthMaxHz: 50,
    modulationType: '8-GFSK',
    symbolRateMin: 6.25,
    symbolRateMax: 6.25,
    typicalPaprDb: 0,
    paprToleranceDb: 1.0,
    matchThreshold: 0.7,
    priority: 15,
    isBuiltIn: 1,
  },
  
  // Radar
  {
    name: 'Pulsed Radar',
    category: 'Radar',
    subcategory: 'Pulsed',
    description: 'Generic pulsed radar signal',
    modulationType: 'Pulsed',
    typicalPaprDb: 15.0,
    paprToleranceDb: 5.0,
    matchThreshold: 0.6,
    priority: 5,
    isBuiltIn: 1,
  },
  {
    name: 'FMCW Radar',
    category: 'Radar',
    subcategory: 'FMCW',
    description: 'Frequency Modulated Continuous Wave radar',
    modulationType: 'FMCW',
    typicalPaprDb: 0,
    paprToleranceDb: 2.0,
    matchThreshold: 0.6,
    priority: 5,
    isBuiltIn: 1,
  },
  
  // Digital TV
  {
    name: 'DVB-T',
    category: 'Broadcast',
    subcategory: 'DVB-T',
    description: 'Digital Video Broadcasting - Terrestrial',
    bandwidthMinHz: 6e6,
    bandwidthMaxHz
: 8e6,
    modulationType: 'OFDM',
    ofdmFftSize: 2048,
    ofdmCyclicPrefix: 0.25,
    typicalPaprDb: 11.0,
    paprToleranceDb: 2.0,
    matchThreshold: 0.7,
    priority: 15,
    isBuiltIn: 1,
  },
  {
    name: 'ATSC (US Digital TV)',
    category: 'Broadcast',
    subcategory: 'ATSC',
    description: 'Advanced Television Systems Committee standard',
    bandwidthMinHz: 5.4e6,
    bandwidthMaxHz: 6e6,
    modulationType: '8-VSB',
    typicalPaprDb: 6.5,
    paprToleranceDb: 2.0,
    matchThreshold: 0.7,
    priority: 15,
    isBuiltIn: 1,
  },
];

/**
 * Initialize built-in signatures in the database
 */
export async function initializeBuiltInSignatures(): Promise<{ added: number; existing: number }> {
  const db = await getDb();
  if (!db) {
    return { added: 0, existing: 0 };
  }
  
  let added = 0;
  let existing = 0;
  
  for (const sig of BUILT_IN_SIGNATURES) {
    // Check if signature already exists
    const [existingSig] = await db
      .select()
      .from(signalSignatures)
      .where(and(
        eq(signalSignatures.name, sig.name),
        eq(signalSignatures.isBuiltIn, 1)
      ))
      .limit(1);
    
    if (existingSig) {
      existing++;
      continue;
    }
    
    // Insert new signature
    await db.insert(signalSignatures).values({
      ...sig,
      userId: null, // Built-in signatures have no user
    } as any);
    added++;
  }
  
  return { added, existing };
}

/**
 * Match a signal analysis against all signatures
 */
export async function matchSignatures(
  analysisId: number,
  metrics: {
    bandwidthHz?: number | null;
    paprDb?: number | null;
    modulationType?: string | null;
    ofdmFftSize?: number | null;
    ofdmCyclicPrefix?: number | null;
    symbolRate?: number | null;
  }
): Promise<Array<{
  signatureId: number;
  signatureName: string;
  category: string;
  matchScore: number;
  confidence: 'high' | 'medium' | 'low';
  matchDetails: Record<string, any>;
}>> {
  const db = await getDb();
  if (!db) {
    return [];
  }
  
  // Get all enabled signatures
  const signatures = await db
    .select()
    .from(signalSignatures)
    .where(eq(signalSignatures.enabled, 1))
    .orderBy(desc(signalSignatures.priority));
  
  const matches: Array<{
    signatureId: number;
    signatureName: string;
    category: string;
    matchScore: number;
    confidence: 'high' | 'medium' | 'low';
    matchDetails: Record<string, any>;
  }> = [];
  
  for (const sig of signatures) {
    const matchDetails: Record<string, any> = {};
    let totalScore = 0;
    let totalWeight = 0;
    
    // Bandwidth matching (weight: 3)
    if (metrics.bandwidthHz && sig.bandwidthMinHz && sig.bandwidthMaxHz) {
      const bw = metrics.bandwidthHz;
      if (bw >= sig.bandwidthMinHz && bw <= sig.bandwidthMaxHz) {
        // Perfect match within range
        matchDetails.bandwidth = { match: 'exact', score: 1.0 };
        totalScore += 3.0;
      } else {
        // Calculate how close we are to the range
        const minDist = Math.abs(bw - sig.bandwidthMinHz);
        const maxDist = Math.abs(bw - sig.bandwidthMaxHz);
        const closestDist = Math.min(minDist, maxDist);
        const rangeSize = sig.bandwidthMaxHz - sig.bandwidthMinHz;
        const tolerance = rangeSize * 0.5; // 50% tolerance outside range
        
        if (closestDist < tolerance) {
          const score = 1.0 - (closestDist / tolerance);
          matchDetails.bandwidth = { match: 'partial', score };
          totalScore += score * 3.0;
        } else {
          matchDetails.bandwidth = { match: 'none', score: 0 };
        }
      }
      totalWeight += 3;
    }
    
    // PAPR matching (weight: 2)
    if (metrics.paprDb !== undefined && metrics.paprDb !== null && sig.typicalPaprDb !== null) {
      const paprDiff = Math.abs(metrics.paprDb - sig.typicalPaprDb);
      const tolerance = sig.paprToleranceDb || 3.0;
      
      if (paprDiff <= tolerance) {
        const score = 1.0 - (paprDiff / tolerance);
        matchDetails.papr = { match: paprDiff < tolerance * 0.5 ? 'exact' : 'partial', score };
        totalScore += score * 2.0;
      } else {
        matchDetails.papr = { match: 'none', score: 0 };
      }
      totalWeight += 2;
    }
    
    // Modulation type matching (weight: 4)
    if (metrics.modulationType && sig.modulationType) {
      const sigMod = sig.modulationType.toUpperCase();
      const metricMod = metrics.modulationType.toUpperCase();
      
      if (sigMod === metricMod || sigMod.includes(metricMod) || metricMod.includes(sigMod)) {
        matchDetails.modulation = { match: 'exact', score: 1.0 };
        totalScore += 4.0;
      } else if (
        (sigMod.includes('OFDM') && metricMod.includes('OFDM')) ||
        (sigMod.includes('QAM') && metricMod.includes('QAM')) ||
        (sigMod.includes('FSK') && metricMod.includes('FSK')) ||
        (sigMod.includes('PSK') && metricMod.includes('PSK'))
      ) {
        matchDetails.modulation = { match: 'partial', score: 0.5 };
        totalScore += 2.0;
      } else {
        matchDetails.modulation = { match: 'none', score: 0 };
      }
      totalWeight += 4;
    }
    
    // OFDM parameters matching (weight: 2)
    if (metrics.ofdmFftSize && sig.ofdmFftSize) {
      if (metrics.ofdmFftSize === sig.ofdmFftSize) {
        matchDetails.ofdmFft = { match: 'exact', score: 1.0 };
        totalScore += 2.0;
      } else {
        // Check if it's a power-of-2 neighbor
        const ratio = metrics.ofdmFftSize / sig.ofdmFftSize;
        if (ratio === 2 || ratio === 0.5) {
          matchDetails.ofdmFft = { match: 'partial', score: 0.5 };
          totalScore += 1.0;
        } else {
          matchDetails.ofdmFft = { match: 'none', score: 0 };
        }
      }
      totalWeight += 2;
    }
    
    // Symbol rate matching (weight: 2)
    if (metrics.symbolRate && sig.symbolRateMin && sig.symbolRateMax) {
      const sr = metrics.symbolRate;
      if (sr >= sig.symbolRateMin && sr <= sig.symbolRateMax) {
        matchDetails.symbolRate = { match: 'exact', score: 1.0 };
        totalScore += 2.0;
      } else {
        const minDist = Math.abs(sr - sig.symbolRateMin);
        const maxDist = Math.abs(sr - sig.symbolRateMax);
        const closestDist = Math.min(minDist, maxDist);
        const rangeSize = sig.symbolRateMax - sig.symbolRateMin || sig.symbolRateMin;
        const tolerance = rangeSize * 0.3;
        
        if (closestDist < tolerance) {
          const score = 1.0 - (closestDist / tolerance);
          matchDetails.symbolRate = { match: 'partial', score };
          totalScore += score * 2.0;
        } else {
          matchDetails.symbolRate = { match: 'none', score: 0 };
        }
      }
      totalWeight += 2;
    }
    
    // Calculate final score
    if (totalWeight > 0) {
      const matchScore = totalScore / totalWeight;
      const threshold = sig.matchThreshold || 0.7;
      
      if (matchScore >= threshold) {
        let confidence: 'high' | 'medium' | 'low';
        if (matchScore >= 0.85) {
          confidence = 'high';
        } else if (matchScore >= 0.7) {
          confidence = 'medium';
        } else {
          confidence = 'low';
        }
        
        matches.push({
          signatureId: sig.id,
          signatureName: sig.name,
          category: sig.category,
          matchScore,
          confidence,
          matchDetails,
        });
      }
    }
  }
  
  // Sort by score descending
  matches.sort((a, b) => b.matchScore - a.matchScore);
  
  // Save matches to database
  for (const match of matches) {
    await db.insert(signatureMatches).values({
      analysisId,
      signatureId: match.signatureId,
      matchScore: match.matchScore,
      confidence: match.confidence,
      matchDetails: match.matchDetails,
    });
  }
  
  return matches;
}

/**
 * Get all signatures for a category
 */
export async function getSignaturesByCategory(category: string): Promise<any[]> {
  const db = await getDb();
  if (!db) {
    return [];
  }
  
  return db
    .select()
    .from(signalSignatures)
    .where(and(
      eq(signalSignatures.category, category),
      eq(signalSignatures.enabled, 1)
    ))
    .orderBy(desc(signalSignatures.priority));
}

/**
 * Get all signature categories
 */
export async function getSignatureCategories(): Promise<string[]> {
  const db = await getDb();
  if (!db) {
    return [];
  }
  
  const results = await db
    .selectDistinct({ category: signalSignatures.category })
    .from(signalSignatures)
    .where(eq(signalSignatures.enabled, 1));
  
  return results.map(r => r.category);
}

/**
 * Get matches for an analysis
 */
export async function getMatchesForAnalysis(analysisId: number): Promise<any[]> {
  const db = await getDb();
  if (!db) {
    return [];
  }
  
  const matches = await db
    .select({
      match: signatureMatches,
      signature: signalSignatures,
    })
    .from(signatureMatches)
    .innerJoin(signalSignatures, eq(signatureMatches.signatureId, signalSignatures.id))
    .where(eq(signatureMatches.analysisId, analysisId))
    .orderBy(desc(signatureMatches.matchScore));
  
  return matches.map(m => ({
    ...m.match,
    signature: m.signature,
  }));
}

/**
 * Create a custom user signature
 */
export async function createUserSignature(
  userId: number,
  signature: Omit<typeof BUILT_IN_SIGNATURES[0], 'isBuiltIn'>
): Promise<number> {
  const db = await getDb();
  if (!db) {
    throw new Error('Database not available');
  }
  
  const [result] = await db.insert(signalSignatures).values({
    ...signature,
    userId,
    isBuiltIn: 0,
  } as any);
  
  return result.insertId;
}
