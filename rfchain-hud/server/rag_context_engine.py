#!/usr/bin/env python3
"""
RAG Context Engine for Signal Analysis Pipeline

This module provides contextual insights from historical signal analyses
during the analysis process. It queries the vector store to find similar
signals and provides:
- Historical pattern matches
- Past solutions and recommendations
- Comparative analysis
- Anomaly context

Integration points:
1. After anomaly detection → query for similar anomalies
2. After modulation classification → query for similar modulation types
3. After quality assessment → query for similar quality issues
4. After spectral analysis → match against historical signatures
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Try to import FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[RAG] Warning: FAISS not available, using fallback similarity search")

# Try to import sentence transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[RAG] Warning: sentence-transformers not available, embeddings disabled")


@dataclass
class SimilarSignal:
    """A similar signal from history"""
    signal_id: str
    filename: str
    similarity_score: float
    analysis_date: str
    key_metrics: Dict[str, Any]
    anomalies: List[str]
    notes: Optional[str] = None


@dataclass
class PatternMatch:
    """A matched signal pattern/profile"""
    pattern_name: str
    confidence: float
    description: str
    typical_characteristics: Dict[str, Any]
    examples: List[str]


@dataclass
class Recommendation:
    """A recommendation based on historical solutions"""
    action: str
    reason: str
    confidence: float
    based_on: List[str]  # Signal IDs this recommendation is based on
    success_rate: Optional[float] = None


@dataclass
class RAGContext:
    """Complete RAG context for an analysis"""
    similar_signals: List[SimilarSignal]
    pattern_matches: List[PatternMatch]
    recommendations: List[Recommendation]
    quality_comparison: Dict[str, Any]
    anomaly_context: Dict[str, Any]
    modulation_context: Dict[str, Any]
    spectral_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'similar_signals': [asdict(s) for s in self.similar_signals],
            'pattern_matches': [asdict(p) for p in self.pattern_matches],
            'recommendations': [asdict(r) for r in self.recommendations],
            'quality_comparison': self.quality_comparison,
            'anomaly_context': self.anomaly_context,
            'modulation_context': self.modulation_context,
            'spectral_context': self.spectral_context,
        }


class RAGContextEngine:
    """
    RAG Context Engine for signal analysis.
    
    Provides contextual insights from historical analyses during
    the analysis pipeline.
    """
    
    # Known signal patterns/profiles
    KNOWN_PATTERNS = {
        'wifi_80211n': {
            'name': 'WiFi 802.11n',
            'description': 'IEEE 802.11n wireless LAN signal',
            'characteristics': {
                'modulation': ['OFDM', 'BPSK', 'QPSK', '16-QAM', '64-QAM'],
                'bandwidth_hz': [20000000, 40000000],
                'ofdm_fft_size': [64, 128],
                'papr_db_range': [8, 12],
            }
        },
        'lte': {
            'name': 'LTE',
            'description': '4G LTE cellular signal',
            'characteristics': {
                'modulation': ['OFDM', 'QPSK', '16-QAM', '64-QAM'],
                'bandwidth_hz': [1400000, 3000000, 5000000, 10000000, 15000000, 20000000],
                'ofdm_fft_size': [128, 256, 512, 1024, 1536, 2048],
                'papr_db_range': [7, 11],
            }
        },
        'bluetooth': {
            'name': 'Bluetooth',
            'description': 'Bluetooth wireless signal',
            'characteristics': {
                'modulation': ['GFSK', '8DPSK', 'DQPSK'],
                'bandwidth_hz': [1000000, 2000000],
                'papr_db_range': [0, 3],
            }
        },
        'fsk_narrowband': {
            'name': 'Narrowband FSK',
            'description': 'Narrowband FSK signal (IoT, telemetry)',
            'characteristics': {
                'modulation': ['FSK', '2FSK', '4FSK', 'GFSK'],
                'bandwidth_hz': [5000, 50000],
                'papr_db_range': [0, 2],
            }
        },
        'qam_cable': {
            'name': 'Cable QAM',
            'description': 'Cable TV/DOCSIS QAM signal',
            'characteristics': {
                'modulation': ['64-QAM', '256-QAM', '1024-QAM'],
                'bandwidth_hz': [6000000, 8000000],
                'papr_db_range': [5, 8],
            }
        },
        'radar_pulse': {
            'name': 'Radar Pulse',
            'description': 'Pulsed radar signal',
            'characteristics': {
                'modulation': ['Pulse', 'LFM', 'Barker'],
                'papr_db_range': [10, 20],
                'anomalies': ['pulsed_signal', 'high_papr'],
            }
        },
        'spread_spectrum': {
            'name': 'Spread Spectrum',
            'description': 'Direct sequence spread spectrum signal',
            'characteristics': {
                'modulation': ['DSSS', 'CDMA', 'BPSK'],
                'papr_db_range': [0, 3],
            }
        },
    }
    
    # Common anomaly solutions
    ANOMALY_SOLUTIONS = {
        'dc_spike': [
            Recommendation(
                action='Apply DC blocking filter (high-pass at 0.1% of sample rate)',
                reason='DC spike indicates DC offset in signal or LO leakage',
                confidence=0.9,
                based_on=[],
                success_rate=0.95
            ),
            Recommendation(
                action='Check receiver LO isolation',
                reason='LO leakage can cause DC component',
                confidence=0.7,
                based_on=[],
                success_rate=0.8
            ),
        ],
        'saturation': [
            Recommendation(
                action='Reduce input gain by 6-10 dB',
                reason='Signal clipping detected, ADC saturation',
                confidence=0.95,
                based_on=[],
                success_rate=0.98
            ),
            Recommendation(
                action='Add attenuator before ADC',
                reason='Hardware attenuation more reliable than digital gain',
                confidence=0.85,
                based_on=[],
                success_rate=0.9
            ),
        ],
        'iq_imbalance': [
            Recommendation(
                action='Apply I/Q imbalance correction (amplitude and phase)',
                reason='Receiver I/Q paths have gain/phase mismatch',
                confidence=0.9,
                based_on=[],
                success_rate=0.85
            ),
            Recommendation(
                action='Calibrate receiver with known signal',
                reason='Hardware calibration more accurate than blind correction',
                confidence=0.8,
                based_on=[],
                success_rate=0.9
            ),
        ],
        'frequency_drift': [
            Recommendation(
                action='Apply frequency tracking loop (Costas loop or PLL)',
                reason='Carrier frequency offset detected',
                confidence=0.85,
                based_on=[],
                success_rate=0.9
            ),
            Recommendation(
                action='Check oscillator temperature stability',
                reason='Temperature drift causes frequency offset',
                confidence=0.7,
                based_on=[],
                success_rate=0.75
            ),
        ],
        'low_snr': [
            Recommendation(
                action='Apply matched filtering or correlation',
                reason='Improve SNR through optimal filtering',
                confidence=0.8,
                based_on=[],
                success_rate=0.85
            ),
            Recommendation(
                action='Average multiple captures',
                reason='Coherent averaging improves SNR by sqrt(N)',
                confidence=0.9,
                based_on=[],
                success_rate=0.95
            ),
            Recommendation(
                action='Check antenna connection and positioning',
                reason='Poor antenna can cause low SNR',
                confidence=0.6,
                based_on=[],
                success_rate=0.7
            ),
        ],
        'spurious_signals': [
            Recommendation(
                action='Apply notch filter at spurious frequencies',
                reason='Remove interfering signals',
                confidence=0.85,
                based_on=[],
                success_rate=0.9
            ),
            Recommendation(
                action='Check for intermodulation products',
                reason='Spurs may be IMD from strong signals',
                confidence=0.7,
                based_on=[],
                success_rate=0.65
            ),
        ],
    }
    
    def __init__(self, 
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG context engine.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            model_name: Sentence transformer model for embeddings
        """
        self.index_path = index_path or os.environ.get('RAG_INDEX_PATH', 'rag_index.faiss')
        self.metadata_path = metadata_path or os.environ.get('RAG_METADATA_PATH', 'rag_metadata.json')
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.model = None
        
        # Load FAISS index if available
        if FAISS_AVAILABLE and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"[RAG] Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"[RAG] Failed to load FAISS index: {e}")
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"[RAG] Loaded {len(self.metadata)} metadata entries")
            except Exception as e:
                print(f"[RAG] Failed to load metadata: {e}")
        
        # Load embedding model
        if SBERT_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"[RAG] Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"[RAG] Failed to load embedding model: {e}")
    
    def _create_query_text(self, metrics: Dict[str, Any]) -> str:
        """Create a text query from signal metrics for embedding."""
        parts = []
        
        # Power characteristics
        if 'avg_power_dbm' in metrics:
            parts.append(f"Average power {metrics['avg_power_dbm']:.1f} dBm")
        if 'papr_db' in metrics:
            parts.append(f"PAPR {metrics['papr_db']:.1f} dB")
        
        # Frequency characteristics
        if 'bandwidth_estimate_hz' in metrics:
            bw_khz = metrics['bandwidth_estimate_hz'] / 1000
            parts.append(f"Bandwidth {bw_khz:.1f} kHz")
        
        # Quality
        if 'snr_estimate_db' in metrics:
            parts.append(f"SNR {metrics['snr_estimate_db']:.1f} dB")
        
        # Modulation
        if 'modulation_type' in metrics:
            parts.append(f"Modulation {metrics['modulation_type']}")
        
        # Anomalies
        anomalies = metrics.get('anomalies', {})
        detected = [k for k, v in anomalies.items() if v is True]
        if detected:
            parts.append(f"Anomalies: {', '.join(detected)}")
        
        return ". ".join(parts)
    
    def _find_similar_signals(self, 
                              metrics: Dict[str, Any], 
                              top_k: int = 5,
                              min_similarity: float = 0.5) -> List[SimilarSignal]:
        """Find similar signals from history using vector search."""
        similar = []
        
        if not self.index or not self.model or self.index.ntotal == 0:
            return similar
        
        try:
            # Create query embedding
            query_text = self._create_query_text(metrics)
            query_embedding = self.model.encode([query_text])[0]
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                similarity = 1 - dist  # Convert distance to similarity
                if similarity < min_similarity:
                    continue
                
                meta = self.metadata[idx]
                similar.append(SimilarSignal(
                    signal_id=meta.get('signal_id', str(idx)),
                    filename=meta.get('filename', 'unknown'),
                    similarity_score=float(similarity),
                    analysis_date=meta.get('analysis_date', ''),
                    key_metrics=meta.get('metrics', {}),
                    anomalies=meta.get('anomalies', []),
                    notes=meta.get('notes'),
                ))
        except Exception as e:
            print(f"[RAG] Error finding similar signals: {e}")
        
        return similar
    
    def _match_patterns(self, metrics: Dict[str, Any]) -> List[PatternMatch]:
        """Match signal against known patterns/profiles."""
        matches = []
        
        modulation = metrics.get('modulation_type', '').upper()
        bandwidth = metrics.get('bandwidth_estimate_hz', 0)
        papr = metrics.get('papr_db', 0)
        ofdm_fft = metrics.get('ofdm_fft_size', 0)
        
        for pattern_id, pattern in self.KNOWN_PATTERNS.items():
            confidence = 0.0
            match_reasons = []
            
            chars = pattern['characteristics']
            
            # Check modulation match
            if 'modulation' in chars:
                for mod in chars['modulation']:
                    if mod.upper() in modulation or modulation in mod.upper():
                        confidence += 0.3
                        match_reasons.append(f"Modulation matches {mod}")
                        break
            
            # Check bandwidth match
            if 'bandwidth_hz' in chars:
                bw_list = chars['bandwidth_hz']
                for bw in bw_list:
                    if abs(bandwidth - bw) / max(bw, 1) < 0.2:  # Within 20%
                        confidence += 0.25
                        match_reasons.append(f"Bandwidth matches {bw/1e6:.1f} MHz")
                        break
            
            # Check PAPR match
            if 'papr_db_range' in chars:
                papr_min, papr_max = chars['papr_db_range']
                if papr_min <= papr <= papr_max:
                    confidence += 0.2
                    match_reasons.append(f"PAPR in expected range [{papr_min}, {papr_max}] dB")
            
            # Check OFDM FFT size
            if 'ofdm_fft_size' in chars and ofdm_fft > 0:
                if ofdm_fft in chars['ofdm_fft_size']:
                    confidence += 0.25
                    match_reasons.append(f"OFDM FFT size matches {ofdm_fft}")
            
            if confidence >= 0.4:
                matches.append(PatternMatch(
                    pattern_name=pattern['name'],
                    confidence=min(confidence, 1.0),
                    description=pattern['description'] + ". " + "; ".join(match_reasons),
                    typical_characteristics=chars,
                    examples=[],
                ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:3]  # Return top 3 matches
    
    def _get_anomaly_recommendations(self, 
                                     anomalies: Dict[str, bool],
                                     similar_signals: List[SimilarSignal]) -> List[Recommendation]:
        """Get recommendations based on detected anomalies and historical solutions."""
        recommendations = []
        
        # Get recommendations from known solutions
        for anomaly, detected in anomalies.items():
            if not detected:
                continue
            
            # Normalize anomaly name
            anomaly_key = anomaly.lower().replace(' ', '_')
            
            if anomaly_key in self.ANOMALY_SOLUTIONS:
                for rec in self.ANOMALY_SOLUTIONS[anomaly_key]:
                    # Add signal IDs from similar signals that had this anomaly
                    based_on = [s.signal_id for s in similar_signals 
                               if anomaly_key in [a.lower().replace(' ', '_') for a in s.anomalies]]
                    rec_copy = Recommendation(
                        action=rec.action,
                        reason=rec.reason,
                        confidence=rec.confidence,
                        based_on=based_on[:3],  # Limit to 3 examples
                        success_rate=rec.success_rate,
                    )
                    recommendations.append(rec_copy)
        
        # Deduplicate and sort by confidence
        seen_actions = set()
        unique_recs = []
        for rec in sorted(recommendations, key=lambda x: x.confidence, reverse=True):
            if rec.action not in seen_actions:
                seen_actions.add(rec.action)
                unique_recs.append(rec)
        
        return unique_recs[:5]  # Return top 5 recommendations
    
    def _compute_quality_comparison(self, 
                                    metrics: Dict[str, Any],
                                    similar_signals: List[SimilarSignal]) -> Dict[str, Any]:
        """Compare signal quality against historical averages."""
        comparison = {
            'snr_vs_average': None,
            'snr_percentile': None,
            'quality_assessment': 'unknown',
            'historical_average': {},
            'this_signal': {},
        }
        
        if not similar_signals:
            # Use absolute thresholds
            snr = metrics.get('snr_estimate_db', 0)
            if snr > 25:
                comparison['quality_assessment'] = 'excellent'
            elif snr > 15:
                comparison['quality_assessment'] = 'good'
            elif snr > 10:
                comparison['quality_assessment'] = 'fair'
            else:
                comparison['quality_assessment'] = 'poor'
            comparison['this_signal']['snr_db'] = snr
            return comparison
        
        # Compute averages from similar signals
        snr_values = [s.key_metrics.get('snr_estimate_db', 0) for s in similar_signals 
                     if 'snr_estimate_db' in s.key_metrics]
        power_values = [s.key_metrics.get('avg_power_dbm', 0) for s in similar_signals
                       if 'avg_power_dbm' in s.key_metrics]
        
        current_snr = metrics.get('snr_estimate_db', 0)
        current_power = metrics.get('avg_power_dbm', 0)
        
        if snr_values:
            avg_snr = np.mean(snr_values)
            comparison['snr_vs_average'] = current_snr - avg_snr
            comparison['snr_percentile'] = np.sum(np.array(snr_values) < current_snr) / len(snr_values) * 100
            comparison['historical_average']['snr_db'] = float(avg_snr)
        
        if power_values:
            avg_power = np.mean(power_values)
            comparison['historical_average']['avg_power_dbm'] = float(avg_power)
        
        comparison['this_signal']['snr_db'] = current_snr
        comparison['this_signal']['avg_power_dbm'] = current_power
        
        # Quality assessment
        if comparison['snr_vs_average'] is not None:
            if comparison['snr_vs_average'] > 5:
                comparison['quality_assessment'] = f'excellent (SNR {comparison["snr_vs_average"]:.1f} dB above average)'
            elif comparison['snr_vs_average'] > 0:
                comparison['quality_assessment'] = f'good (SNR {comparison["snr_vs_average"]:.1f} dB above average)'
            elif comparison['snr_vs_average'] > -5:
                comparison['quality_assessment'] = f'fair (SNR {abs(comparison["snr_vs_average"]):.1f} dB below average)'
            else:
                comparison['quality_assessment'] = f'poor (SNR {abs(comparison["snr_vs_average"]):.1f} dB below average)'
        
        return comparison
    
    def _get_anomaly_context(self,
                             anomalies: Dict[str, bool],
                             similar_signals: List[SimilarSignal]) -> Dict[str, Any]:
        """Get context for detected anomalies from historical data."""
        context = {
            'detected_anomalies': [],
            'historical_frequency': {},
            'similar_cases': [],
        }
        
        detected = [k for k, v in anomalies.items() if v is True]
        context['detected_anomalies'] = detected
        
        if not similar_signals:
            return context
        
        # Count anomaly frequency in similar signals
        anomaly_counts = {}
        for signal in similar_signals:
            for anomaly in signal.anomalies:
                anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        
        total = len(similar_signals)
        context['historical_frequency'] = {
            k: f"{v}/{total} ({v/total*100:.0f}%)" 
            for k, v in anomaly_counts.items()
        }
        
        # Find signals with same anomalies
        for anomaly in detected:
            matching = [s for s in similar_signals 
                       if anomaly.lower() in [a.lower() for a in s.anomalies]]
            if matching:
                context['similar_cases'].append({
                    'anomaly': anomaly,
                    'matching_signals': [s.filename for s in matching[:3]],
                    'count': len(matching),
                })
        
        return context
    
    def _get_modulation_context(self,
                                metrics: Dict[str, Any],
                                similar_signals: List[SimilarSignal]) -> Dict[str, Any]:
        """Get context for modulation classification."""
        context = {
            'detected_modulation': metrics.get('modulation_type', 'unknown'),
            'confidence': metrics.get('modulation_confidence', 0),
            'similar_modulations': [],
            'typical_applications': [],
        }
        
        mod_type = context['detected_modulation'].upper()
        
        # Map modulation to typical applications
        mod_applications = {
            'BPSK': ['Satellite communications', 'Deep space links', 'Low SNR environments'],
            'QPSK': ['Satellite TV', 'Cable modems', 'Cellular (LTE control)'],
            'QAM': ['Cable TV', 'WiFi', 'LTE data'],
            '16-QAM': ['WiFi (802.11a/g/n)', 'LTE', 'Cable modems'],
            '64-QAM': ['WiFi (802.11n/ac)', 'LTE', 'DOCSIS'],
            '128-QAM': ['High-speed data links', 'Cable (DOCSIS 3.0)'],
            '256-QAM': ['WiFi (802.11ac/ax)', 'DOCSIS 3.1'],
            'OFDM': ['WiFi', 'LTE', 'DVB-T', 'DAB'],
            'FSK': ['IoT', 'Paging', 'Telemetry'],
            'GFSK': ['Bluetooth', 'DECT', 'Zigbee'],
        }
        
        for mod, apps in mod_applications.items():
            if mod in mod_type:
                context['typical_applications'] = apps
                break
        
        # Find similar modulations in history
        if similar_signals:
            mod_counts = {}
            for signal in similar_signals:
                sig_mod = signal.key_metrics.get('modulation_type', 'unknown')
                mod_counts[sig_mod] = mod_counts.get(sig_mod, 0) + 1
            
            context['similar_modulations'] = [
                {'type': k, 'count': v} 
                for k, v in sorted(mod_counts.items(), key=lambda x: x[1], reverse=True)
            ][:5]
        
        return context
    
    def _get_spectral_context(self,
                              metrics: Dict[str, Any],
                              similar_signals: List[SimilarSignal]) -> Dict[str, Any]:
        """Get context for spectral characteristics."""
        context = {
            'bandwidth_classification': 'unknown',
            'spectral_efficiency': None,
            'frequency_band_guess': [],
            'similar_bandwidths': [],
        }
        
        bandwidth = metrics.get('bandwidth_estimate_hz', 0)
        
        # Classify bandwidth
        if bandwidth < 10000:
            context['bandwidth_classification'] = 'narrowband'
            context['frequency_band_guess'] = ['VHF/UHF land mobile', 'IoT', 'Telemetry']
        elif bandwidth < 100000:
            context['bandwidth_classification'] = 'standard'
            context['frequency_band_guess'] = ['Amateur radio', 'PMR', 'Trunked radio']
        elif bandwidth < 1000000:
            context['bandwidth_classification'] = 'wideband'
            context['frequency_band_guess'] = ['WiFi', 'Bluetooth', 'ISM band']
        elif bandwidth < 20000000:
            context['bandwidth_classification'] = 'very wideband'
            context['frequency_band_guess'] = ['WiFi 802.11n/ac', 'LTE', 'Cellular']
        else:
            context['bandwidth_classification'] = 'ultra-wideband'
            context['frequency_band_guess'] = ['WiFi 802.11ax', '5G NR', 'Radar']
        
        # Compute spectral efficiency if we have symbol rate
        symbol_rate = metrics.get('symbol_rate', 0)
        if symbol_rate > 0 and bandwidth > 0:
            context['spectral_efficiency'] = symbol_rate / bandwidth
        
        # Find similar bandwidths in history
        if similar_signals:
            bw_values = [s.key_metrics.get('bandwidth_estimate_hz', 0) for s in similar_signals
                        if 'bandwidth_estimate_hz' in s.key_metrics]
            if bw_values:
                context['similar_bandwidths'] = {
                    'min_hz': min(bw_values),
                    'max_hz': max(bw_values),
                    'avg_hz': np.mean(bw_values),
                    'this_signal_hz': bandwidth,
                }
        
        return context
    
    def get_context(self, metrics: Dict[str, Any]) -> RAGContext:
        """
        Get complete RAG context for a signal analysis.
        
        This is the main entry point for the analysis pipeline.
        
        Args:
            metrics: Dictionary of signal metrics from analysis
            
        Returns:
            RAGContext with all contextual information
        """
        # Find similar signals
        similar_signals = self._find_similar_signals(metrics)
        
        # Match against known patterns
        pattern_matches = self._match_patterns(metrics)
        
        # Get anomaly information
        anomalies = metrics.get('anomalies', {})
        if isinstance(anomalies, dict):
            anomaly_context = self._get_anomaly_context(anomalies, similar_signals)
            recommendations = self._get_anomaly_recommendations(anomalies, similar_signals)
        else:
            anomaly_context = {'detected_anomalies': [], 'historical_frequency': {}, 'similar_cases': []}
            recommendations = []
        
        # Get quality comparison
        quality_comparison = self._compute_quality_comparison(metrics, similar_signals)
        
        # Get modulation context
        modulation_context = self._get_modulation_context(metrics, similar_signals)
        
        # Get spectral context
        spectral_context = self._get_spectral_context(metrics, similar_signals)
        
        return RAGContext(
            similar_signals=similar_signals,
            pattern_matches=pattern_matches,
            recommendations=recommendations,
            quality_comparison=quality_comparison,
            anomaly_context=anomaly_context,
            modulation_context=modulation_context,
            spectral_context=spectral_context,
        )
    
    def add_to_index(self, 
                     signal_id: str,
                     filename: str,
                     metrics: Dict[str, Any],
                     analysis_date: str,
                     notes: Optional[str] = None) -> bool:
        """
        Add a new signal analysis to the RAG index.
        
        Args:
            signal_id: Unique identifier for the signal
            filename: Original filename
            metrics: Analysis metrics
            analysis_date: Date of analysis
            notes: Optional notes
            
        Returns:
            True if successfully added
        """
        if not self.model:
            print("[RAG] Cannot add to index: embedding model not available")
            return False
        
        try:
            # Create embedding
            query_text = self._create_query_text(metrics)
            embedding = self.model.encode([query_text])[0]
            embedding = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding)
            
            # Add to index
            if self.index is None:
                # Create new index
                dim = embedding.shape[1]
                self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            
            self.index.add(embedding)
            
            # Add metadata
            anomalies = metrics.get('anomalies', {})
            detected_anomalies = [k for k, v in anomalies.items() if v is True] if isinstance(anomalies, dict) else []
            
            self.metadata.append({
                'signal_id': signal_id,
                'filename': filename,
                'analysis_date': analysis_date,
                'metrics': {
                    'avg_power_dbm': metrics.get('avg_power_dbm'),
                    'snr_estimate_db': metrics.get('snr_estimate_db'),
                    'bandwidth_estimate_hz': metrics.get('bandwidth_estimate_hz'),
                    'modulation_type': metrics.get('modulation_type'),
                    'papr_db': metrics.get('papr_db'),
                },
                'anomalies': detected_anomalies,
                'notes': notes,
            })
            
            # Save index and metadata
            if self.index_path:
                faiss.write_index(self.index, self.index_path)
            if self.metadata_path:
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            
            print(f"[RAG] Added signal {signal_id} to index (total: {self.index.ntotal})")
            return True
            
        except Exception as e:
            print(f"[RAG] Error adding to index: {e}")
            return False


def main():
    """Test the RAG context engine."""
    # Sample metrics
    test_metrics = {
        'avg_power_dbm': -45.2,
        'peak_power_dbm': -32.1,
        'papr_db': 9.5,
        'snr_estimate_db': 18.3,
        'bandwidth_estimate_hz': 20000000,
        'modulation_type': 'OFDM/64-QAM',
        'ofdm_fft_size': 64,
        'anomalies': {
            'dc_spike': True,
            'saturation': False,
            'iq_imbalance': True,
            'frequency_drift': False,
        }
    }
    
    # Create engine
    engine = RAGContextEngine()
    
    # Get context
    context = engine.get_context(test_metrics)
    
    # Print results
    print("\n" + "="*60)
    print("RAG CONTEXT ENGINE TEST")
    print("="*60)
    
    print("\n--- Pattern Matches ---")
    for match in context.pattern_matches:
        print(f"  {match.pattern_name}: {match.confidence*100:.0f}% confidence")
        print(f"    {match.description}")
    
    print("\n--- Recommendations ---")
    for rec in context.recommendations:
        print(f"  [{rec.confidence*100:.0f}%] {rec.action}")
        print(f"    Reason: {rec.reason}")
    
    print("\n--- Quality Comparison ---")
    print(f"  Assessment: {context.quality_comparison['quality_assessment']}")
    
    print("\n--- Anomaly Context ---")
    print(f"  Detected: {context.anomaly_context['detected_anomalies']}")
    
    print("\n--- Modulation Context ---")
    print(f"  Type: {context.modulation_context['detected_modulation']}")
    print(f"  Applications: {context.modulation_context['typical_applications']}")
    
    print("\n--- Spectral Context ---")
    print(f"  Classification: {context.spectral_context['bandwidth_classification']}")
    print(f"  Band Guess: {context.spectral_context['frequency_band_guess']}")
    
    # Output as JSON
    print("\n--- Full JSON Output ---")
    print(json.dumps(context.to_dict(), indent=2))


if __name__ == "__main__":
    main()
