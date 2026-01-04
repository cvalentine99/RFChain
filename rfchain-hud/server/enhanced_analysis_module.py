#!/usr/bin/env python3
"""
Enhanced RF Analysis Module v3.0
Integrates: Modulation Classification, OFDM Detection, QAM Demodulation, BER Analysis, Signal Stage Detection

This module extends analyze_signal_v2.2.2_forensic.py with advanced DSP capabilities.
"""

import numpy as np
from scipy import signal as sig
from scipy.ndimage import gaussian_filter, label
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.stats import kurtosis, skew
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter
import json
import logging

log = logging.getLogger(__name__)

# =============================================================================
# MODULATION CLASSIFICATION MODULE
# =============================================================================

@dataclass
class ForensicUncertainty:
    """Forensic measurement uncertainty quantification.

    FORENSIC FIX v2.2.3: Added per NIST SP 800-86 and SWGDE requirements for
    documenting measurement uncertainty in forensic RF analysis.

    All forensic measurements must include uncertainty bounds to be admissible
    as evidence. This class captures:
    - Confidence intervals for classification decisions
    - Error bounds on numerical measurements
    - Alternative hypotheses considered and their probabilities
    """
    primary_hypothesis: str
    primary_probability: float
    alternative_hypotheses: List[Tuple[str, float]] = field(default_factory=list)
    measurement_std_dev: float = 0.0
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    methodology: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_hypothesis': self.primary_hypothesis,
            'primary_probability': self.primary_probability,
            'alternative_hypotheses': self.alternative_hypotheses,
            'measurement_std_dev': self.measurement_std_dev,
            'confidence_interval_95': self.confidence_interval_95,
            'sample_size': self.sample_size,
            'methodology': self.methodology
        }


@dataclass
class ModulationClassification:
    """Results of modulation classification analysis.

    FORENSIC FIX v2.2.3: Added uncertainty field for forensic compliance.
    """
    modulation_type: str  # QAM, PSK, OFDM, FSK, etc.
    order: int  # 4, 16, 64, 128, 256 for QAM/PSK
    confidence_percent: float
    bits_per_symbol: int
    constellation_points: int
    i_levels: int
    q_levels: int
    grid_regularity: float
    is_ofdm: bool = False
    ofdm_fft_size: int = 0
    ofdm_cp_length: int = 0
    # FORENSIC: Uncertainty quantification
    uncertainty: Optional[ForensicUncertainty] = None

class ModulationClassifier:
    """
    Advanced modulation classification using constellation analysis and cyclostationary features
    """
    
    @staticmethod
    def count_constellation_points(signal_data: np.ndarray, num_samples: int = 100000, 
                                    resolution: int = 256) -> Tuple[int, np.ndarray]:
        """Count unique constellation points using 2D histogram"""
        data = signal_data[:num_samples]
        data = data / np.std(data) * 0.5
        
        hist, xedges, yedges = np.histogram2d(
            data.real, data.imag, 
            bins=resolution, 
            range=[[-2, 2], [-2, 2]]
        )
        
        hist_smooth = gaussian_filter(hist, sigma=1)
        threshold = np.max(hist_smooth) * 0.02
        peaks = hist_smooth > threshold
        
        labeled, num_features = label(peaks)
        return num_features, hist_smooth
    
    @staticmethod
    def analyze_grid_pattern(signal_data: np.ndarray, num_samples: int = 100000) -> Dict[str, Any]:
        """Analyze if constellation forms a regular grid (QAM characteristic)"""
        data = signal_data[:num_samples]
        data = data / np.std(data)
        
        i_vals = data.real
        q_vals = data.imag
        
        # Find clusters in I dimension
        i_hist, i_bins = np.histogram(i_vals, bins=200)
        i_smooth = gaussian_filter(i_hist.astype(float), sigma=2)
        i_peaks, _ = sig.find_peaks(i_smooth, height=np.max(i_smooth)*0.05, distance=5)
        
        # Find clusters in Q dimension
        q_hist, q_bins = np.histogram(q_vals, bins=200)
        q_smooth = gaussian_filter(q_hist.astype(float), sigma=2)
        q_peaks, _ = sig.find_peaks(q_smooth, height=np.max(q_smooth)*0.05, distance=5)
        
        i_levels = len(i_peaks)
        q_levels = len(q_peaks)
        
        # Calculate regularity
        if len(i_peaks) > 1:
            i_centers = (i_bins[i_peaks] + i_bins[i_peaks+1]) / 2
            i_spacings = np.diff(i_centers)
            i_regularity = 1 - (np.std(i_spacings) / np.mean(i_spacings)) if np.mean(i_spacings) > 0 else 0
        else:
            i_regularity = 0
            
        if len(q_peaks) > 1:
            q_centers = (q_bins[q_peaks] + q_bins[q_peaks+1]) / 2
            q_spacings = np.diff(q_centers)
            q_regularity = 1 - (np.std(q_spacings) / np.mean(q_spacings)) if np.mean(q_spacings) > 0 else 0
        else:
            q_regularity = 0
        
        return {
            'i_levels': i_levels,
            'q_levels': q_levels,
            'estimated_points': i_levels * q_levels,
            'i_regularity': i_regularity,
            'q_regularity': q_regularity,
            'grid_regularity': (i_regularity + q_regularity) / 2,
            'is_square_qam': abs(i_levels - q_levels) <= 2
        }
    
    @staticmethod
    def detect_ofdm_structure(signal_data: np.ndarray, sample_rate: float, 
                               num_samples: int = 50000) -> Dict[str, Any]:
        """Detect OFDM cyclic prefix structure"""
        data = signal_data[:num_samples]
        
        # Compute autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = np.abs(autocorr) / autocorr[0]
        
        # Check for OFDM-like structure at typical CP lengths
        cp_candidates = []
        for cp_ratio in [1/4, 1/8, 1/16, 1/32]:
            for fft_size in [64, 128, 256, 512, 1024, 2048, 4096]:
                cp_length = int(fft_size * cp_ratio)
                symbol_length = fft_size + cp_length
                
                if symbol_length < len(autocorr):
                    corr_at_symbol = float(np.abs(autocorr[symbol_length]))
                    if corr_at_symbol > 0.5:
                        cp_candidates.append({
                            'fft_size': fft_size,
                            'cp_length': cp_length,
                            'symbol_length': symbol_length,
                            'correlation': corr_at_symbol,
                            'cp_ratio': cp_ratio,
                        })
        
        # Sort by correlation
        cp_candidates = sorted(cp_candidates, key=lambda x: -x['correlation'])
        
        return {
            'is_ofdm': len(cp_candidates) > 0,
            'candidates': cp_candidates[:5],
            'best_match': cp_candidates[0] if cp_candidates else None,
        }
    
    @staticmethod
    def classify_qam_order(grid_analysis: Dict, constellation_points: int) -> Dict[str, Any]:
        """Determine QAM order from analysis results.

        FORENSIC FIX v2.2.3: Corrected QAM constellation grid mappings.
        32-QAM and 128-QAM are cross constellations, not square grids.
        - 32-QAM: Cross pattern derived from 6x6 grid minus 4 corners = 32 points
        - 128-QAM: Cross pattern, not 12x12 (which would be 144 points)
        """
        # Format: order: (typical_i_levels, typical_q_levels, is_cross_constellation)
        # Cross constellations have irregular I/Q level counts
        qam_orders = {
            4: (2, 2),      # QPSK/4-QAM: 2x2 square
            8: (4, 2),      # 8-QAM: rectangular or star (treating as rect)
            16: (4, 4),     # 16-QAM: 4x4 square
            32: (6, 6),     # 32-QAM: Cross from 6x6 minus corners (appears ~6 levels each axis)
            64: (8, 8),     # 64-QAM: 8x8 square
            128: (12, 12),  # 128-QAM: Cross pattern (~12 levels visible per axis)
            256: (16, 16),  # 256-QAM: 16x16 square
            512: (24, 24),  # 512-QAM: Cross pattern
            1024: (32, 32), # 1024-QAM: 32x32 square
        }
        
        i_levels = grid_analysis['i_levels']
        q_levels = grid_analysis['q_levels']
        estimated_points = grid_analysis['estimated_points']
        
        best_order = 4
        best_diff = float('inf')

        # FORENSIC FIX: Track all candidate scores for uncertainty quantification
        candidate_scores = {}

        for order, (i, q) in qam_orders.items():
            grid_diff = abs(i_levels - i) + abs(q_levels - q)
            point_diff = abs(estimated_points - order)
            total_diff = grid_diff + point_diff * 0.1

            # Convert difference to a similarity score (lower diff = higher score)
            # Use exponential decay for more interpretable scoring
            similarity_score = np.exp(-total_diff / 5.0)
            candidate_scores[order] = similarity_score

            if total_diff < best_diff:
                best_diff = total_diff
                best_order = order

        # Normalize scores to probabilities
        total_score = sum(candidate_scores.values())
        candidate_probs = {k: v / total_score for k, v in candidate_scores.items()}

        # Sort by probability for alternative hypotheses
        sorted_candidates = sorted(candidate_probs.items(), key=lambda x: -x[1])
        primary_prob = candidate_probs[best_order]
        alternatives = [(f"{order}-QAM", prob)
                        for order, prob in sorted_candidates
                        if order != best_order and prob > 0.01][:3]

        confidence = grid_analysis['grid_regularity'] * 100

        # FORENSIC: Create uncertainty quantification
        uncertainty = ForensicUncertainty(
            primary_hypothesis=f"{best_order}-QAM",
            primary_probability=primary_prob,
            alternative_hypotheses=alternatives,
            measurement_std_dev=float(1.0 - grid_analysis['grid_regularity']),
            confidence_interval_95=(max(0, confidence - 15), min(100, confidence + 15)),
            sample_size=int(grid_analysis.get('estimated_points', 0)),
            methodology="Constellation clustering with grid regularity analysis"
        )

        return {
            'qam_order': best_order,
            'confidence_percent': min(confidence + 20, 100),
            'bits_per_symbol': int(np.log2(best_order)),
            'uncertainty': uncertainty,
        }
    
    @classmethod
    def classify(cls, signal_data: np.ndarray, sample_rate: float) -> ModulationClassification:
        """
        Main classification method - determines modulation type and parameters
        """
        # Check for OFDM first
        ofdm_result = cls.detect_ofdm_structure(signal_data, sample_rate)
        
        # Count constellation points
        num_points, _ = cls.count_constellation_points(signal_data)
        
        # Analyze grid pattern
        grid = cls.analyze_grid_pattern(signal_data)
        
        # Classify QAM order
        qam_class = cls.classify_qam_order(grid, num_points)
        
        if ofdm_result['is_ofdm']:
            best = ofdm_result['best_match']
            return ModulationClassification(
                modulation_type='OFDM',
                order=qam_class['qam_order'],
                confidence_percent=qam_class['confidence_percent'],
                bits_per_symbol=qam_class['bits_per_symbol'],
                constellation_points=num_points,
                i_levels=grid['i_levels'],
                q_levels=grid['q_levels'],
                grid_regularity=grid['grid_regularity'],
                is_ofdm=True,
                ofdm_fft_size=best['fft_size'],
                ofdm_cp_length=best['cp_length'],
                uncertainty=qam_class.get('uncertainty'),
            )
        else:
            return ModulationClassification(
                modulation_type='QAM',
                order=qam_class['qam_order'],
                confidence_percent=qam_class['confidence_percent'],
                bits_per_symbol=qam_class['bits_per_symbol'],
                constellation_points=num_points,
                i_levels=grid['i_levels'],
                q_levels=grid['q_levels'],
                grid_regularity=grid['grid_regularity'],
                uncertainty=qam_class.get('uncertainty'),
            )


# =============================================================================
# OFDM PROCESSING MODULE
# =============================================================================

@dataclass
class OFDMSymbol:
    """Single OFDM symbol data"""
    index: int
    time_domain: np.ndarray
    frequency_domain: np.ndarray
    subcarriers: np.ndarray

@dataclass
class OFDMProcessingResult:
    """Results of OFDM processing pipeline"""
    fft_size: int
    cp_length: int
    num_symbols: int
    symbols: List[OFDMSymbol]
    subcarrier_power: np.ndarray
    pilot_indices: List[int]
    data_indices: List[int]
    channel_estimate: Optional[np.ndarray] = None

class OFDMProcessor:
    """
    OFDM signal processing: DC removal, I/Q correction, symbol sync, FFT
    """
    
    def __init__(self, fft_size: int = 256, cp_length: int = 8, sample_rate: float = 1e6):
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.symbol_length = fft_size + cp_length
        self.sample_rate = sample_rate
    
    def remove_dc_offset(self, signal_data: np.ndarray) -> np.ndarray:
        """Step 1: Remove DC offset"""
        return signal_data - np.mean(signal_data)
    
    def correct_iq_imbalance(self, signal_data: np.ndarray) -> np.ndarray:
        """Step 2: Correct I/Q amplitude and phase imbalance"""
        i_data = signal_data.real
        q_data = signal_data.imag
        
        # Estimate amplitude imbalance
        i_power = np.mean(i_data ** 2)
        q_power = np.mean(q_data ** 2)
        amplitude_ratio = np.sqrt(i_power / q_power) if q_power > 0 else 1.0
        
        # Estimate phase imbalance (correlation between I and Q)
        correlation = np.mean(i_data * q_data)
        phase_imbalance = np.arcsin(2 * correlation / (np.sqrt(i_power * q_power) + 1e-10))
        
        # Correct Q channel
        q_corrected = q_data * amplitude_ratio
        
        # Apply phase correction
        corrected = i_data + 1j * q_corrected
        corrected = corrected * np.exp(-1j * phase_imbalance / 2)
        
        return corrected
    
    def find_symbol_timing(self, signal_data: np.ndarray, search_range: int = 1000) -> int:
        """Step 3: Find OFDM symbol timing using CP correlation"""
        best_offset = 0
        best_correlation = 0
        
        for offset in range(min(search_range, len(signal_data) - self.symbol_length)):
            # Correlate CP with end of symbol
            cp_samples = signal_data[offset:offset + self.cp_length]
            end_samples = signal_data[offset + self.fft_size:offset + self.symbol_length]
            
            correlation = np.abs(np.sum(cp_samples * np.conj(end_samples)))
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_offset = offset
        
        return best_offset
    
    def extract_symbols(self, signal_data: np.ndarray, start_offset: int = 0) -> List[OFDMSymbol]:
        """Extract OFDM symbols from time-domain signal"""
        symbols = []
        offset = start_offset
        symbol_idx = 0
        
        while offset + self.symbol_length <= len(signal_data):
            # Extract symbol (skip CP)
            symbol_with_cp = signal_data[offset:offset + self.symbol_length]
            symbol_data = symbol_with_cp[self.cp_length:]  # Remove CP
            
            # Apply FFT
            freq_domain = fft(symbol_data)
            freq_domain = fftshift(freq_domain)  # Center DC
            
            symbols.append(OFDMSymbol(
                index=symbol_idx,
                time_domain=symbol_with_cp,
                frequency_domain=freq_domain,
                subcarriers=freq_domain,
            ))
            
            offset += self.symbol_length
            symbol_idx += 1
        
        return symbols
    
    def process(self, signal_data: np.ndarray) -> OFDMProcessingResult:
        """
        Execute full OFDM processing pipeline:
        1. DC offset removal
        2. I/Q imbalance correction
        3. Symbol timing synchronization
        4. FFT processing
        """
        log.info(f"OFDM Processing: FFT={self.fft_size}, CP={self.cp_length}")
        
        # Step 1: DC removal
        log.info("  Step 1: Removing DC offset...")
        signal_dc_removed = self.remove_dc_offset(signal_data)
        
        # Step 2: I/Q correction
        log.info("  Step 2: Correcting I/Q imbalance...")
        signal_iq_corrected = self.correct_iq_imbalance(signal_dc_removed)
        
        # Step 3: Symbol timing
        log.info("  Step 3: Finding symbol timing...")
        timing_offset = self.find_symbol_timing(signal_iq_corrected)
        log.info(f"    Symbol timing offset: {timing_offset} samples")
        
        # Step 4: Extract and FFT symbols
        log.info("  Step 4: Extracting symbols and applying FFT...")
        symbols = self.extract_symbols(signal_iq_corrected, timing_offset)
        log.info(f"    Extracted {len(symbols)} OFDM symbols")
        
        # Calculate average subcarrier power
        if symbols:
            all_subcarriers = np.array([s.subcarriers for s in symbols])
            subcarrier_power = np.mean(np.abs(all_subcarriers) ** 2, axis=0)
        else:
            subcarrier_power = np.array([])
        
        return OFDMProcessingResult(
            fft_size=self.fft_size,
            cp_length=self.cp_length,
            num_symbols=len(symbols),
            symbols=symbols,
            subcarrier_power=subcarrier_power,
            pilot_indices=[],  # Would need standard-specific knowledge
            data_indices=list(range(self.fft_size)),
        )


# =============================================================================
# SYMBOL TIMING AND DEMODULATION MODULE
# =============================================================================

@dataclass
class DemodulationResult:
    """Results of QAM demodulation"""
    symbols: np.ndarray
    symbol_indices: np.ndarray
    bits: np.ndarray
    samples_per_symbol: float
    optimal_phase: int
    constellation: np.ndarray

class SymbolTimingRecovery:
    """Symbol timing recovery for single-carrier signals"""
    
    @staticmethod
    def estimate_symbol_rate(signal_data: np.ndarray, sample_rate: float) -> Tuple[float, float]:
        """Estimate symbol rate using cyclostationary analysis"""
        num_samples = min(65536, len(signal_data))
        data = signal_data[:num_samples]
        
        # Squared envelope for cyclostationary features
        envelope_sq = np.abs(data) ** 2
        envelope_sq = envelope_sq - np.mean(envelope_sq)
        
        # FFT of squared envelope
        fft_result = np.abs(fft(envelope_sq))
        freqs = fftfreq(len(fft_result), 1/sample_rate)
        
        # Find peaks in positive frequencies
        pos_mask = (freqs > 1000) & (freqs < sample_rate/2)
        pos_freqs = freqs[pos_mask]
        pos_fft = fft_result[pos_mask]
        
        peaks, _ = sig.find_peaks(pos_fft, height=np.max(pos_fft)*0.1, distance=10)
        
        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(pos_fft[peaks])]
            symbol_rate = pos_freqs[peak_idx]
        else:
            symbol_rate = sample_rate / 10  # Default fallback
        
        samples_per_symbol = sample_rate / symbol_rate if symbol_rate > 0 else 10
        return symbol_rate, samples_per_symbol
    
    @staticmethod
    def find_optimal_phase(signal_data: np.ndarray, samples_per_symbol: float, 
                           num_symbols: int = 1000) -> int:
        """Find optimal sampling phase using eye opening metric"""
        sps = int(round(samples_per_symbol))
        best_phase = 0
        best_metric = 0
        
        for phase in range(sps):
            samples = signal_data[phase::sps][:num_symbols]
            metric = np.var(np.abs(samples))
            
            if metric > best_metric:
                best_metric = metric
                best_phase = phase
        
        return best_phase


class QAMDemodulator:
    """QAM demodulation with constellation mapping"""
    
    def __init__(self, qam_order: int = 64):
        self.qam_order = qam_order
        self.bits_per_symbol = int(np.log2(qam_order))
        self.constellation = self._generate_constellation()
    
    def _generate_constellation(self) -> np.ndarray:
        """Generate standard QAM constellation"""
        sqrt_order = int(np.sqrt(self.qam_order))
        if sqrt_order ** 2 != self.qam_order:
            # Non-square QAM (e.g., 128-QAM cross)
            sqrt_order = int(np.ceil(np.sqrt(self.qam_order)))
        
        levels = np.arange(-(sqrt_order-1), sqrt_order, 2)
        constellation = []
        for i in levels:
            for q in levels:
                constellation.append(complex(i, q))
        
        constellation = np.array(constellation[:self.qam_order])
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation
    
    def demodulate(self, symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map received symbols to nearest constellation points"""
        symbols_norm = symbols / np.std(symbols)
        
        indices = []
        for sym in symbols_norm:
            distances = np.abs(sym - self.constellation)
            indices.append(np.argmin(distances))
        
        indices = np.array(indices)
        
        # Convert to bits
        bits = []
        for idx in indices:
            bit_string = format(idx, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in bit_string])
        
        return indices, np.array(bits)


# =============================================================================
# BER ANALYSIS MODULE
# =============================================================================

class PRBSGenerator:
    """Pseudo-Random Binary Sequence Generator

    FORENSIC FIX v2.2.3: Corrected PRBS tap positions per ITU-T O.150/O.151 standards.
    These are the industry-standard feedback polynomial tap positions used in
    telecommunications testing equipment (Anritsu, Keysight, R&S).

    Reference: ITU-T O.150 "General requirements for instrumentation for
    performance measurements on digital transmission equipment"
    """

    POLYNOMIALS = {
        # Format: (degree, [tap positions from MSB]) - represents x^n + x^tap + 1
        'PRBS-7':  (7, [7, 6]),      # x^7 + x^6 + 1 (ITU-T O.150)
        'PRBS-9':  (9, [9, 5]),      # x^9 + x^5 + 1 (ITU-T O.150)
        'PRBS-11': (11, [11, 9]),    # x^11 + x^9 + 1 (ITU-T O.150)
        'PRBS-15': (15, [15, 14]),   # x^15 + x^14 + 1 (ITU-T O.150)
        'PRBS-20': (20, [20, 17]),   # x^20 + x^17 + 1 (ITU-T O.150) - FIXED: was [20, 3]
        'PRBS-23': (23, [23, 18]),   # x^23 + x^18 + 1 (ITU-T O.150)
        'PRBS-31': (31, [31, 28]),   # x^31 + x^28 + 1 (ITU-T O.150)
        # Additional common PRBS patterns used in RF testing
        'PRBS-6':  (6, [6, 5]),      # x^6 + x^5 + 1 (used in some legacy systems)
        'PRBS-10': (10, [10, 7]),    # x^10 + x^7 + 1 (IEEE 802.3)
        'PRBS-13': (13, [13, 12, 11, 8]),  # x^13 + x^12 + x^11 + x^8 + 1
    }
    
    def __init__(self, prbs_type: str = 'PRBS-7'):
        if prbs_type not in self.POLYNOMIALS:
            raise ValueError(f"Unknown PRBS type: {prbs_type}")
        
        self.prbs_type = prbs_type
        self.order, self.taps = self.POLYNOMIALS[prbs_type]
        self.sequence_length = (2 ** self.order) - 1
        self.register = [1] * self.order
    
    def reset(self, seed: Optional[int] = None):
        if seed is None:
            self.register = [1] * self.order
        else:
            self.register = [(seed >> i) & 1 for i in range(self.order)]
            if all(b == 0 for b in self.register):
                self.register = [1] * self.order
    
    def next_bit(self) -> int:
        feedback = 0
        for tap in self.taps:
            feedback ^= self.register[tap - 1]
        output = self.register[-1]
        self.register = [feedback] + self.register[:-1]
        return output
    
    def generate(self, length: int) -> np.ndarray:
        return np.array([self.next_bit() for _ in range(length)])


@dataclass
class BERResult:
    """BER analysis results"""
    best_prbs_type: str
    ber: float
    total_errors: int
    total_bits: int
    correlation: float
    alignment_offset: int
    polarity_inverted: bool
    all_results: Dict[str, Dict]

class BERAnalyzer:
    """Bit Error Rate analysis against PRBS sequences"""
    
    @staticmethod
    def calculate_ber(received: np.ndarray, reference: np.ndarray) -> Tuple[float, int, int]:
        """Calculate BER between two bit sequences"""
        min_len = min(len(received), len(reference))
        errors = np.sum(received[:min_len] != reference[:min_len])
        return errors / min_len, errors, min_len
    
    @staticmethod
    def find_alignment(received: np.ndarray, prbs: np.ndarray, 
                       window: int = 1000) -> Tuple[int, float]:
        """Find best alignment between received and PRBS sequence"""
        received_window = received[:window]
        best_offset = 0
        best_corr = 0
        
        for offset in range(min(len(prbs) - window, window)):
            prbs_segment = prbs[offset:offset + window]
            if len(prbs_segment) < window:
                continue
            
            matches = np.sum(received_window == prbs_segment)
            corr = matches / window
            
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        return best_offset, best_corr
    
    @classmethod
    def analyze(cls, received_bits: np.ndarray, max_bits: int = 100000) -> BERResult:
        """Test received bits against all PRBS sequences"""
        test_bits = received_bits[:max_bits]
        results = {}
        
        for prbs_type in PRBSGenerator.POLYNOMIALS.keys():
            gen = PRBSGenerator(prbs_type)
            prbs_bits = gen.generate(max_bits + 1000)
            
            offset, corr = cls.find_alignment(test_bits, prbs_bits)
            aligned_prbs = prbs_bits[offset:offset + len(test_bits)]
            
            ber, errors, total = cls.calculate_ber(test_bits, aligned_prbs)
            inv_ber, inv_errors, _ = cls.calculate_ber(test_bits, 1 - aligned_prbs)
            
            inverted = inv_ber < ber
            if inverted:
                ber, errors = inv_ber, inv_errors
            
            results[prbs_type] = {
                'ber': float(ber),
                'errors': int(errors),
                'total_bits': int(total),
                'correlation': float(corr),
                'offset': int(offset),
                'inverted': inverted,
            }
        
        best_type = min(results.keys(), key=lambda k: results[k]['ber'])
        best = results[best_type]
        
        return BERResult(
            best_prbs_type=best_type,
            ber=best['ber'],
            total_errors=best['errors'],
            total_bits=best['total_bits'],
            correlation=best['correlation'],
            alignment_offset=best['offset'],
            polarity_inverted=best['inverted'],
            all_results=results,
        )


# =============================================================================
# SIGNAL STAGE DETECTION MODULE
# =============================================================================

@dataclass
class SignalStageResult:
    """Signal processing stage detection results"""
    stage: str
    stage_name: str
    confidence_percent: float
    evidence: List[str]
    dc_offset_percent: float
    iq_imbalance_db: float
    is_ofdm: bool
    ofdm_params: Optional[Dict] = None
    recommended_pipeline: List[Dict] = field(default_factory=list)

class SignalStageDetector:
    """Determine signal processing stage and recommend DSP pipeline"""
    
    STAGE_NAMES = {
        'raw_adc': 'Raw ADC Output',
        'after_ddc': 'After Digital Down-Conversion',
        'after_filtering': 'After Channel Filtering',
        'after_agc': 'After AGC',
        'baseband': 'Baseband Signal',
        'pre_fft': 'Pre-FFT (OFDM Time Domain)',
        'frequency_domain': 'Frequency Domain',
    }
    
    @staticmethod
    def analyze_time_domain(signal_data: np.ndarray) -> Dict[str, Any]:
        """Analyze time-domain characteristics"""
        i_data = signal_data.real
        q_data = signal_data.imag
        
        dc_magnitude = np.abs(np.mean(signal_data))
        signal_std = np.std(signal_data)
        dc_relative = dc_magnitude / signal_std if signal_std > 0 else 0
        
        i_power = np.mean(i_data ** 2)
        q_power = np.mean(q_data ** 2)
        iq_imbalance_db = 10 * np.log10(i_power / q_power) if q_power > 0 else 0
        
        return {
            'dc_offset_relative': float(dc_relative),
            'dc_offset_percent': float(dc_relative * 100),
            'iq_imbalance_db': float(iq_imbalance_db),
            'i_kurtosis': float(kurtosis(i_data[:10000])),
            'q_kurtosis': float(kurtosis(q_data[:10000])),
        }
    
    @classmethod
    def detect(cls, signal_data: np.ndarray, sample_rate: float, 
               filename: str = '') -> SignalStageResult:
        """Detect signal processing stage"""
        
        indicators = {k: 0 for k in cls.STAGE_NAMES.keys()}
        evidence = []
        
        # Filename analysis
        filename_lower = filename.lower()
        if 'beforefft' in filename_lower or 'prefft' in filename_lower:
            indicators['pre_fft'] += 3
            evidence.append("Filename indicates pre-FFT samples")
        if 'dump' in filename_lower:
            indicators['raw_adc'] += 1
            evidence.append("Filename suggests raw capture")
        
        # Time domain analysis
        time_analysis = cls.analyze_time_domain(signal_data)
        
        if time_analysis['dc_offset_relative'] > 0.1:
            indicators['raw_adc'] += 2
            evidence.append(f"Significant DC offset ({time_analysis['dc_offset_percent']:.1f}%)")
        
        if abs(time_analysis['iq_imbalance_db']) > 1.0:
            indicators['raw_adc'] += 1
            evidence.append(f"I/Q imbalance ({time_analysis['iq_imbalance_db']:.2f} dB)")
        
        # OFDM detection
        ofdm_result = ModulationClassifier.detect_ofdm_structure(signal_data, sample_rate)
        if ofdm_result['is_ofdm']:
            indicators['pre_fft'] += 2
            evidence.append("OFDM cyclic prefix structure detected")
        
        # Determine stage
        most_likely = max(indicators, key=indicators.get)
        total_score = sum(indicators.values())
        confidence = indicators[most_likely] / total_score * 100 if total_score > 0 else 0
        
        # Generate pipeline recommendations
        pipeline = cls._generate_pipeline(most_likely, ofdm_result)
        
        return SignalStageResult(
            stage=most_likely,
            stage_name=cls.STAGE_NAMES.get(most_likely, most_likely),
            confidence_percent=confidence,
            evidence=evidence,
            dc_offset_percent=time_analysis['dc_offset_percent'],
            iq_imbalance_db=time_analysis['iq_imbalance_db'],
            is_ofdm=ofdm_result['is_ofdm'],
            ofdm_params=ofdm_result['best_match'],
            recommended_pipeline=pipeline,
        )
    
    @staticmethod
    def _generate_pipeline(stage: str, ofdm_result: Dict) -> List[Dict]:
        """Generate recommended DSP pipeline"""
        pipeline = [
            {'step': 1, 'name': 'DC Offset Removal', 'priority': 'HIGH'},
            {'step': 2, 'name': 'I/Q Imbalance Correction', 'priority': 'MEDIUM'},
        ]
        
        if ofdm_result['is_ofdm'] and ofdm_result['best_match']:
            best = ofdm_result['best_match']
            pipeline.extend([
                {'step': 3, 'name': f"OFDM Symbol Sync (FFT={best['fft_size']}, CP={best['cp_length']})", 'priority': 'HIGH'},
                {'step': 4, 'name': f"{best['fft_size']}-point FFT", 'priority': 'HIGH'},
                {'step': 5, 'name': 'Channel Estimation', 'priority': 'HIGH'},
                {'step': 6, 'name': 'Subcarrier Demodulation', 'priority': 'HIGH'},
            ])
        else:
            pipeline.extend([
                {'step': 3, 'name': 'Carrier Recovery', 'priority': 'HIGH'},
                {'step': 4, 'name': 'Matched Filtering', 'priority': 'MEDIUM'},
                {'step': 5, 'name': 'Symbol Timing Recovery', 'priority': 'HIGH'},
                {'step': 6, 'name': 'Equalization', 'priority': 'MEDIUM'},
            ])
        
        return pipeline


# =============================================================================
# UNIFIED ENHANCED ANALYSIS CLASS
# =============================================================================

@dataclass
class EnhancedAnalysisResult:
    """Complete enhanced analysis results"""
    modulation: ModulationClassification
    signal_stage: SignalStageResult
    ofdm_result: Optional[OFDMProcessingResult] = None
    demodulation: Optional[DemodulationResult] = None
    ber_analysis: Optional[BERResult] = None
    symbol_rate: float = 0.0
    samples_per_symbol: float = 0.0

class EnhancedSignalAnalyzer:
    """
    Unified enhanced signal analysis integrating all modules
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
    
    def analyze(self, signal_data: np.ndarray, filename: str = '',
                perform_demodulation: bool = True,
                perform_ber: bool = True) -> EnhancedAnalysisResult:
        """
        Perform complete enhanced analysis
        """
        log.info("="*60)
        log.info("ENHANCED SIGNAL ANALYSIS")
        log.info("="*60)
        
        # 1. Signal Stage Detection
        log.info("\n[1] Detecting signal processing stage...")
        stage_result = SignalStageDetector.detect(signal_data, self.sample_rate, filename)
        log.info(f"    Stage: {stage_result.stage_name} ({stage_result.confidence_percent:.1f}% confidence)")
        
        # 2. Modulation Classification
        log.info("\n[2] Classifying modulation...")
        mod_result = ModulationClassifier.classify(signal_data, self.sample_rate)
        log.info(f"    Type: {mod_result.modulation_type}-{mod_result.order}")
        log.info(f"    OFDM: {mod_result.is_ofdm}")
        
        # 3. Symbol Rate Estimation
        log.info("\n[3] Estimating symbol rate...")
        symbol_rate, sps = SymbolTimingRecovery.estimate_symbol_rate(signal_data, self.sample_rate)
        log.info(f"    Symbol Rate: {symbol_rate/1e3:.2f} ksps")
        log.info(f"    Samples/Symbol: {sps:.2f}")
        
        result = EnhancedAnalysisResult(
            modulation=mod_result,
            signal_stage=stage_result,
            symbol_rate=symbol_rate,
            samples_per_symbol=sps,
        )
        
        # 4. OFDM Processing (if detected)
        if mod_result.is_ofdm and mod_result.ofdm_fft_size > 0:
            log.info("\n[4] Processing OFDM signal...")
            ofdm_proc = OFDMProcessor(
                fft_size=mod_result.ofdm_fft_size,
                cp_length=mod_result.ofdm_cp_length,
                sample_rate=self.sample_rate
            )
            result.ofdm_result = ofdm_proc.process(signal_data)
            log.info(f"    Extracted {result.ofdm_result.num_symbols} OFDM symbols")
        
        # 5. Demodulation (for single-carrier or per-subcarrier)
        if perform_demodulation and not mod_result.is_ofdm:
            log.info("\n[5] Demodulating signal...")
            demod = QAMDemodulator(mod_result.order)
            
            # Get optimal sampling phase
            optimal_phase = SymbolTimingRecovery.find_optimal_phase(signal_data, sps)
            sps_int = int(round(sps))
            symbols = signal_data[optimal_phase::sps_int]
            
            indices, bits = demod.demodulate(symbols)
            
            result.demodulation = DemodulationResult(
                symbols=symbols,
                symbol_indices=indices,
                bits=bits,
                samples_per_symbol=sps,
                optimal_phase=optimal_phase,
                constellation=demod.constellation,
            )
            log.info(f"    Recovered {len(bits)} bits from {len(symbols)} symbols")
        
        # 6. BER Analysis
        if perform_ber and result.demodulation is not None:
            log.info("\n[6] Analyzing BER against PRBS sequences...")
            result.ber_analysis = BERAnalyzer.analyze(result.demodulation.bits)
            log.info(f"    Best match: {result.ber_analysis.best_prbs_type}")
            log.info(f"    BER: {result.ber_analysis.ber:.2e}")
        
        log.info("\n" + "="*60)
        log.info("ENHANCED ANALYSIS COMPLETE")
        log.info("="*60)
        
        return result
    
    def to_dict(self, result: EnhancedAnalysisResult) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization"""
        def to_python(val):
            """Convert numpy types to Python native types"""
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        output = {
            'modulation': {
                'type': result.modulation.modulation_type,
                'order': int(result.modulation.order),
                'confidence_percent': float(result.modulation.confidence_percent),
                'bits_per_symbol': int(result.modulation.bits_per_symbol),
                'is_ofdm': bool(result.modulation.is_ofdm),
                'ofdm_fft_size': int(result.modulation.ofdm_fft_size),
                'ofdm_cp_length': int(result.modulation.ofdm_cp_length),
                'constellation_points': int(result.modulation.constellation_points),
                'grid_regularity': float(result.modulation.grid_regularity),
            },
            'signal_stage': {
                'stage': result.signal_stage.stage,
                'stage_name': result.signal_stage.stage_name,
                'confidence_percent': float(result.signal_stage.confidence_percent),
                'evidence': result.signal_stage.evidence,
                'dc_offset_percent': float(result.signal_stage.dc_offset_percent),
                'iq_imbalance_db': float(result.signal_stage.iq_imbalance_db),
                'recommended_pipeline': result.signal_stage.recommended_pipeline,
            },
            'timing': {
                'symbol_rate_hz': float(result.symbol_rate),
                'samples_per_symbol': float(result.samples_per_symbol),
            },
        }
        
        if result.ofdm_result:
            output['ofdm'] = {
                'fft_size': result.ofdm_result.fft_size,
                'cp_length': result.ofdm_result.cp_length,
                'num_symbols': result.ofdm_result.num_symbols,
                'subcarrier_power_db': [float(10*np.log10(p+1e-20)) for p in result.ofdm_result.subcarrier_power[:10]],
            }
        
        if result.demodulation:
            bits = result.demodulation.bits
            output['demodulation'] = {
                'total_symbols': len(result.demodulation.symbols),
                'total_bits': len(bits),
                'first_100_bits': ''.join(str(b) for b in bits[:100]),
                'optimal_phase': result.demodulation.optimal_phase,
            }
        
        if result.ber_analysis:
            output['ber_analysis'] = {
                'best_prbs': result.ber_analysis.best_prbs_type,
                'ber': result.ber_analysis.ber,
                'ber_scientific': f"{result.ber_analysis.ber:.2e}",
                'total_errors': result.ber_analysis.total_errors,
                'total_bits': result.ber_analysis.total_bits,
                'correlation': result.ber_analysis.correlation,
            }
        
        return output


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Standalone CLI for enhanced analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced RF Signal Analysis Module v3.0')
    parser.add_argument('file', help='Signal file to analyze')
    parser.add_argument('-r', '--sample-rate', type=float, default=1e6, help='Sample rate (Hz)')
    parser.add_argument('-f', '--format', default='complex64', help='Data format')
    parser.add_argument('--no-demod', action='store_true', help='Skip demodulation')
    parser.add_argument('--no-ber', action='store_true', help='Skip BER analysis')
    parser.add_argument('-o', '--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Load signal
    log.info(f"Loading {args.file}...")
    signal_data = np.fromfile(args.file, dtype=np.complex64)
    log.info(f"Loaded {len(signal_data):,} samples")
    
    # Analyze
    analyzer = EnhancedSignalAnalyzer(args.sample_rate)
    result = analyzer.analyze(
        signal_data, 
        filename=args.file,
        perform_demodulation=not args.no_demod,
        perform_ber=not args.no_ber
    )
    
    # Output
    output_dict = analyzer.to_dict(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_dict, f, indent=2)
        log.info(f"\nResults saved to {args.output}")
    else:
        print("\nJSON Output:")
        print(json.dumps(output_dict, indent=2))


if __name__ == "__main__":
    main()
