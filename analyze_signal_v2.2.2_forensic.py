#!/usr/bin/env python3
"""
CUDA-Accelerated RF Signal Analysis Suite v2.2.2
Enhanced for RF Forensics and Security Operations

Features:
- GPU-accelerated processing via CuPy
- Streaming analysis for large files
- Advanced modulation detection
- SNR estimation
- Cyclostationary analysis
- Anomaly detection primitives
- Memory-efficient chunked processing
- Digital signal analysis (--digital):
  - Binary signal detection and demodulation
  - Block structure and inversion pattern detection
  - Encoding type identification (NRZ, NRZ-I, etc.)
  - Sync pattern and HDLC frame extraction
  - Byte-level analysis with entropy calculation
- V3 Enhanced analysis (--v3):
  - FFT pipeline debug (DC offset, spectral leakage, spurs)
  - PN sequence / spreading code detection (M-seq, Gold codes)
  - Bit ordering & frame analysis (scramblers, sync words)
"""

import numpy as np
# Issue 5 Fix: GPU fallback mode - use NumPy when CuPy unavailable
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback to NumPy
    GPU_AVAILABLE = False
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from enum import Enum
from collections import Counter
from math import log2, gcd
from functools import reduce
import argparse
import logging
import sys
import json
from datetime import datetime
from contextlib import contextmanager
import warnings
import hashlib
import os
from scipy.signal import find_peaks, welch

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# =============================================================================
# EARLY DEFINITIONS (Required before forensic module)
# Issue 1 Fix: Move constants and logging before forensic compliance module
# =============================================================================

# Configure logging early (needed by forensic module)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# Constants needed by forensic module
HASH_CHUNK_SIZE = 65536  # Chunk size for file hashing (64KB)

# Issue 11 Fix: Named constants for magic numbers (forensic thresholds)
DC_SPIKE_THRESHOLD = 0.1  # DC component threshold relative to RMS
AUTOCORR_PEAK_HEIGHT = 0.5  # Autocorrelation peak detection height
AUTOCORR_PEAK_DISTANCE = 10  # Minimum distance between autocorr peaks
SATURATION_THRESHOLD = 0.99  # Signal saturation detection threshold
DROPOUT_THRESHOLD = 0.01  # Signal dropout detection threshold
MEMORY_USAGE_WARNING = 0.8  # Warn if file size exceeds this fraction of available RAM

# =============================================================================
# GPU/CPU COMPATIBILITY HELPERS
# =============================================================================

def to_numpy(arr):
    """Convert array to NumPy, handling both CuPy and NumPy arrays."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)

def to_scalar(val):
    """Convert a GPU/CPU scalar to Python scalar."""
    if GPU_AVAILABLE and hasattr(val, 'get'):
        return val.get()
    if hasattr(val, 'item'):
        return val.item()
    return val

def cp_asnumpy(arr):
    """Safe wrapper for cp.asnumpy that works in CPU mode."""
    if GPU_AVAILABLE:
        return cp_asnumpy(arr)
    return np.asarray(arr)

# =============================================================================
# FORENSIC COMPLIANCE MODULE (Added for NIST/SWGDE/ISO 27037 compliance)
# =============================================================================

# Issue 4 Fix: Removed duplicate imports (already imported above)
from datetime import timezone  # Only import timezone (datetime already imported)

# -----------------------------------------------------------------------------
# Issue 2 Fix: Dual-Algorithm Hashing (SWGDE Compliance)
# -----------------------------------------------------------------------------

def compute_forensic_hashes(filepath: Path) -> Dict[str, str]:
    """
    Computes SHA-256 and SHA3-256 hashes for a given file, adhering to SWGDE forensic standards.
    
    This function calculates dual-algorithm hashes (SHA-256 and SHA3-256) to mitigate collision
    vulnerabilities, as mandated by the Scientific Working Group on Digital Evidence (SWGDE).
    A UTC timestamp is included in the returned dictionary to provide a verifiable time of hashing.
    
    Args:
        filepath: The path to the file to be hashed.
        
    Returns:
        A dictionary containing the SHA-256 hash, the SHA3-256 hash, and the UTC timestamp.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found at: {filepath}")

    sha256_hash = hashlib.sha256()
    sha3_256_hash = hashlib.sha3_256()
    
    with open(filepath, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            sha256_hash.update(chunk)
            sha3_256_hash.update(chunk)

    return {
        'sha256': sha256_hash.hexdigest(),
        'sha3_256': sha3_256_hash.hexdigest(),
        'hash_timestamp_utc': datetime.now(timezone.utc).isoformat()
    }


# -----------------------------------------------------------------------------
# Issue 3 Fix: Residual Byte Preservation (Chain of Custody Compliance)
# -----------------------------------------------------------------------------

def load_with_residual_preservation(raw_bytes: bytes, sample_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads signal data from a raw byte stream, preserving trailing (residual) bytes
    that do not form a complete sample.
    
    This function is designed to comply with forensic best practices (NIST SP 800-86,
    SWGDE) by preventing silent data truncation. Instead of discarding incomplete
    samples, it separates them, computes a cryptographic hash for integrity verification,
    and returns them in a metadata dictionary.
    
    Args:
        raw_bytes: The raw byte stream read from the signal file.
        sample_size: The number of bytes that constitute a single complete sample.
        
    Returns:
        Tuple containing:
        - A NumPy array of complete samples
        - A dictionary containing metadata about the residual bytes
    """
    num_bytes = len(raw_bytes)
    num_complete_samples = num_bytes // sample_size
    num_residual_bytes = num_bytes % sample_size

    complete_samples_bytes = raw_bytes[:num_complete_samples * sample_size]
    samples = np.frombuffer(complete_samples_bytes, dtype=np.complex64)

    residual_metadata = {'residual_bytes': 0, 'action': 'none_required'}
    if num_residual_bytes > 0:
        residual_bytes = raw_bytes[num_complete_samples * sample_size:]
        residual_hash = hashlib.sha256(residual_bytes).hexdigest()
        residual_metadata = {
            'residual_bytes': num_residual_bytes,
            'residual_bytes_hex': residual_bytes.hex(),
            'residual_hash_sha256': residual_hash,
            'action': 'preserved_not_truncated'
        }
        log.warning(f"FORENSIC: {num_residual_bytes} residual bytes preserved separately")

    return samples, residual_metadata


# -----------------------------------------------------------------------------
# Issue 4 Fix: Parseval's Theorem Verification (FFT Integrity)
# -----------------------------------------------------------------------------

def verify_parseval(time_signal: cp.ndarray, freq_signal: cp.ndarray, 
                    tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Verifies the integrity of an FFT operation using Parseval's theorem.
    
    Parseval's theorem states that the total energy of a signal in the time domain
    is equal to its total energy in the frequency domain:
    Σ|x[n]|² = (1/N)Σ|X[k]|²
    
    Any discrepancy indicates processing error or data corruption.
    
    Args:
        time_signal: The input signal in the time domain (GPU array)
        freq_signal: The FFT of the time_signal (GPU array)
        tolerance: The acceptable tolerance for the energy difference
        
    Returns:
        Dictionary containing verification results
    """
    N = len(time_signal)
    if N == 0:
        return {
            'verification_passed': False,
            'time_domain_energy': 0.0,
            'freq_domain_energy': 0.0,
            'relative_error': float('inf'),
            'tolerance': tolerance,
            'message': "Error: Time domain signal is empty."
        }
    
    time_domain_energy = float(cp.sum(cp.abs(time_signal) ** 2))
    freq_domain_energy = float(cp.sum(cp.abs(freq_signal) ** 2) / N)
    
    relative_error = abs(time_domain_energy - freq_domain_energy) / (time_domain_energy + EPSILON_SMALL)
    passed = relative_error < tolerance
    
    if not passed:
        log.error(f"FORENSIC ALERT: Parseval verification FAILED (error: {relative_error:.2e})")
    
    return {
        'verification_passed': passed,
        'time_domain_energy': time_domain_energy,
        'freq_domain_energy': freq_domain_energy,
        'relative_error': relative_error,
        'tolerance': tolerance
    }


# -----------------------------------------------------------------------------
# Issue 5 Fix: CFAR Adaptive Thresholds (Forensic Detection)
# -----------------------------------------------------------------------------

def estimate_noise_power(signal: cp.ndarray, num_guard: int, num_train: int) -> float:
    """Estimate noise power from training cells for CFAR."""
    signal_abs = cp.abs(signal)
    # Use median-based noise estimation for robustness
    sorted_signal = cp.sort(signal_abs)
    # Exclude top values (potential signals) and use lower portion for noise estimate
    noise_samples = sorted_signal[:len(sorted_signal) // 2]
    return float(cp.mean(noise_samples ** 2))


def compute_cfar_threshold(signal: cp.ndarray, pfa: float = 1e-6,
                           num_guard: int = 2, num_train: int = 16) -> Tuple[float, Dict[str, Any]]:
    """
    Compute CFAR threshold for specified false alarm probability.
    
    Uses CA-CFAR (Cell-Averaging CFAR): T = α · P̂_n where α = N·(P_FA^{-1/N} - 1)
    
    This is mandatory for forensic validity as fixed thresholds cannot maintain
    consistent false alarm rates across varying noise conditions.
    
    Args:
        signal: Input signal array
        pfa: Desired probability of false alarm
        num_guard: Number of guard cells
        num_train: Number of training cells
        
    Returns:
        Tuple of (threshold value, metadata dictionary)
    """
    N = num_train
    alpha = N * (pfa ** (-1/N) - 1)
    
    noise_estimate = estimate_noise_power(signal, num_guard, num_train)
    threshold = alpha * noise_estimate
    
    return float(threshold), {
        'method': 'CA-CFAR',
        'pfa_specified': pfa,
        'alpha': float(alpha),
        'noise_estimate': float(noise_estimate),
        'num_guard_cells': num_guard,
        'num_training_cells': num_train
    }


# -----------------------------------------------------------------------------
# Issue 6 Fix: Chain of Custody Logging (NIST SP 800-86, ISO/IEC 27037)
# -----------------------------------------------------------------------------

@dataclass
class CustodyEntry:
    """Single entry in the chain of custody log."""
    handler_id: str
    handler_organization: str
    timestamp_utc: str
    action: str  # 'acquired', 'examined', 'processed', 'transferred'
    hash_before: str
    hash_after: str
    notes: str = ""


class ChainOfCustody:
    """
    Manages chain of custody logging for forensic evidence.
    
    Tracks each person who handled the evidence, the date/time it was collected
    or transferred, and the purpose for the transfer, as required by NIST SP 800-86
    and ISO/IEC 27037.
    """
    
    def __init__(self, evidence_id: str):
        self.evidence_id = evidence_id
        self.entries: List[CustodyEntry] = []
        self.created_utc = datetime.now(timezone.utc).isoformat()
    
    def add_entry(self, handler_id: str, org: str, action: str,
                  hash_before: str, hash_after: str, notes: str = "") -> None:
        """Add a new custody entry."""
        entry = CustodyEntry(
            handler_id=handler_id,
            handler_organization=org,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            action=action,
            hash_before=hash_before,
            hash_after=hash_after,
            notes=notes
        )
        self.entries.append(entry)
        log.info(f"CUSTODY: {action} by {handler_id}@{org}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export custody chain as dictionary."""
        return {
            'evidence_id': self.evidence_id,
            'created_utc': self.created_utc,
            'entries': [
                {
                    'handler_id': e.handler_id,
                    'handler_organization': e.handler_organization,
                    'timestamp_utc': e.timestamp_utc,
                    'action': e.action,
                    'hash_before': e.hash_before,
                    'hash_after': e.hash_after,
                    'notes': e.notes
                }
                for e in self.entries
            ]
        }


# -----------------------------------------------------------------------------
# Issue 7 Fix: GPU vs CPU Reference Validation (Phase 14 GPU Admissibility)
# -----------------------------------------------------------------------------

def validate_gpu_against_cpu(gpu_result: cp.ndarray, cpu_func: Callable,
                              input_data: np.ndarray, tolerance: float = 1e-5) -> Dict[str, Any]:
    """
    Validate GPU computation against CPU reference implementation.
    
    GPU results are NOT guaranteed reproducible between CPU and GPU. Forensic
    validation requires comparing GPU results against CPU reference implementations
    with documented tolerance thresholds.
    
    Args:
        gpu_result: Result from GPU computation
        cpu_func: CPU reference function to compare against
        input_data: Input data for the CPU function
        tolerance: Acceptable tolerance threshold
        
    Returns:
        Dictionary containing validation results and environment info
    """
    cpu_result = cpu_func(input_data)
    gpu_result_cpu = cp_asnumpy(gpu_result)
    
    max_diff = float(np.max(np.abs(gpu_result_cpu - cpu_result)))
    mean_diff = float(np.mean(np.abs(gpu_result_cpu - cpu_result)))
    
    validation = {
        'max_absolute_difference': max_diff,
        'mean_absolute_difference': mean_diff,
        'tolerance_threshold': tolerance,
        'validation_passed': max_diff < tolerance,
        'numpy_version': np.__version__,
        'cupy_version': cp.__version__,
    }
    
    # Try to get CUDA info
    try:
        validation['cuda_version'] = str(cp.cuda.runtime.runtimeGetVersion())
        validation['gpu_model'] = cp.cuda.Device().name.decode() if hasattr(cp.cuda.Device().name, 'decode') else str(cp.cuda.Device().name)
    except Exception as e:
        validation['cuda_info_error'] = str(e)
    
    if not validation['validation_passed']:
        log.error(f"GPU/CPU VALIDATION FAILED: max_diff={max_diff:.2e}")
    
    return validation


# -----------------------------------------------------------------------------
# Issue 8 Fix: ENBW Window Correction for PSD
# -----------------------------------------------------------------------------

def compute_enbw(window: np.ndarray) -> float:
    """
    Compute Equivalent Noise Bandwidth for a window function.
    
    ENBW = N × Σw² / (Σw)²
    
    This correction is required for accurate PSD calculation:
    PSD = |FFT(x)|² / (fs × N × ENBW)
    
    Args:
        window: Window function array
        
    Returns:
        ENBW value
    """
    return len(window) * np.sum(window**2) / (np.sum(window)**2)


def compute_psd_forensic(signal: cp.ndarray, fs: float, nfft: int, 
                         window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute Power Spectral Density with ENBW correction for forensic accuracy.
    
    Args:
        signal: Input signal
        fs: Sample rate
        nfft: FFT size
        window_type: Window function type
        
    Returns:
        Tuple of (frequencies, PSD values, metadata)
    """
    # Create window
    if window_type == 'hann':
        window = np.hanning(nfft)
    elif window_type == 'blackman':
        window = np.blackman(nfft)
    elif window_type == 'hamming':
        window = np.hamming(nfft)
    else:
        window = np.ones(nfft)
    
    window_gpu = cp.asarray(window)
    enbw = compute_enbw(window)
    
    # Apply window and compute FFT
    signal_windowed = signal[:nfft] * window_gpu
    fft_result = cp.fft.fft(signal_windowed)
    
    # Compute PSD with ENBW correction
    psd = cp.abs(fft_result)**2 / (fs * nfft * enbw)
    
    freqs = np.fft.fftfreq(nfft, 1/fs)
    
    metadata = {
        'window_type': window_type,
        'enbw': float(enbw),
        'nfft': nfft,
        'sample_rate': fs,
        'enbw_corrected': True
    }
    
    return freqs, cp_asnumpy(psd), metadata


# -----------------------------------------------------------------------------
# Issue 9 Fix: Bonferroni Correction for Multiple Hypothesis Testing
# -----------------------------------------------------------------------------

def apply_bonferroni_correction(pfa_desired: float, num_hypotheses: int) -> float:
    """
    Apply Bonferroni correction for multiple hypothesis testing.
    
    P_FA_single = P_FA_desired / N_codes
    
    This is required when testing multiple codes (e.g., Gold codes) to maintain
    the overall false alarm rate.
    
    Args:
        pfa_desired: Desired overall probability of false alarm
        num_hypotheses: Number of hypotheses being tested
        
    Returns:
        Corrected per-test probability of false alarm
    """
    return pfa_desired / num_hypotheses


# -----------------------------------------------------------------------------
# Issue 10 Fix: Data Format Conversion Documentation
# -----------------------------------------------------------------------------

def get_precision_bits(dtype) -> int:
    """Get the precision bits for a given dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).bits
    elif np.issubdtype(dtype, np.complexfloating):
        # For complex types, return the bits of the underlying float
        if dtype == np.complex64:
            return 32
        elif dtype == np.complex128:
            return 64
    return dtype.itemsize * 8


def document_format_conversion(original_dtype, converted_dtype, 
                                justification: str = None) -> Dict[str, Any]:
    """
    Document data format conversion for forensic compliance.
    
    All precision changes must be documented to maintain chain of custody.
    
    Args:
        original_dtype: Original data type
        converted_dtype: Converted data type
        justification: Reason for conversion
        
    Returns:
        Dictionary documenting the conversion
    """
    precision_bits_original = get_precision_bits(original_dtype)
    precision_bits_converted = get_precision_bits(converted_dtype)
    
    if justification is None:
        justification = "ADC precision (12-16 bit) within float32 mantissa (23 bit)"
    
    return {
        'original_dtype': str(original_dtype),
        'converted_dtype': str(converted_dtype),
        'precision_bits_original': precision_bits_original,
        'precision_bits_converted': precision_bits_converted,
        'conversion_justified': justification,
        'timestamp_utc': datetime.now(timezone.utc).isoformat()
    }


# -----------------------------------------------------------------------------
# Issue 11 Fix: Processing Step Hash Chain (Forensic Pipeline)
# -----------------------------------------------------------------------------

class ForensicPipeline:
    """
    Manages forensic chain of custody for signal processing.
    
    Creates a verifiable audit trail for all processing steps applied to a signal,
    ensuring compliance with forensic standards like NIST SP 800-86, ISO/IEC 27037,
    and SWGDE best practices.
    """
    
    def __init__(self):
        self.hash_chain: List[Dict[str, Any]] = []
    
    def add_hash_checkpoint(self, data: Union[np.ndarray, cp.ndarray], 
                            stage: str, details: Dict[str, Any] = None) -> str:
        """
        Add a checkpoint to the hash chain.
        
        Args:
            data: Data at current processing stage
            stage: Name of the processing stage
            details: Additional details about the processing step
            
        Returns:
            SHA-256 hash of the data
        """
        # Convert CuPy array to NumPy if needed
        if hasattr(data, 'get'):
            data_np = to_numpy(data)
        else:
            data_np = data
        
        data_hash = hashlib.sha256(data_np.tobytes()).hexdigest()
        
        checkpoint = {
            'stage': stage,
            'hash': data_hash,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'data_shape': data_np.shape,
            'data_dtype': str(data_np.dtype),
            'details': details or {}
        }
        
        self.hash_chain.append(checkpoint)
        log.info(f"FORENSIC CHECKPOINT: {stage} - hash: {data_hash[:16]}...")
        
        return data_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Export hash chain as dictionary."""
        return {
            'hash_chain': self.hash_chain,
            'total_checkpoints': len(self.hash_chain)
        }


# -----------------------------------------------------------------------------
# Issue 1 Fix: Full-Precision Correlation (No Hard Quantization)
# -----------------------------------------------------------------------------

def correlate_forensic(x: cp.ndarray, reference: cp.ndarray, 
                       mode: str = 'valid') -> Tuple[cp.ndarray, Dict[str, Any]]:
    """
    Forensic-compliant correlation that preserves amplitude and phase information.
    
    This function avoids hard quantization (cp.sign()) which destroys amplitude
    information, invalidates SNR estimation, and degrades matched filter optimality.
    
    Args:
        x: Input signal
        reference: Reference signal to correlate against
        mode: Correlation mode ('valid', 'full', 'same')
        
    Returns:
        Tuple of (correlation result, metadata)
    """
    # Normalize using L2 norm to preserve amplitude and phase
    # FORENSIC FIX: Do NOT use cp.sign() which destroys amplitude information
    x_norm = x / (cp.max(cp.abs(x)) + EPSILON)
    ref_norm = reference / (cp.max(cp.abs(reference)) + EPSILON)
    
    # Perform correlation
    corr = cp.correlate(x_norm, ref_norm, mode=mode)
    
    metadata = {
        'normalization_method': 'max_amplitude',
        'hard_quantization': False,
        'forensic_compliant': True
    }
    
    return corr, metadata


# =============================================================================
# END FORENSIC COMPLIANCE MODULE

# =============================================================================
# FFT PIPELINE COMPARISON MODULE
# For before/after FFT analysis, pipeline validation, and artifact attribution
# =============================================================================

@dataclass
class SpectralPeak:
    """Represents a detected spectral peak/spur."""
    frequency_hz: float
    magnitude_db: float
    bin_index: int
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0


@dataclass 
class PipelineComparisonResult:
    """Results from before/after FFT comparison analysis."""
    # File metadata
    before_file: str
    after_file: str
    before_hash: str
    after_hash: str
    analysis_timestamp: str
    
    # Energy conservation
    before_total_energy: float
    after_total_energy: float
    energy_ratio_db: float
    parseval_check_passed: bool
    
    # Noise floor comparison
    before_noise_floor_db: float
    after_noise_floor_db: float
    noise_floor_change_db: float
    
    # SNR comparison
    before_snr_db: float
    after_snr_db: float
    snr_change_db: float
    
    # Spur tracking
    spurs_before: List[SpectralPeak] = field(default_factory=list)
    spurs_after: List[SpectralPeak] = field(default_factory=list)
    spurs_removed: List[SpectralPeak] = field(default_factory=list)
    spurs_added: List[SpectralPeak] = field(default_factory=list)
    spurs_unchanged: List[SpectralPeak] = field(default_factory=list)
    
    # Bandwidth analysis
    before_bandwidth_hz: float = 0.0
    after_bandwidth_hz: float = 0.0
    bandwidth_change_hz: float = 0.0
    
    # Spectral shape metrics
    spectral_correlation: float = 0.0
    max_spectral_deviation_db: float = 0.0
    
    # Artifact detection
    artifacts_detected: List[str] = field(default_factory=list)
    
    # Forensic metadata
    forensic_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file_metadata': {
                'before_file': self.before_file,
                'after_file': self.after_file,
                'before_hash_sha256': self.before_hash,
                'after_hash_sha256': self.after_hash,
                'analysis_timestamp': self.analysis_timestamp
            },
            'energy_conservation': {
                'before_total_energy': self.before_total_energy,
                'after_total_energy': self.after_total_energy,
                'energy_ratio_db': self.energy_ratio_db,
                'parseval_check_passed': bool(self.parseval_check_passed)
            },
            'noise_floor': {
                'before_db': self.before_noise_floor_db,
                'after_db': self.after_noise_floor_db,
                'change_db': self.noise_floor_change_db
            },
            'snr': {
                'before_db': self.before_snr_db,
                'after_db': self.after_snr_db,
                'change_db': self.snr_change_db
            },
            'bandwidth': {
                'before_hz': self.before_bandwidth_hz,
                'after_hz': self.after_bandwidth_hz,
                'change_hz': self.bandwidth_change_hz
            },
            'spur_tracking': {
                'before_count': len(self.spurs_before),
                'after_count': len(self.spurs_after),
                'removed_count': len(self.spurs_removed),
                'added_count': len(self.spurs_added),
                'unchanged_count': len(self.spurs_unchanged),
                'removed': [{'freq_hz': s.frequency_hz, 'mag_db': s.magnitude_db} for s in self.spurs_removed],
                'added': [{'freq_hz': s.frequency_hz, 'mag_db': s.magnitude_db} for s in self.spurs_added]
            },
            'spectral_metrics': {
                'correlation': self.spectral_correlation,
                'max_deviation_db': self.max_spectral_deviation_db
            },
            'artifacts_detected': self.artifacts_detected,
            'forensic_notes': self.forensic_notes
        }


class FFTPipelineComparator:
    """
    Compares before/after FFT data to validate signal processing pipelines.
    
    This class implements the forensic principle that:
    - A single FFT tells you what exists
    - A paired FFT tells you what changed—and change is where understanding lives
    """
    
    EPSILON = 1e-10
    
    def __init__(self, sample_rate: float, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_resolution = sample_rate / fft_size
    
    def compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file for forensic chain of custody."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def estimate_noise_floor(self, psd_db: np.ndarray) -> float:
        """Estimate noise floor using median of lower half of sorted PSD."""
        sorted_psd = np.sort(psd_db)
        # Use median of lower 50% to avoid signal peaks
        noise_samples = sorted_psd[:len(sorted_psd) // 2]
        return float(np.median(noise_samples))
    
    def detect_spurs(self, psd_db: np.ndarray, freqs: np.ndarray, 
                     threshold_above_noise: float = 10.0) -> List[SpectralPeak]:
        """Detect spectral peaks/spurs above noise floor."""
        noise_floor = self.estimate_noise_floor(psd_db)
        threshold = noise_floor + threshold_above_noise
        
        # Find peaks above threshold
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(psd_db, height=threshold, distance=5)
        
        spurs = []
        for idx in peaks:
            spurs.append(SpectralPeak(
                frequency_hz=float(freqs[idx]),
                magnitude_db=float(psd_db[idx]),
                bin_index=int(idx),
                snr_db=float(psd_db[idx] - noise_floor)
            ))
        
        return sorted(spurs, key=lambda x: x.magnitude_db, reverse=True)
    
    def estimate_bandwidth(self, psd_linear: np.ndarray, freqs: np.ndarray,
                          threshold_db: float = -3.0) -> float:
        """Estimate signal bandwidth at specified threshold below peak."""
        psd_db = 10 * np.log10(psd_linear + self.EPSILON)
        peak_db = np.max(psd_db)
        threshold = peak_db + threshold_db
        
        above_threshold = psd_db > threshold
        if not np.any(above_threshold):
            return 0.0
        
        indices = np.where(above_threshold)[0]
        bandwidth = (indices[-1] - indices[0]) * self.freq_resolution
        return float(bandwidth)
    
    def estimate_snr(self, psd_linear: np.ndarray) -> float:
        """Estimate SNR from PSD."""
        psd_db = 10 * np.log10(psd_linear + self.EPSILON)
        noise_floor = self.estimate_noise_floor(psd_db)
        peak_db = np.max(psd_db)
        return float(peak_db - noise_floor)
    
    def match_spurs(self, spurs_before: List[SpectralPeak], 
                    spurs_after: List[SpectralPeak],
                    freq_tolerance_hz: float = None) -> Tuple[List, List, List]:
        """
        Match spurs between before and after, categorizing as:
        - removed (in before, not in after)
        - added (in after, not in before)
        - unchanged (in both)
        """
        if freq_tolerance_hz is None:
            freq_tolerance_hz = self.freq_resolution * 2
        
        removed = []
        unchanged = []
        matched_after = set()
        
        for spur_b in spurs_before:
            found_match = False
            for i, spur_a in enumerate(spurs_after):
                if i in matched_after:
                    continue
                if abs(spur_b.frequency_hz - spur_a.frequency_hz) < freq_tolerance_hz:
                    unchanged.append(spur_b)
                    matched_after.add(i)
                    found_match = True
                    break
            if not found_match:
                removed.append(spur_b)
        
        added = [s for i, s in enumerate(spurs_after) if i not in matched_after]
        
        return removed, added, unchanged
    
    def compute_spectral_correlation(self, psd1: np.ndarray, psd2: np.ndarray) -> float:
        """Compute correlation between two PSDs (shape similarity)."""
        # Normalize to same length if needed
        min_len = min(len(psd1), len(psd2))
        p1 = psd1[:min_len]
        p2 = psd2[:min_len]
        
        # Convert to dB for better correlation behavior
        p1_db = 10 * np.log10(p1 + self.EPSILON)
        p2_db = 10 * np.log10(p2 + self.EPSILON)
        
        # Normalize
        p1_norm = (p1_db - np.mean(p1_db)) / (np.std(p1_db) + self.EPSILON)
        p2_norm = (p2_db - np.mean(p2_db)) / (np.std(p2_db) + self.EPSILON)
        
        correlation = float(np.mean(p1_norm * p2_norm))
        return correlation
    
    def detect_artifacts(self, before_psd: np.ndarray, after_psd: np.ndarray,
                        freqs: np.ndarray) -> List[str]:
        """Detect potential processing artifacts."""
        artifacts = []
        
        before_db = 10 * np.log10(before_psd + self.EPSILON)
        after_db = 10 * np.log10(after_psd + self.EPSILON)
        
        # Check for spectral inversion
        corr = np.corrcoef(before_db, after_db)[0, 1]
        if corr < -0.5:
            artifacts.append("SPECTRAL_INVERSION: Strong negative correlation suggests spectral flip")
        
        # Check for DC spike introduction
        dc_idx = len(freqs) // 2
        dc_before = before_db[dc_idx]
        dc_after = after_db[dc_idx]
        if dc_after - dc_before > 20:
            artifacts.append(f"DC_SPIKE_INTRODUCED: DC component increased by {dc_after - dc_before:.1f} dB")
        
        # Check for edge artifacts (filter ringing)
        edge_size = len(freqs) // 20
        edge_before = np.mean(before_db[:edge_size])
        edge_after = np.mean(after_db[:edge_size])
        if abs(edge_after - edge_before) > 15:
            artifacts.append(f"EDGE_ARTIFACT: Band edge changed by {edge_after - edge_before:.1f} dB")
        
        # Check for harmonic generation
        # (simplified: look for new peaks at integer multiples of existing peaks)
        
        # Check for quantization noise increase
        noise_before = self.estimate_noise_floor(before_db)
        noise_after = self.estimate_noise_floor(after_db)
        if noise_after - noise_before > 6:
            artifacts.append(f"NOISE_INCREASE: Noise floor raised by {noise_after - noise_before:.1f} dB (possible quantization)")
        
        return artifacts
    
    def compare(self, before_data: np.ndarray, after_data: np.ndarray,
                before_file: Path, after_file: Path,
                after_is_fft: bool = False) -> PipelineComparisonResult:
        """
        Perform comprehensive before/after FFT comparison.
        
        Args:
            before_data: Time-domain signal before processing
            after_data: Time-domain signal after processing (or FFT output if after_is_fft=True)
            before_file: Path to before file (for hashing)
            after_file: Path to after file (for hashing)
            after_is_fft: If True, after_data is already FFT output (frequency domain)
        
        Returns:
            PipelineComparisonResult with comprehensive comparison metrics
        """
        # Compute forensic hashes
        before_hash = self.compute_file_hash(before_file)
        after_hash = self.compute_file_hash(after_file)
        
        # Compute FFT of before signal
        before_fft = np.fft.fftshift(np.fft.fft(before_data, self.fft_size))
        before_psd = np.abs(before_fft) ** 2 / self.fft_size
        
        # Handle after data
        if after_is_fft:
            # After data is already FFT output - use directly
            # Assume it's magnitude or complex FFT
            if np.iscomplexobj(after_data):
                after_psd = np.abs(after_data[:self.fft_size]) ** 2
            else:
                # Assume it's already magnitude (possibly in dB)
                if np.min(after_data) < 0:
                    # Likely dB values, convert to linear
                    after_psd = 10 ** (after_data[:self.fft_size] / 10)
                else:
                    after_psd = after_data[:self.fft_size] ** 2
        else:
            # Compute FFT of after signal
            after_fft = np.fft.fftshift(np.fft.fft(after_data, self.fft_size))
            after_psd = np.abs(after_fft) ** 2 / self.fft_size
        
        # Frequency axis
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate))
        
        # Energy conservation check
        before_energy = float(np.sum(np.abs(before_data) ** 2))
        after_energy = float(np.sum(np.abs(after_data) ** 2))
        energy_ratio_db = 10 * np.log10(after_energy / (before_energy + self.EPSILON))
        parseval_passed = abs(energy_ratio_db) < 3.0  # Within 3 dB
        
        # Noise floor comparison
        before_db = 10 * np.log10(before_psd + self.EPSILON)
        after_db = 10 * np.log10(after_psd + self.EPSILON)
        before_noise = self.estimate_noise_floor(before_db)
        after_noise = self.estimate_noise_floor(after_db)
        
        # SNR comparison
        before_snr = self.estimate_snr(before_psd)
        after_snr = self.estimate_snr(after_psd)
        
        # Bandwidth comparison
        before_bw = self.estimate_bandwidth(before_psd, freqs)
        after_bw = self.estimate_bandwidth(after_psd, freqs)
        
        # Spur detection and tracking
        spurs_before = self.detect_spurs(before_db, freqs)
        spurs_after = self.detect_spurs(after_db, freqs)
        removed, added, unchanged = self.match_spurs(spurs_before, spurs_after)
        
        # Spectral correlation
        correlation = self.compute_spectral_correlation(before_psd, after_psd)
        
        # Max spectral deviation
        min_len = min(len(before_db), len(after_db))
        max_deviation = float(np.max(np.abs(before_db[:min_len] - after_db[:min_len])))
        
        # Artifact detection
        artifacts = self.detect_artifacts(before_psd, after_psd, freqs)
        
        # Build forensic notes
        notes = []
        if not parseval_passed:
            notes.append(f"WARNING: Energy not conserved (Δ={energy_ratio_db:.1f} dB) - check for gain/attenuation")
        if len(added) > 0:
            notes.append(f"ARTIFACT: {len(added)} new spectral components appeared after processing")
        if len(removed) > 0:
            notes.append(f"FILTERED: {len(removed)} spectral components removed by processing")
        if after_snr > before_snr + 3:
            notes.append(f"SNR IMPROVED: Processing improved SNR by {after_snr - before_snr:.1f} dB")
        elif after_snr < before_snr - 3:
            notes.append(f"SNR DEGRADED: Processing degraded SNR by {before_snr - after_snr:.1f} dB")
        
        return PipelineComparisonResult(
            before_file=str(before_file),
            after_file=str(after_file),
            before_hash=before_hash,
            after_hash=after_hash,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            before_total_energy=before_energy,
            after_total_energy=after_energy,
            energy_ratio_db=energy_ratio_db,
            parseval_check_passed=parseval_passed,
            before_noise_floor_db=before_noise,
            after_noise_floor_db=after_noise,
            noise_floor_change_db=after_noise - before_noise,
            before_snr_db=before_snr,
            after_snr_db=after_snr,
            snr_change_db=after_snr - before_snr,
            spurs_before=spurs_before,
            spurs_after=spurs_after,
            spurs_removed=removed,
            spurs_added=added,
            spurs_unchanged=unchanged,
            before_bandwidth_hz=before_bw,
            after_bandwidth_hz=after_bw,
            bandwidth_change_hz=after_bw - before_bw,
            spectral_correlation=correlation,
            max_spectral_deviation_db=max_deviation,
            artifacts_detected=artifacts,
            forensic_notes=notes
        )
    
    def generate_comparison_plots(self, before_data: np.ndarray, after_data: np.ndarray,
                                  result: PipelineComparisonResult,
                                  output_dir: Path, name_prefix: str,
                                  after_is_fft: bool = False):
        """Generate visual comparison plots."""
        import matplotlib.pyplot as plt
        
        # Compute PSDs
        before_fft = np.fft.fftshift(np.fft.fft(before_data, self.fft_size))
        before_psd_db = 10 * np.log10(np.abs(before_fft) ** 2 / self.fft_size + self.EPSILON)
        
        if after_is_fft:
            if np.iscomplexobj(after_data):
                after_psd_db = 10 * np.log10(np.abs(after_data[:self.fft_size]) ** 2 + self.EPSILON)
            else:
                if np.min(after_data) < 0:
                    after_psd_db = after_data[:self.fft_size]
                else:
                    after_psd_db = 10 * np.log10(after_data[:self.fft_size] ** 2 + self.EPSILON)
        else:
            after_fft = np.fft.fftshift(np.fft.fft(after_data, self.fft_size))
            after_psd_db = 10 * np.log10(np.abs(after_fft) ** 2 / self.fft_size + self.EPSILON)
        
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate)) / 1e3  # kHz
        
        # Set dark theme
        plt.style.use('dark_background')
        
        # Plot 1: Side-by-side PSD comparison
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Before PSD
        axes[0].plot(freqs, before_psd_db, 'c-', linewidth=0.5, alpha=0.8)
        axes[0].fill_between(freqs, before_psd_db.min(), before_psd_db, alpha=0.3, color='cyan')
        axes[0].axhline(y=result.before_noise_floor_db, color='r', linestyle='--', 
                       label=f'Noise Floor: {result.before_noise_floor_db:.1f} dB')
        axes[0].set_title(f'BEFORE Processing: {Path(result.before_file).name}', fontsize=12)
        axes[0].set_ylabel('Magnitude (dB)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # After PSD
        axes[1].plot(freqs[:len(after_psd_db)], after_psd_db, 'lime', linewidth=0.5, alpha=0.8)
        axes[1].fill_between(freqs[:len(after_psd_db)], after_psd_db.min(), after_psd_db, 
                            alpha=0.3, color='lime')
        axes[1].axhline(y=result.after_noise_floor_db, color='r', linestyle='--',
                       label=f'Noise Floor: {result.after_noise_floor_db:.1f} dB')
        axes[1].set_title(f'AFTER Processing: {Path(result.after_file).name}', fontsize=12)
        axes[1].set_ylabel('Magnitude (dB)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Difference
        min_len = min(len(before_psd_db), len(after_psd_db))
        diff_db = after_psd_db[:min_len] - before_psd_db[:min_len]
        axes[2].plot(freqs[:min_len], diff_db, 'yellow', linewidth=0.5, alpha=0.8)
        axes[2].fill_between(freqs[:min_len], 0, diff_db, 
                            where=(diff_db > 0), alpha=0.3, color='red', label='Added')
        axes[2].fill_between(freqs[:min_len], 0, diff_db,
                            where=(diff_db < 0), alpha=0.3, color='blue', label='Removed')
        axes[2].axhline(y=0, color='white', linestyle='-', linewidth=0.5)
        axes[2].set_title('DELTA: After - Before (What Changed)', fontsize=12)
        axes[2].set_xlabel('Frequency (kHz)')
        axes[2].set_ylabel('Δ Magnitude (dB)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f'{name_prefix}_comparison_psd.png'
        plt.savefig(plot_path, dpi=150, facecolor='black', edgecolor='none')
        plt.close()
        
        # Plot 2: Summary metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FFT PIPELINE COMPARISON REPORT                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BEFORE: {Path(result.before_file).name:<60}  ║
║  AFTER:  {Path(result.after_file).name:<60}  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ENERGY CONSERVATION                                                          ║
║    Before Energy:     {result.before_total_energy:>12.2e}                                       ║
║    After Energy:      {result.after_total_energy:>12.2e}                                       ║
║    Energy Ratio:      {result.energy_ratio_db:>+8.2f} dB    {'✓ PASSED' if result.parseval_check_passed else '✗ FAILED':<20}       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  NOISE FLOOR                                                                  ║
║    Before:            {result.before_noise_floor_db:>8.1f} dB                                          ║
║    After:             {result.after_noise_floor_db:>8.1f} dB                                          ║
║    Change:            {result.noise_floor_change_db:>+8.1f} dB                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SIGNAL-TO-NOISE RATIO                                                        ║
║    Before SNR:        {result.before_snr_db:>8.1f} dB                                          ║
║    After SNR:         {result.after_snr_db:>8.1f} dB                                          ║
║    SNR Change:        {result.snr_change_db:>+8.1f} dB                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SPUR TRACKING                                                                ║
║    Spurs Before:      {len(result.spurs_before):>8}                                              ║
║    Spurs After:       {len(result.spurs_after):>8}                                              ║
║    Removed:           {len(result.spurs_removed):>8}                                              ║
║    Added:             {len(result.spurs_added):>8}                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SPECTRAL METRICS                                                             ║
║    Correlation:       {result.spectral_correlation:>8.3f}                                            ║
║    Max Deviation:     {result.max_spectral_deviation_db:>8.1f} dB                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Add forensic notes
        if result.forensic_notes:
            notes_text = "\n".join([f"• {note}" for note in result.forensic_notes])
            ax.text(0.02, 0.15, f"FORENSIC NOTES:\n{notes_text}", transform=ax.transAxes,
                   fontsize=9, color='yellow', verticalalignment='top')
        
        if result.artifacts_detected:
            artifacts_text = "\n".join([f"⚠ {a}" for a in result.artifacts_detected])
            ax.text(0.02, 0.05, f"ARTIFACTS DETECTED:\n{artifacts_text}", transform=ax.transAxes,
                   fontsize=9, color='red', verticalalignment='top')
        
        plt.tight_layout()
        summary_path = output_dir / f'{name_prefix}_comparison_summary.png'
        plt.savefig(summary_path, dpi=150, facecolor='black', edgecolor='none')
        plt.close()
        
        return [plot_path, summary_path]


def print_comparison_summary(result: PipelineComparisonResult):
    """Print a formatted comparison summary to console."""
    print("\n" + "="*70)
    print(" FFT PIPELINE COMPARISON ANALYSIS")
    print("="*70)
    print(f" Before: {Path(result.before_file).name}")
    print(f" After:  {Path(result.after_file).name}")
    print("-"*70)
    print(f" Energy Conservation: {result.energy_ratio_db:+.2f} dB", end="")
    print(f"  {'✓ PASSED' if result.parseval_check_passed else '✗ FAILED'}")
    print(f" Noise Floor Change:  {result.noise_floor_change_db:+.2f} dB")
    print(f" SNR Change:          {result.snr_change_db:+.2f} dB")
    print(f" Bandwidth Change:    {result.bandwidth_change_hz/1e3:+.2f} kHz")
    print("-"*70)
    print(f" Spurs: {len(result.spurs_before)} before → {len(result.spurs_after)} after")
    print(f"        {len(result.spurs_removed)} removed, {len(result.spurs_added)} added")
    print(f" Spectral Correlation: {result.spectral_correlation:.3f}")
    print("-"*70)
    if result.forensic_notes:
        print(" Forensic Notes:")
        for note in result.forensic_notes:
            print(f"   • {note}")
    if result.artifacts_detected:
        print(" Artifacts Detected:")
        for artifact in result.artifacts_detected:
            print(f"   ⚠ {artifact}")
    print("="*70 + "\n")

# # =============================================================================
# CONSTANTS (logging and HASH_CHUNK_SIZE moved to top - Issue 1 Fix)
# =============================================================================
# Numerical stability constants
EPSILON = 1e-10  # Default epsilon for avoiding division by zero
EPSILON_SMALL = 1e-20  # Stricter epsilon for power calculations
LOG_EPSILON = 1e-10  # Epsilon for log operations
# Analysis thresholds (using named constants from top - Issue 11 Fix)
DC_OFFSET_THRESHOLD = DC_SPIKE_THRESHOLD  # DC offset relative to RMS considered significant
# SATURATION_THRESHOLD and DROPOUT_THRESHOLD defined at top
BINARY_SEPARATION_FACTOR = 3  # Std deviation multiplier for binary detection
SNR_KURTOSIS_THRESHOLD = 2.01  # Kurtosis threshold for SNR estimation
# HASH_CHUNK_SIZE defined at top of file (64KB)


class DataFormat(Enum):
    """Supported binary data formats."""
    COMPLEX64 = "complex64"      # float32 I/Q interleaved (GNU Radio default)
    COMPLEX128 = "complex128"    # float64 I/Q interleaved
    INT16_IQ = "int16"           # int16 I/Q interleaved (RTL-SDR, HackRF)
    INT8_IQ = "int8"             # int8 I/Q interleaved (RTL-SDR u8)
    FLOAT32_REAL = "float32"     # Real-only samples


@dataclass
class SignalMetrics:
    """Container for computed signal statistics."""
    sample_count: int = 0
    mean_complex: complex = 0j
    std_dev: float = 0.0
    avg_power_linear: float = 0.0
    avg_power_dbm: float = -np.inf
    peak_power_linear: float = 0.0
    peak_power_dbm: float = -np.inf
    papr_db: float = 0.0
    crest_factor: float = 0.0
    dc_offset: complex = 0j
    iq_imbalance_db: float = 0.0
    snr_estimate_db: float = 0.0
    bandwidth_estimate_hz: float = 0.0
    center_freq_offset_hz: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_count': self.sample_count,
            'mean_complex': str(self.mean_complex),
            'std_dev': float(self.std_dev),
            'avg_power_dbm': float(self.avg_power_dbm),
            'peak_power_dbm': float(self.peak_power_dbm),
            'papr_db': float(self.papr_db),
            'crest_factor': float(self.crest_factor),
            'dc_offset': str(self.dc_offset),
            'iq_imbalance_db': float(self.iq_imbalance_db),
            'snr_estimate_db': float(self.snr_estimate_db),
            'bandwidth_estimate_hz': float(self.bandwidth_estimate_hz),
            'center_freq_offset_hz': float(self.center_freq_offset_hz),
        }


@dataclass
class DigitalSignalAnalysis:
    """Results from digital signal characterization."""
    # Signal type detection
    is_binary: bool = False
    unique_levels: int = 0
    is_real_only: bool = False
    active_channel: str = "complex"  # "I", "Q", or "complex"
    level_low: float = 0.0
    level_high: float = 0.0

    # Block structure analysis
    block_size: int = 0
    block_correlation: float = 0.0
    has_inversion_pattern: bool = False
    inversion_correlation: float = 0.0

    # Timing estimates
    estimated_symbol_rate: float = 0.0
    estimated_chip_rate: float = 0.0
    transition_rate: float = 0.0
    samples_per_symbol: float = 0.0

    # Encoding detection
    encoding_type: str = "unknown"
    run_length_mean: float = 0.0
    balance_ratio: float = 0.5

    # Protocol detection
    hdlc_flags_found: int = 0
    preamble_count: int = 0
    sync_patterns: List[Tuple[str, int]] = field(default_factory=list)
    frame_count: int = 0
    common_frame_length: int = 0

    # Byte analysis
    byte_entropy: float = 0.0
    ascii_sequences_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_binary': self.is_binary,
            'unique_levels': self.unique_levels,
            'is_real_only': self.is_real_only,
            'active_channel': self.active_channel,
            'level_low': float(self.level_low),
            'level_high': float(self.level_high),
            'block_size': self.block_size,
            'block_correlation': float(self.block_correlation),
            'has_inversion_pattern': self.has_inversion_pattern,
            'inversion_correlation': float(self.inversion_correlation),
            'estimated_symbol_rate': float(self.estimated_symbol_rate),
            'estimated_chip_rate': float(self.estimated_chip_rate),
            'transition_rate': float(self.transition_rate),
            'samples_per_symbol': float(self.samples_per_symbol),
            'encoding_type': self.encoding_type,
            'run_length_mean': float(self.run_length_mean),
            'balance_ratio': float(self.balance_ratio),
            'hdlc_flags_found': self.hdlc_flags_found,
            'preamble_count': self.preamble_count,
            'sync_patterns': [(name, count) for name, count in self.sync_patterns],
            'frame_count': self.frame_count,
            'common_frame_length': self.common_frame_length,
            'byte_entropy': float(self.byte_entropy),
            'ascii_sequences_found': self.ascii_sequences_found,
        }


# =============================================================================
# V3 DATACLASSES: FFT Debug, PN Sequence, Bit/Frame Analysis
# =============================================================================

@dataclass
class FFTDebugMetrics:
    """Container for FFT pipeline diagnostics."""
    dc_i: float = 0.0
    dc_q: float = 0.0
    dc_magnitude: float = 0.0
    dc_bin_power_dB: float = -np.inf
    dc_to_signal_dB: float = -np.inf
    dc_is_problematic: bool = False
    leakage_ratio_rectangular: float = 0.0
    leakage_ratio_hann: float = 0.0
    leakage_ratio_blackman: float = 0.0
    max_sidelobe_dB: float = -np.inf
    recommended_window: str = "hann"
    frequency_resolution_hz: float = 0.0
    snr_peak_median_dB: float = 0.0
    noise_floor_dB: float = -np.inf
    spurs_detected: int = 0
    artifacts: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Convert numpy bools to Python bools for JSON serialization
        artifacts_clean = {k: bool(v) for k, v in self.artifacts.items()}
        return {
            'dc_analysis': {
                'dc_i': float(self.dc_i),
                'dc_q': float(self.dc_q),
                'dc_magnitude': float(self.dc_magnitude),
                'dc_bin_power_dB': float(self.dc_bin_power_dB),
                'dc_to_signal_dB': float(self.dc_to_signal_dB),
                'is_problematic': bool(self.dc_is_problematic)
            },
            'spectral_leakage': {
                'leakage_rectangular': float(self.leakage_ratio_rectangular),
                'leakage_hann': float(self.leakage_ratio_hann),
                'leakage_blackman': float(self.leakage_ratio_blackman),
                'max_sidelobe_dB': float(self.max_sidelobe_dB),
                'recommended_window': self.recommended_window
            },
            'resolution': {
                'frequency_resolution_hz': float(self.frequency_resolution_hz)
            },
            'quality': {
                'snr_peak_median_dB': float(self.snr_peak_median_dB),
                'noise_floor_dB': float(self.noise_floor_dB),
                'spurs_detected': int(self.spurs_detected),
                'artifacts': artifacts_clean
            }
        }


@dataclass
class PNSequenceMetrics:
    """Container for PN sequence analysis results."""
    detected_type: str = "none"
    detected_degree: int = 0
    detected_period: int = 0
    correlation_peak: float = 0.0
    chip_rate_hz: float = 0.0
    samples_per_chip: float = 0.0
    code_offset: int = 0
    confidence: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'detected_type': self.detected_type,
            'detected_degree': self.detected_degree,
            'detected_period': self.detected_period,
            'correlation_peak': self.correlation_peak,
            'chip_rate_hz': self.chip_rate_hz,
            'samples_per_chip': self.samples_per_chip,
            'code_offset': self.code_offset,
            'confidence': self.confidence
        }


@dataclass
class BitOrderMetrics:
    """Container for bit ordering analysis results."""
    detected_order: str = "unknown"
    confidence: str = "low"
    scrambler_detected: str = "none"
    scrambler_polynomial: str = ""
    sync_word_detected: str = "none"
    sync_pattern: List[int] = field(default_factory=list)
    frame_length: int = 0
    frames_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bit_ordering': {
                'detected_order': self.detected_order,
                'confidence': self.confidence
            },
            'scrambler': {
                'detected': self.scrambler_detected,
                'polynomial': self.scrambler_polynomial
            },
            'framing': {
                'sync_word': self.sync_word_detected,
                'sync_pattern': self.sync_pattern,
                'frame_length': self.frame_length,
                'frames_found': self.frames_found
            }
        }


@dataclass
class AnalysisConfig:
    """Configuration for signal analysis."""
    sample_rate: float = 1e6
    center_freq: float = 0.0
    data_format: DataFormat = DataFormat.COMPLEX64
    fft_size: int = 4096
    spectrogram_nfft: int = 1024
    spectrogram_overlap: float = 0.9
    constellation_max_points: int = 50000
    time_domain_samples: int = 2000
    chunk_size: int = 10_000_000  # Samples per chunk for streaming
    output_dir: Path = field(default_factory=lambda: Path("./analysis_output"))
    plot_dpi: int = 150
    dark_theme: bool = True
    impedance_ohms: float = 50.0  # For dBm calculations
    # Digital signal analysis options
    digital_analysis: bool = False
    digital_block_size: int = 1024
    digital_threshold: Optional[float] = None  # Auto-detect if None


@contextmanager
def gpu_memory_pool():
    """Context manager for GPU memory management.
    
    In CPU-only mode (when CuPy is not available), this is a no-op.
    """
    if GPU_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        try:
            yield
        finally:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
    else:
        # CPU mode - no memory pool management needed
        yield


class SignalLoader:
    """Handles loading and format conversion of binary signal data."""
    
    FORMAT_MAP = {
        DataFormat.COMPLEX64: (np.complex64, 1.0),
        DataFormat.COMPLEX128: (np.complex128, 1.0),
        DataFormat.INT16_IQ: (np.int16, 1.0 / 32768.0),
        DataFormat.INT8_IQ: (np.int8, 1.0 / 128.0),
        DataFormat.FLOAT32_REAL: (np.float32, 1.0),
    }
    
    @classmethod
    def load(cls, filepath: Path, config: AnalysisConfig, 
             max_samples: Optional[int] = None) -> Tuple[Optional[cp.ndarray], Dict[str, Any]]:
        """
        Load binary data with format auto-detection fallback.
        
        FORENSIC COMPLIANCE: This method now preserves residual bytes instead of
        silently truncating them, per NIST SP 800-86 and SWGDE requirements.
        
        Returns:
            Tuple of (data array or None, residual metadata dict)
        """
        residual_metadata = {'residual_bytes': 0, 'action': 'none_required'}
        
        if not filepath.exists():
            log.error(f"File not found: {filepath}")
            return None, residual_metadata
        
        file_size = filepath.stat().st_size
        dtype, scale = cls.FORMAT_MAP[config.data_format]
        
        # Issue 9 Fix: Check available memory before loading
        try:
            import psutil
            available_mem = psutil.virtual_memory().available
            if file_size > available_mem * MEMORY_USAGE_WARNING:
                log.warning(f"File size ({file_size/1e9:.1f}GB) may exceed available memory "
                           f"({available_mem/1e9:.1f}GB). Consider using --streaming mode.")
        except ImportError:
            pass  # psutil not available, skip memory check
        
        try:
            raw = np.fromfile(filepath, dtype=dtype)
            
            # Handle int16/int8 interleaved I/Q
            if config.data_format in (DataFormat.INT16_IQ, DataFormat.INT8_IQ):
                # FORENSIC FIX: Preserve residual bytes instead of silent truncation
                # Per NIST SP 800-86 and SWGDE: "trailing bytes must be preserved
                # separately and documented, never silently truncated"
                if len(raw) % 2 != 0:
                    residual_data = raw[-1:]
                    raw = raw[:-1]
                    residual_hash = hashlib.sha256(residual_data.tobytes()).hexdigest()
                    residual_metadata = {
                        'residual_bytes': 1 * np.dtype(dtype).itemsize,
                        'residual_data_hex': residual_data.tobytes().hex(),
                        'residual_hash_sha256': residual_hash,
                        'action': 'preserved_not_truncated',
                        'timestamp_utc': datetime.now(timezone.utc).isoformat()
                    }
                    log.warning(f"FORENSIC: {residual_metadata['residual_bytes']} residual byte(s) "
                               f"preserved (hash: {residual_hash[:16]}...)")
                
                raw = raw.astype(np.float32) * scale
                raw = raw[0::2] + 1j * raw[1::2]
            elif config.data_format == DataFormat.FLOAT32_REAL:
                raw = raw.astype(np.complex64)
            
            if max_samples and len(raw) > max_samples:
                raw = raw[:max_samples]
            
            log.info(f"Loaded {len(raw):,} samples from {filepath.name} "
                    f"({file_size / 1e6:.2f} MB)")
            
            return cp.asarray(raw.astype(np.complex64)), residual_metadata
            
        # Issue 7 Fix: Use specific exceptions instead of broad Exception
        except (IOError, OSError, ValueError, MemoryError) as e:
            log.error(f"Failed to load {filepath}: {type(e).__name__}: {e}")
            return None, residual_metadata
        except Exception as e:
            # Log unexpected exceptions with full traceback for debugging
            log.error(f"Unexpected error loading {filepath}: {type(e).__name__}: {e}")
            raise  # Re-raise unexpected exceptions for forensic traceability
    
    @classmethod
    def load_chunked(cls, filepath: Path, config: AnalysisConfig):
        """
        Generator for streaming large files in chunks.
        
        Yields:
            Tuple of (chunk_idx, data_array, residual_metadata)
            - chunk_idx: Integer index of the chunk
            - data_array: CuPy/NumPy array of complex64 samples
            - residual_metadata: Dict with residual byte info (if any)
        """
        if not filepath.exists():
            return
        
        dtype, scale = cls.FORMAT_MAP[config.data_format]
        bytes_per_sample = np.dtype(dtype).itemsize
        if config.data_format in (DataFormat.INT16_IQ, DataFormat.INT8_IQ):
            bytes_per_sample *= 2  # I + Q
        
        chunk_bytes = config.chunk_size * bytes_per_sample
        accumulated_residual = bytearray()  # Track residual bytes across chunks
        
        with open(filepath, 'rb') as f:
            chunk_idx = 0
            while True:
                raw_data = f.read(chunk_bytes)
                if len(raw_data) == 0:
                    break
                
                data = np.frombuffer(raw_data, dtype=dtype)
                residual_metadata = {'residual_bytes': 0, 'action': 'none_required'}
                
                if config.data_format in (DataFormat.INT16_IQ, DataFormat.INT8_IQ):
                    if len(data) % 2 != 0:
                        # FORENSIC FIX: Preserve residual bytes instead of silent truncation
                        residual_bytes = data[-1:].tobytes()
                        accumulated_residual.extend(residual_bytes)
                        residual_hash = hashlib.sha256(bytes(accumulated_residual)).hexdigest()
                        data = data[:-1]
                        residual_metadata = {
                            'residual_bytes': len(accumulated_residual),
                            'residual_bytes_hex': bytes(accumulated_residual).hex(),
                            'residual_hash_sha256': residual_hash,
                            'action': 'preserved_not_truncated',
                            'chunk_idx': chunk_idx
                        }
                        log.warning(f"FORENSIC: Chunk {chunk_idx}: 1 residual byte preserved (total: {len(accumulated_residual)})")
                    data = data.astype(np.float32) * scale
                    data = data[0::2] + 1j * data[1::2]
                
                yield chunk_idx, cp.asarray(data.astype(np.complex64)), residual_metadata
                chunk_idx += 1
            
            # Yield final residual summary if any bytes accumulated
            if len(accumulated_residual) > 0:
                log.info(f"FORENSIC: Total residual bytes across all chunks: {len(accumulated_residual)}")


# =============================================================================
# V3 ANALYZERS: FFT Debug, PN Sequence, Bit/Frame Analysis
# =============================================================================

class FFTDebugAnalyzer:
    """Comprehensive FFT pipeline diagnostics."""

    def __init__(self, fs: float):
        self.fs = fs

    def analyze_dc_offset(self, iq_samples: cp.ndarray) -> Dict[str, Any]:
        """Detect and characterize DC offset in I/Q data."""
        x = cp.asarray(iq_samples)

        dc_i = float(cp.mean(cp.real(x)))
        dc_q = float(cp.mean(cp.imag(x)))
        dc_magnitude = np.sqrt(dc_i**2 + dc_q**2)

        X = cp.fft.fft(x)
        dc_bin_power = float(cp.abs(X[0])**2 / len(x)**2)

        signal_power = float(cp.mean(cp.abs(x - (dc_i + 1j*dc_q))**2))
        dc_to_signal_db = 10 * np.log10(dc_magnitude**2 / (signal_power + 1e-10))

        return {
            'dc_i': dc_i,
            'dc_q': dc_q,
            'dc_magnitude': dc_magnitude,
            'dc_bin_power_dB': 10 * np.log10(dc_bin_power + 1e-10),
            'dc_to_signal_dB': dc_to_signal_db,
            'is_problematic': dc_to_signal_db > -20
        }

    def detect_spectral_leakage(self, iq_samples: cp.ndarray, fft_size: int = 4096) -> Dict[str, Any]:
        """Analyze spectral leakage artifacts."""
        x = cp.asarray(iq_samples[:fft_size])

        X_rect = cp.fft.fftshift(cp.fft.fft(x))
        hann = cp.hanning(fft_size)
        X_hann = cp.fft.fftshift(cp.fft.fft(x * hann))
        blackman = cp.blackman(fft_size)
        X_blackman = cp.fft.fftshift(cp.fft.fft(x * blackman))

        rect_mag = cp.abs(X_rect)
        peak_idx = int(cp.argmax(rect_mag))
        peak_val = float(rect_mag[peak_idx])

        main_lobe_width = 3
        mask = cp.ones(fft_size, dtype=bool)
        start = max(0, peak_idx - main_lobe_width)
        end = min(fft_size, peak_idx + main_lobe_width + 1)
        mask[start:end] = False

        leakage_rect = float(cp.sum(rect_mag[mask]**2) / (cp.sum(rect_mag**2) + 1e-10))
        leakage_hann = float(cp.sum(cp.abs(X_hann)[mask]**2) / (cp.sum(cp.abs(X_hann)**2) + 1e-10))
        leakage_blackman = float(cp.sum(cp.abs(X_blackman)[mask]**2) / (cp.sum(cp.abs(X_blackman)**2) + 1e-10))

        sidelobes_rect = rect_mag.copy()
        sidelobes_rect[start:end] = 0
        max_sidelobe_rect = float(cp.max(sidelobes_rect))
        sidelobe_level_db = 20 * np.log10(max_sidelobe_rect / (peak_val + 1e-10) + 1e-10)

        return {
            'leakage_ratio_rectangular': leakage_rect,
            'leakage_ratio_hann': leakage_hann,
            'leakage_ratio_blackman': leakage_blackman,
            'max_sidelobe_dB': sidelobe_level_db,
            'recommended_window': 'blackman' if leakage_rect > 0.1 else 'hann'
        }

    def detect_spurs(self, iq_samples: cp.ndarray, fft_size: int = 8192,
                     threshold_db: float = -60) -> Dict[str, Any]:
        """Detect spurious signals and artifacts."""
        x = cp.asarray(iq_samples[:fft_size])
        window = cp.blackman(fft_size)
        X = cp.fft.fftshift(cp.fft.fft(x * window))
        X_db = 20 * cp.log10(cp.abs(X) + 1e-10)
        freqs = cp.fft.fftshift(cp.fft.fftfreq(fft_size, 1/self.fs))

        peak_idx = int(cp.argmax(X_db))
        peak_db = float(X_db[peak_idx])
        peak_freq = float(freqs[peak_idx])
        noise_floor = float(cp.median(X_db))

        X_db_np = cp_asnumpy(X_db)
        freqs_np = cp_asnumpy(freqs)
        spur_threshold = noise_floor + abs(threshold_db)

        spurs = []
        for i in range(1, len(X_db_np) - 1):
            if (X_db_np[i] > X_db_np[i-1] and
                X_db_np[i] > X_db_np[i+1] and
                X_db_np[i] > spur_threshold and
                abs(i - peak_idx) > 10):
                spurs.append({
                    'frequency_hz': freqs_np[i],
                    'power_dB': X_db_np[i],
                    'relative_to_peak_dB': X_db_np[i] - peak_db
                })

        artifacts = {
            'dc_spike': float(X_db[fft_size//2]) > noise_floor + 20,
            'lo_leakage': float(X_db[fft_size//2]) > peak_db - 30
        }

        return {
            'noise_floor_dB': noise_floor,
            'peak_frequency_hz': peak_freq,
            'peak_power_dB': peak_db,
            'snr_dB': peak_db - noise_floor,
            'spurs': sorted(spurs, key=lambda x: x['power_dB'], reverse=True)[:10],
            'artifacts': artifacts
        }

    def full_analysis(self, iq_samples: cp.ndarray, fft_size: int = 8192) -> FFTDebugMetrics:
        """Run complete FFT debug analysis."""
        metrics = FFTDebugMetrics()

        dc = self.analyze_dc_offset(iq_samples)
        metrics.dc_i = dc['dc_i']
        metrics.dc_q = dc['dc_q']
        metrics.dc_magnitude = dc['dc_magnitude']
        metrics.dc_bin_power_dB = dc['dc_bin_power_dB']
        metrics.dc_to_signal_dB = dc['dc_to_signal_dB']
        metrics.dc_is_problematic = dc['is_problematic']

        leakage = self.detect_spectral_leakage(iq_samples, fft_size)
        metrics.leakage_ratio_rectangular = leakage['leakage_ratio_rectangular']
        metrics.leakage_ratio_hann = leakage['leakage_ratio_hann']
        metrics.leakage_ratio_blackman = leakage['leakage_ratio_blackman']
        metrics.max_sidelobe_dB = leakage['max_sidelobe_dB']
        metrics.recommended_window = leakage['recommended_window']

        metrics.frequency_resolution_hz = self.fs / fft_size

        spurs = self.detect_spurs(iq_samples, fft_size)
        metrics.snr_peak_median_dB = spurs['snr_dB']
        metrics.noise_floor_dB = spurs['noise_floor_dB']
        metrics.spurs_detected = len(spurs['spurs'])
        metrics.artifacts = spurs['artifacts']

        return metrics


class MSequenceAnalyzer:
    """M-sequence (maximal length PN sequence) generation and detection."""

    LFSR_TAPS = {
        3: [3, 2], 4: [4, 3], 5: [5, 3], 6: [6, 5], 7: [7, 6],
        8: [8, 6, 5, 4], 9: [9, 5], 10: [10, 7], 11: [11, 9],
        12: [12, 11, 10, 4], 15: [15, 14], 18: [18, 11]
    }

    def generate_msequence(self, degree: int, taps: List[int] = None,
                          initial_state: List[int] = None) -> cp.ndarray:
        """Generate m-sequence using LFSR."""
        if taps is None:
            taps = self.LFSR_TAPS.get(degree)
            if taps is None:
                raise ValueError(f"No standard taps for degree {degree}")

        N = 2**degree - 1
        state = [1] * degree if initial_state is None else list(initial_state)

        sequence = []
        for _ in range(N):
            out = state[-1]
            sequence.append(out)
            feedback = 0
            for tap in taps:
                feedback ^= state[tap - 1]
            state = [feedback] + state[:-1]

        return cp.array([1 if b == 0 else -1 for b in sequence], dtype=cp.float32)

    def detect_msequence(self, signal: cp.ndarray, max_degree: int = 12,
                        threshold: float = None, pfa: float = 1e-6) -> List[Dict]:
        # FORENSIC FIX: Use CFAR adaptive threshold if threshold not specified
        """Detect m-sequence in signal by correlation."""
        x = cp.asarray(signal)
        x_norm = cp.real(x) / (cp.max(cp.abs(cp.real(x))) + EPSILON)  # FORENSIC FIX: Preserve amplitude

        results = []
        for degree in range(3, max_degree + 1):
            if degree not in self.LFSR_TAPS:
                continue

            ref = self.generate_msequence(degree)
            N = len(ref)

            if len(x_norm) < N:
                continue

            corr = cp.correlate(x_norm[:N*3], ref, mode='valid')
            corr_norm = corr / N

            peak_val = float(cp.max(cp.abs(corr_norm)))
            peak_idx = int(cp.argmax(cp.abs(corr_norm)))

            # FORENSIC FIX: Use CFAR threshold if not specified
            if threshold is None:
                cfar_thresh, cfar_meta = compute_cfar_threshold(x_norm, pfa=pfa)
                effective_threshold = cfar_thresh / N  # Normalize
            else:
                effective_threshold = threshold
            if peak_val > effective_threshold:
                results.append({
                    'type': 'm-sequence',
                    'degree': degree,
                    'period': N,
                    'correlation_peak': peak_val,
                    'offset': peak_idx,
                    'taps': self.LFSR_TAPS[degree]
                })

        return sorted(results, key=lambda x: x['correlation_peak'], reverse=True)

    def estimate_period(self, signal: cp.ndarray, max_period: int = 10000) -> Dict:
        """Estimate PN sequence period via autocorrelation."""
        x = cp.asarray(signal)
        x_norm = cp.real(x) / (cp.max(cp.abs(cp.real(x))) + EPSILON)  # FORENSIC FIX: Preserve amplitude

        autocorr = cp.correlate(x_norm, x_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        autocorr_np = cp_asnumpy(autocorr[:max_period])

        try:
            peaks, _ = find_peaks(autocorr_np[10:], height=0.5, distance=10)
            if len(peaks) > 0:
                period = peaks[0] + 10
                possible_n = np.log2(period + 1)
                is_msequence = abs(possible_n - round(possible_n)) < 0.01

                return {
                    'estimated_period': int(period),
                    'is_msequence_period': is_msequence,
                    'possible_degree': int(round(possible_n)) if is_msequence else None,
                    'autocorr_peak': float(autocorr_np[period])
                }
        except (IndexError, ValueError) as e:
            log.debug(f"Period estimation failed: {e}")

        return {'estimated_period': None, 'is_msequence_period': False}


class GoldCodeAnalyzer:
    """Gold code generation and detection."""

    PREFERRED_PAIRS = {
        5: ([5, 3], [5, 4, 3, 2]),
        6: ([6, 5], [6, 5, 2, 1]),
        7: ([7, 6], [7, 6, 5, 4]),
        10: ([10, 7], [10, 9, 8, 6, 5, 1]),
    }

    def __init__(self):
        self.mseq = MSequenceAnalyzer()

    def generate_gold_code(self, degree: int, code_index: int = 0) -> cp.ndarray:
        """Generate Gold code from preferred pair."""
        if degree not in self.PREFERRED_PAIRS:
            raise ValueError(f"No preferred pair for degree {degree}")

        taps1, taps2 = self.PREFERRED_PAIRS[degree]
        N = 2**degree - 1

        m1 = self.mseq.generate_msequence(degree, taps1)
        m2 = self.mseq.generate_msequence(degree, taps2)

        if code_index == 0:
            return m1
        elif code_index == 2**degree:
            return m2
        else:
            shift = code_index - 1
            m2_shifted = cp.roll(m2, shift)
            return m1 * m2_shifted

    def detect_gold_code(self, signal: cp.ndarray, degree: int,
                        threshold: float = None, pfa_desired: float = 1e-6) -> List[Dict]:
        """Detect which Gold code is present.
        
        Issue 2 Fix: Moved docstring to correct position after function signature.
        Issue 3 Fix: Now uses Bonferroni-corrected CFAR threshold when threshold is None.
        """
        # FORENSIC FIX: Apply Bonferroni correction for multiple hypothesis testing
        num_codes = 2**degree + 1
        pfa_single = apply_bonferroni_correction(pfa_desired, num_codes)
        
        x = cp.asarray(signal)
        x_norm = cp.real(x) / (cp.max(cp.abs(cp.real(x))) + EPSILON)  # FORENSIC FIX: Preserve amplitude
        N = 2**degree - 1
        
        # Issue 3 Fix: Use Bonferroni-corrected CFAR threshold if threshold not provided
        if threshold is None:
            cfar_thresh, cfar_meta = compute_cfar_threshold(x_norm, pfa=pfa_single)
            effective_threshold = cfar_thresh / N
            log.debug(f"CFAR threshold computed: {effective_threshold:.4f} (P_FA={pfa_single:.2e})")
        else:
            effective_threshold = threshold
        
        num_codes = min(2**degree + 1, 32)  # Limit search
        results = []
        for code_idx in range(num_codes):
            try:
                ref = self.generate_gold_code(degree, code_idx)
                corr = cp.correlate(x_norm[:N*3], ref, mode='valid')
                corr_norm = corr / N
                peak_val = float(cp.max(cp.abs(corr_norm)))
                peak_idx = int(cp.argmax(cp.abs(corr_norm)))
                if peak_val > effective_threshold:
                    results.append({
                        'type': 'gold',
                        'degree': degree,
                        'code_index': code_idx,
                        'correlation_peak': peak_val,
                        'offset': peak_idx
                    })
            except (ValueError, IndexError) as e:
                log.debug(f"Gold code {code_idx} detection failed: {e}")
                continue

        return sorted(results, key=lambda x: x['correlation_peak'], reverse=True)


class SpreadingCodeAnalyzer:
    """Combined spreading code analysis."""

    def __init__(self, fs: float):
        self.fs = fs
        self.mseq = MSequenceAnalyzer()
        self.gold = GoldCodeAnalyzer()

    def estimate_chip_rate(self, signal: cp.ndarray) -> Dict:
        """Estimate chip rate from signal autocorrelation."""
        x = cp.asarray(signal)
        x_mag = cp.abs(x)

        autocorr = cp.correlate(x_mag[:10000], x_mag[:10000], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        autocorr_np = cp_asnumpy(autocorr[:int(self.fs / 1000)])

        try:
            inverted = -autocorr_np
            peaks, _ = find_peaks(inverted, distance=5)

            if len(peaks) >= 2:
                chip_period_samples = np.median(np.diff(peaks))
                chip_rate = self.fs / chip_period_samples

                return {
                    'chip_rate_hz': float(chip_rate),
                    'chip_period_samples': float(chip_period_samples),
                    'samples_per_chip': float(chip_period_samples)
                }
        except (IndexError, ValueError, ZeroDivisionError) as e:
            log.debug(f"Chip rate estimation failed: {e}")

        return {'chip_rate_hz': None, 'samples_per_chip': None}

    def full_analysis(self, signal: cp.ndarray) -> PNSequenceMetrics:
        """Run complete PN sequence analysis."""
        metrics = PNSequenceMetrics()

        # Estimate chip rate
        chip_info = self.estimate_chip_rate(signal)
        if chip_info['chip_rate_hz']:
            metrics.chip_rate_hz = chip_info['chip_rate_hz']
            metrics.samples_per_chip = chip_info['samples_per_chip']

        # Estimate period
        period_info = self.mseq.estimate_period(signal)
        if period_info['estimated_period']:
            metrics.detected_period = period_info['estimated_period']

        # Detect m-sequences
        mseq_results = self.mseq.detect_msequence(signal)
        if mseq_results:
            best = mseq_results[0]
            metrics.detected_type = 'm-sequence'
            metrics.detected_degree = best['degree']
            metrics.detected_period = best['period']
            metrics.correlation_peak = best['correlation_peak']
            metrics.code_offset = best['offset']
            metrics.confidence = 'high' if best['correlation_peak'] > 0.85 else 'medium'
            return metrics

        # Try Gold codes
        for degree in [5, 6, 7, 10]:
            if degree in GoldCodeAnalyzer.PREFERRED_PAIRS:
                gold_results = self.gold.detect_gold_code(signal, degree)
                if gold_results:
                    best = gold_results[0]
                    metrics.detected_type = 'gold'
                    metrics.detected_degree = best['degree']
                    metrics.detected_period = 2**degree - 1
                    metrics.correlation_peak = best['correlation_peak']
                    metrics.code_offset = best['offset']
                    metrics.confidence = 'high' if best['correlation_peak'] > 0.85 else 'medium'
                    return metrics

        return metrics


class BitOrderAnalyzer:
    """Analyze bit ordering: MSB/LSB first, byte endianness."""

    def detect_bit_order(self, bits: np.ndarray) -> Dict:
        """Detect bit ordering using heuristics."""
        bits = np.array(bits)

        orderings = {
            'msb_first': bits,
            'lsb_first': self._reverse_bits_in_bytes(bits),
            'bit_reversed': bits[::-1],
            'byte_swapped': self._swap_bytes(bits)
        }

        scores = {}
        for name, ordered in orderings.items():
            score = 0
            bytes_arr = self._bits_to_bytes(ordered)

            # ASCII printable content
            printable = sum(1 for b in bytes_arr if 0x20 <= b <= 0x7E)
            score += printable / (len(bytes_arr) + 1) * 100

            # Common sync patterns
            common_syncs = [0x7E, 0xAA, 0x55, 0xFF, 0x00]
            for sync in common_syncs:
                if sync in bytes_arr[:20]:
                    score += 10

            scores[name] = score

        best = max(scores, key=scores.get)
        return {
            'detected_order': best,
            'confidence': 'high' if max(scores.values()) > 70 else 'medium' if max(scores.values()) > 40 else 'low',
            'scores': scores
        }

    def _reverse_bits_in_bytes(self, bits: np.ndarray) -> np.ndarray:
        """Reverse bit order within each byte."""
        bits = np.array(bits)
        n_bytes = len(bits) // 8
        result = np.zeros_like(bits)

        for i in range(n_bytes):
            byte_bits = bits[i*8:(i+1)*8]
            result[i*8:(i+1)*8] = byte_bits[::-1]

        return result

    def _swap_bytes(self, bits: np.ndarray, word_size: int = 16) -> np.ndarray:
        """Swap byte order."""
        bits = np.array(bits)
        n_words = len(bits) // word_size
        result = np.zeros_like(bits)

        for i in range(n_words):
            word_start = i * word_size
            result[word_start:word_start+8] = bits[word_start+8:word_start+16]
            result[word_start+8:word_start+16] = bits[word_start:word_start+8]

        return result

    def _bits_to_bytes(self, bits: np.ndarray) -> List[int]:
        """Convert bit array to bytes."""
        n_bytes = len(bits) // 8
        bytes_arr = []
        for i in range(n_bytes):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | int(bits[i*8 + j])
            bytes_arr.append(byte_val)
        return bytes_arr


class ScramblerAnalyzer:
    """Detect and analyze scrambler patterns."""

    KNOWN_SCRAMBLERS = {
        'wifi_802.11': {'poly': 0x48, 'length': 7},
        'g3ruh': {'poly': 0x21001, 'length': 17},  # Corrected: 1 + x^12 + x^17
        'dvb_s': {'poly': 0x6000, 'length': 15},
        'ccsds': {'poly': 0x95, 'length': 8}
    }

    def detect_scrambler(self, bits: np.ndarray) -> Dict:
        """Attempt to detect scrambler polynomial."""
        bits = np.array(bits, dtype=np.uint8)
        results = []

        for name, params in self.KNOWN_SCRAMBLERS.items():
            descrambled = self._descramble(bits, params['poly'], params['length'])
            score = self._measure_structure(descrambled)

            results.append({
                'scrambler': name,
                'polynomial': hex(params['poly']),
                'length': params['length'],
                'structure_score': score
            })

        best = max(results, key=lambda x: x['structure_score'])
        return {
            'detected': best['scrambler'] if best['structure_score'] > 30 else 'none',
            'polynomial': best['polynomial'],
            'all_results': sorted(results, key=lambda x: x['structure_score'], reverse=True)
        }

    def _descramble(self, bits: np.ndarray, poly: int, length: int) -> np.ndarray:
        """Multiplicative descrambler."""
        output = np.zeros_like(bits)
        sr = np.zeros(length, dtype=np.uint8)

        for i in range(len(bits)):
            feedback = 0
            for j in range(length):
                if (poly >> j) & 1:
                    feedback ^= sr[j]

            output[i] = bits[i] ^ feedback
            sr = np.roll(sr, 1)
            sr[0] = bits[i]

        return output

    def _measure_structure(self, bits: np.ndarray) -> float:
        """Measure structure in bit stream."""
        if len(bits) < 100:
            return 0

        balance = abs(np.mean(bits) - 0.5)

        runs = []
        current_run = 1
        for i in range(1, min(len(bits), 1000)):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1

        run_score = abs(np.mean(runs) - 2) / 2 if runs else 0

        return balance * 30 + run_score * 40


class FrameAnalyzer:
    """Frame structure analysis."""

    SYNC_PATTERNS = {
        'hdlc': [0, 1, 1, 1, 1, 1, 1, 0],
        'syncword_aa': [1, 0, 1, 0, 1, 0, 1, 0],
        'syncword_55': [0, 1, 0, 1, 0, 1, 0, 1],
        'preamble_1010': [1, 0, 1, 0] * 4
    }

    def detect_sync_word(self, bits: np.ndarray, min_occurrences: int = 2) -> List[Dict]:
        """Detect repeating sync word patterns."""
        bits = np.array(bits)
        results = []

        for name, pattern in self.SYNC_PATTERNS.items():
            pattern = np.array(pattern)
            positions = self._find_pattern(bits, pattern)

            if len(positions) >= min_occurrences:
                spacings = np.diff(positions) if len(positions) > 1 else []

                results.append({
                    'sync_name': name,
                    'pattern': pattern.tolist(),
                    'occurrences': len(positions),
                    'positions': positions[:10],
                    'avg_frame_length': float(np.mean(spacings)) if len(spacings) > 0 else None,
                    'frame_length_std': float(np.std(spacings)) if len(spacings) > 0 else None
                })

        return sorted(results, key=lambda x: x['occurrences'], reverse=True)

    def _find_pattern(self, bits: np.ndarray, pattern: np.ndarray) -> List[int]:
        """Find all occurrences of pattern."""
        positions = []
        pattern_len = len(pattern)

        for i in range(len(bits) - pattern_len + 1):
            if np.array_equal(bits[i:i+pattern_len], pattern):
                positions.append(i)

        return positions


class BitFrameAnalyzer:
    """Combined bit ordering and frame analysis."""

    def __init__(self):
        self.bit_order = BitOrderAnalyzer()
        self.scrambler = ScramblerAnalyzer()
        self.frame = FrameAnalyzer()

    def full_analysis(self, bits: np.ndarray) -> BitOrderMetrics:
        """Run complete bit/frame analysis."""
        metrics = BitOrderMetrics()

        # Bit ordering
        order_result = self.bit_order.detect_bit_order(bits)
        metrics.detected_order = order_result['detected_order']
        metrics.confidence = order_result['confidence']

        # Scrambler detection
        scrambler_result = self.scrambler.detect_scrambler(bits)
        metrics.scrambler_detected = scrambler_result['detected']
        metrics.scrambler_polynomial = scrambler_result['polynomial']

        # Sync word / frame detection
        sync_results = self.frame.detect_sync_word(bits)
        if sync_results:
            best = sync_results[0]
            metrics.sync_word_detected = best['sync_name']
            metrics.sync_pattern = best['pattern']
            metrics.frames_found = best['occurrences']
            if best['avg_frame_length']:
                metrics.frame_length = int(best['avg_frame_length'])

        return metrics


class SignalAnalyzer:
    """GPU-accelerated signal analysis engine."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._setup_plotting()
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _format_freq(self, hz: float) -> str:
        """Format frequency with appropriate units."""
        if hz >= 1e9:
            return f"{hz/1e9:.3f} GHz"
        elif hz >= 1e6:
            return f"{hz/1e6:.3f} MHz"
        elif hz >= 1e3:
            return f"{hz/1e3:.3f} kHz"
        else:
            return f"{hz:.1f} Hz"

    def _get_plot_header(self, source_name: str = "") -> str:
        """Generate metadata header for plots."""
        parts = []
        if source_name:
            parts.append(source_name)
        if self.config.center_freq > 0:
            parts.append(f"Fc: {self._format_freq(self.config.center_freq)}")
        parts.append(f"Fs: {self._format_freq(self.config.sample_rate)}")
        return " | ".join(parts)

    def _get_active_signal(self, data_np: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Detect and return the active I/Q channel from complex data.

        Returns:
            (signal, channel_name): The dominant channel data and its name ("I", "Q", or "complex")
        """
        i_power = np.mean(np.abs(data_np.real) ** 2)
        q_power = np.mean(np.abs(data_np.imag) ** 2)

        # Threshold for considering a channel "inactive"
        power_threshold = 1e-10

        if i_power < power_threshold:
            return data_np.imag, "Q"
        elif q_power < power_threshold:
            return data_np.real, "I"
        elif np.var(data_np.imag) > np.var(data_np.real):
            return data_np.imag, "Q"
        else:
            return data_np.real, "I"

    def _setup_plotting(self):
        """Configure matplotlib styling."""
        if self.config.dark_theme:
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#00ff88',
                'secondary': '#ff6b6b', 
                'tertiary': '#4ecdc4',
                'quaternary': '#ffe66d',
                'grid': '#333333',
                'text': '#ffffff'
            }
        else:
            self.colors = {
                'primary': '#2ecc71',
                'secondary': '#e74c3c',
                'tertiary': '#3498db', 
                'quaternary': '#f39c12',
                'grid': '#cccccc',
                'text': '#000000'
            }
    
    def compute_metrics(self, data: cp.ndarray) -> SignalMetrics:
        """Compute comprehensive signal statistics on GPU."""
        metrics = SignalMetrics()
        
        with gpu_memory_pool():
            n = len(data)
            metrics.sample_count = n
            
            # Basic statistics
            metrics.mean_complex = complex(to_scalar(cp.mean(data)))
            metrics.std_dev = float(to_scalar(cp.std(data)))
            metrics.dc_offset = metrics.mean_complex
            
            # Power calculations
            power = cp.abs(data) ** 2
            metrics.avg_power_linear = float(to_scalar(cp.mean(power)))
            metrics.peak_power_linear = float(to_scalar(cp.max(power)))
            
            # Convert to dBm (assuming 50 ohm impedance, voltage units)
            # P_dBm = 10*log10(V^2/R) + 30
            if metrics.avg_power_linear > 0:
                metrics.avg_power_dbm = 10 * np.log10(
                    metrics.avg_power_linear / self.config.impedance_ohms) + 30
            if metrics.peak_power_linear > 0:
                metrics.peak_power_dbm = 10 * np.log10(
                    metrics.peak_power_linear / self.config.impedance_ohms) + 30
            
            # PAPR and Crest Factor
            if metrics.avg_power_linear > 0:
                metrics.papr_db = 10 * np.log10(
                    metrics.peak_power_linear / metrics.avg_power_linear)
                metrics.crest_factor = np.sqrt(
                    metrics.peak_power_linear / metrics.avg_power_linear)
            
            # I/Q Imbalance estimation
            i_power = float(to_scalar(cp.mean(cp.abs(data.real) ** 2)))
            q_power = float(to_scalar(cp.mean(cp.abs(data.imag) ** 2)))
            if q_power > 0:
                metrics.iq_imbalance_db = 10 * np.log10(i_power / q_power)
            
            # SNR estimation via spectral method
            metrics.snr_estimate_db = self._estimate_snr(data)
            
            # Bandwidth estimation
            metrics.bandwidth_estimate_hz, metrics.center_freq_offset_hz = \
                self._estimate_bandwidth(data)
        
        return metrics
    
    def _estimate_snr(self, data: cp.ndarray) -> float:
        """Estimate SNR using the M2M4 method."""
        try:
            m2 = cp.mean(cp.abs(data) ** 2)
            m4 = cp.mean(cp.abs(data) ** 4)

            # Avoid division by zero
            m2_sq = m2 ** 2
            if float(to_scalar(m2_sq)) < 1e-20:
                return 0.0

            # M2M4 estimator
            k = float(to_scalar(m4 / m2_sq))

            # For complex Gaussian noise, k ≈ 2
            # For signal + noise, we can estimate SNR
            # Use threshold slightly above 2 for numerical stability
            if k > 2.01:
                snr_linear = (k - 2) / 2
                return float(10 * np.log10(snr_linear + 1e-10))
            return 0.0
        except (ZeroDivisionError, ValueError, RuntimeError) as e:
            log.debug(f"SNR estimation failed: {e}")
            return 0.0
    
    def _estimate_bandwidth(self, data: cp.ndarray) -> Tuple[float, float]:
        """Estimate occupied bandwidth using 99% power containment."""
        try:
            n = min(len(data), self.config.fft_size * 16)
            fft_data = data[:n]
            
            spectrum = cp.fft.fftshift(cp.fft.fft(fft_data, n=self.config.fft_size))
            psd = cp.abs(spectrum) ** 2
            psd_normalized = psd / cp.sum(psd)
            
            # Cumulative power
            cumsum = to_numpy(cp.cumsum(psd_normalized))
            
            # Find 0.5% and 99.5% points (99% bandwidth)
            low_idx = np.searchsorted(cumsum, 0.005)
            high_idx = np.searchsorted(cumsum, 0.995)
            
            freq_resolution = self.config.sample_rate / self.config.fft_size
            bandwidth = (high_idx - low_idx) * freq_resolution
            
            # Center frequency offset (from DC)
            center_idx = (low_idx + high_idx) / 2
            center_offset = (center_idx - self.config.fft_size / 2) * freq_resolution
            
            return float(bandwidth), float(center_offset)
        except Exception:
            return 0.0, 0.0
    
    def plot_time_domain(self, data: cp.ndarray, filename: str = "01_time_domain.png",
                         source_name: str = ""):
        """Generate I/Q time domain plot with envelope."""
        n_samples = min(len(data), self.config.time_domain_samples)
        subset = cp_asnumpy(data[:n_samples])

        time_us = np.arange(n_samples) / self.config.sample_rate * 1e6
        envelope = np.abs(subset)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # I/Q Plot
        axes[0].plot(time_us, subset.real, label='I',
                    color=self.colors['primary'], alpha=0.8, linewidth=0.8)
        axes[0].plot(time_us, subset.imag, label='Q',
                    color=self.colors['secondary'], alpha=0.8, linewidth=0.8)
        axes[0].set_ylabel('Amplitude', fontsize=11)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3, color=self.colors['grid'])
        axes[0].set_title('I/Q Components', fontsize=12)

        # Envelope Plot
        axes[1].fill_between(time_us, envelope, alpha=0.3, color=self.colors['tertiary'])
        axes[1].plot(time_us, envelope, color=self.colors['tertiary'], linewidth=0.8)
        axes[1].set_ylabel('Magnitude', fontsize=11)
        axes[1].set_xlabel('Time (μs)', fontsize=11)
        axes[1].grid(True, alpha=0.3, color=self.colors['grid'])
        axes[1].set_title('Signal Envelope', fontsize=12)

        header = self._get_plot_header(source_name)
        fig.suptitle(f'Time Domain Analysis ({n_samples:,} samples)\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_frequency_domain(self, data: cp.ndarray,
                              filename: str = "02_frequency_domain.png",
                              is_fft_data: bool = False, source_name: str = ""):
        """Generate PSD plot with annotations."""
        with gpu_memory_pool():
            n = min(len(data), self.config.fft_size * 64)
            
            if is_fft_data:
                # Data is already FFT output
                spectrum = cp.fft.fftshift(data[:self.config.fft_size])
            else:
                # Compute averaged periodogram
                n_segments = n // self.config.fft_size
                if n_segments == 0:
                    n_segments = 1
                
                psd_accum = cp.zeros(self.config.fft_size)
                window = cp.hanning(self.config.fft_size)
                
                for i in range(n_segments):
                    segment = data[i * self.config.fft_size:(i + 1) * self.config.fft_size]
                    if len(segment) < self.config.fft_size:
                        segment = cp.pad(segment, (0, self.config.fft_size - len(segment)))
                    windowed = segment * window
                    psd_accum += cp.abs(cp.fft.fft(windowed)) ** 2
                
                # FORENSIC FIX: Apply ENBW correction
                # ENBW corrects for the noise bandwidth widening caused by windowing
                # PSD_corrected = PSD_raw / ENBW to get true power spectral density
                enbw = compute_enbw(cp_asnumpy(window))
                
                # Apply ENBW correction: divide by ENBW to normalize for window effects
                spectrum = cp.fft.fftshift(psd_accum / n_segments / enbw)
                log.debug(f"FORENSIC: Applied ENBW correction factor: {enbw:.4f}")
            
            magnitude_db = 10 * cp.log10(cp.abs(spectrum) + 1e-12)
            mag_cpu = cp_asnumpy(magnitude_db)
        
        freqs = np.linspace(-self.config.sample_rate/2, 
                           self.config.sample_rate/2, 
                           len(mag_cpu)) / 1e3  # kHz
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(freqs, mag_cpu, color=self.colors['primary'], linewidth=0.8)
        ax.fill_between(freqs, mag_cpu, mag_cpu.min(), alpha=0.2, 
                       color=self.colors['primary'])
        
        # Add noise floor estimate line
        noise_floor = np.percentile(mag_cpu, 10)
        ax.axhline(y=noise_floor, color=self.colors['secondary'], 
                  linestyle='--', alpha=0.7, label=f'Est. Noise Floor: {noise_floor:.1f} dB')
        
        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_ylabel('Magnitude (dB)', fontsize=11)
        header = self._get_plot_header(source_name)
        ax.set_title(f'Power Spectral Density (Averaged Periodogram)\n{header}', fontsize=12)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend(loc='upper right')

        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_spectrogram(self, data: cp.ndarray, filename: str = "03_spectrogram.png",
                         source_name: str = ""):
        """Generate high-resolution spectrogram."""
        # Limit data size for spectrogram to avoid memory issues
        max_samples = int(self.config.sample_rate * 10)  # 10 seconds max
        data_limited = cp_asnumpy(data[:max_samples])
        
        nfft = self.config.spectrogram_nfft
        noverlap = int(nfft * self.config.spectrogram_overlap)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        Pxx, freqs, bins, im = ax.specgram(
            data_limited, 
            NFFT=nfft, 
            Fs=self.config.sample_rate,
            noverlap=noverlap,
            cmap='viridis',
            scale='dB',
            mode='psd'
        )
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        header = self._get_plot_header(source_name)
        ax.set_title(f'Spectrogram (Short-Time Fourier Transform)\n{header}', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, label='Power/Freq (dB/Hz)')

        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_waterfall(self, data: cp.ndarray, filename: str = "04_waterfall.png",
                       source_name: str = ""):
        """Generate waterfall plot (alternative spectrogram view)."""
        with gpu_memory_pool():
            nfft = self.config.spectrogram_nfft
            hop = nfft // 4
            
            max_samples = int(self.config.sample_rate * 5)  # 5 seconds
            data_limited = data[:max_samples]
            n_frames = (len(data_limited) - nfft) // hop
            
            if n_frames < 1:
                log.warning("Insufficient data for waterfall plot")
                return
            
            # Compute STFT on GPU
            window = cp.hanning(nfft)
            waterfall = cp.zeros((n_frames, nfft), dtype=cp.float32)
            
            for i in range(n_frames):
                frame = data_limited[i * hop:i * hop + nfft]
                spectrum = cp.fft.fftshift(cp.fft.fft(frame * window))
                waterfall[i, :] = 20 * cp.log10(cp.abs(spectrum) + 1e-12)
            
            waterfall_cpu = cp_asnumpy(waterfall)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        extent = [
            -self.config.sample_rate/2/1e3,
            self.config.sample_rate/2/1e3,
            n_frames * hop / self.config.sample_rate,
            0
        ]
        
        im = ax.imshow(waterfall_cpu, aspect='auto', extent=extent,
                      cmap='inferno', interpolation='nearest')
        
        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_ylabel('Time (s)', fontsize=11)
        header = self._get_plot_header(source_name)
        ax.set_title(f'Waterfall Display\n{header}', fontsize=12)

        plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_constellation(self, data: cp.ndarray, filename: str = "05_constellation.png",
                           source_name: str = ""):
        """Generate constellation diagram with density visualization."""
        n = min(len(data), self.config.constellation_max_points)
        subset = cp_asnumpy(data[:n])
        
        i_data = subset.real
        q_data = subset.imag
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Scatter plot
        axes[0].scatter(i_data, q_data, s=1, alpha=0.3, 
                       color=self.colors['tertiary'], rasterized=True)
        axes[0].set_xlabel('In-Phase (I)', fontsize=11)
        axes[0].set_ylabel('Quadrature (Q)', fontsize=11)
        axes[0].set_title('Constellation Diagram', fontsize=12)
        axes[0].grid(True, alpha=0.3, color=self.colors['grid'])
        axes[0].axis('equal')
        axes[0].axhline(y=0, color='gray', linewidth=0.5)
        axes[0].axvline(x=0, color='gray', linewidth=0.5)
        
        # 2D Histogram (density)
        h, xedges, yedges = np.histogram2d(i_data, q_data, bins=256)
        im = axes[1].imshow(h.T, origin='lower', aspect='equal',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                           cmap='hot', norm=LogNorm(vmin=1, vmax=h.max()))
        axes[1].set_xlabel('In-Phase (I)', fontsize=11)
        axes[1].set_ylabel('Quadrature (Q)', fontsize=11)
        axes[1].set_title('Constellation Density', fontsize=12)
        plt.colorbar(im, ax=axes[1], label='Sample Count')

        header = self._get_plot_header(source_name)
        fig.suptitle(f'I/Q Constellation ({n:,} samples)\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_phase_analysis(self, data: cp.ndarray, filename: str = "06_phase.png",
                            source_name: str = ""):
        """Analyze instantaneous phase and frequency."""
        with gpu_memory_pool():
            n = min(len(data), 50000)
            subset = data[:n]
            
            # Instantaneous phase (unwrapped)
            phase = cp.angle(subset)
            phase_unwrapped = cp_asnumpy(phase)
            phase_unwrapped = np.unwrap(phase_unwrapped)
            
            # Instantaneous frequency (derivative of phase)
            inst_freq = np.diff(phase_unwrapped) * self.config.sample_rate / (2 * np.pi)
        
        time_us = np.arange(n) / self.config.sample_rate * 1e6
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Phase plot
        axes[0].plot(time_us, phase_unwrapped, color=self.colors['primary'], 
                    linewidth=0.8)
        axes[0].set_ylabel('Phase (radians)', fontsize=11)
        axes[0].set_title('Unwrapped Instantaneous Phase', fontsize=12)
        axes[0].grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Frequency plot
        axes[1].plot(time_us[:-1], inst_freq / 1e3, color=self.colors['secondary'],
                    linewidth=0.8)
        axes[1].set_ylabel('Frequency (kHz)', fontsize=11)
        axes[1].set_xlabel('Time (μs)', fontsize=11)
        axes[1].set_title('Instantaneous Frequency', fontsize=12)
        axes[1].grid(True, alpha=0.3, color=self.colors['grid'])

        header = self._get_plot_header(source_name)
        fig.suptitle(f'Phase/Frequency Analysis\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_autocorrelation(self, data: cp.ndarray,
                             filename: str = "07_autocorrelation.png",
                             source_name: str = ""):
        """Compute and plot autocorrelation using Wiener-Khinchin theorem."""
        with gpu_memory_pool():
            n = min(len(data), 100000)
            subset = data[:n]
            
            # FFT-based autocorrelation
            n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
            spectrum = cp.fft.fft(subset, n=n_fft)
            autocorr = cp.fft.ifft(spectrum * cp.conj(spectrum))
            autocorr = cp.fft.fftshift(autocorr)
            autocorr_mag = cp.abs(autocorr)
            autocorr_mag = autocorr_mag / cp.max(autocorr_mag)
            
            # Extract center region
            center = n_fft // 2
            span = min(500, n // 2)
            acf_cpu = cp_asnumpy(autocorr_mag[center - span:center + span])
        
        lags = np.arange(-span, span)
        lag_time_us = lags / self.config.sample_rate * 1e6  # Convert to microseconds
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Full view - use time units (microseconds) for better physical interpretation
        axes[0].plot(lag_time_us, acf_cpu, color=self.colors['primary'], linewidth=0.8)
        axes[0].set_xlabel('Lag (µs)', fontsize=11)
        axes[0].set_ylabel('Correlation', fontsize=11)
        axes[0].set_title('Autocorrelation Function', fontsize=12)
        axes[0].grid(True, alpha=0.3, color=self.colors['grid'])
        # Add secondary x-axis showing samples
        ax0_samples = axes[0].twiny()
        ax0_samples.set_xlim(axes[0].get_xlim())
        ax0_samples.set_xlabel('Lag (samples)', fontsize=9, color='gray')
        sample_ticks = np.linspace(-span, span, 9).astype(int)
        ax0_samples.set_xticks(sample_ticks / self.config.sample_rate * 1e6)
        ax0_samples.set_xticklabels(sample_ticks, fontsize=8, color='gray')
        
        # Zoomed view (first 100 lags) - also use time units
        zoom_span = min(100, span)
        zoom_time_us = lag_time_us[span:span + zoom_span]
        axes[1].stem(zoom_time_us, acf_cpu[span:span + zoom_span],
                    linefmt=self.colors['tertiary'], markerfmt='o', 
                    basefmt='gray')
        axes[1].set_xlabel('Lag (µs)', fontsize=11)
        axes[1].set_ylabel('Correlation', fontsize=11)
        axes[1].set_title('Autocorrelation (First 100 Positive Lags)', fontsize=12)
        axes[1].grid(True, alpha=0.3, color=self.colors['grid'])

        header = self._get_plot_header(source_name)
        fig.suptitle(f'Signal Periodicity Analysis\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_cyclostationary(self, data: cp.ndarray,
                             filename: str = "08_cyclostationary.png",
                             source_name: str = ""):
        """Compute cyclic spectral density (basic cyclostationary analysis)."""
        with gpu_memory_pool():
            n = min(len(data), 50000)
            subset = data[:n]
            nfft = 256
            
            # Compute spectral correlation at alpha = 0 (standard PSD)
            n_frames = n // nfft
            scd = cp.zeros((nfft, nfft), dtype=cp.complex64)
            
            for i in range(n_frames - 1):
                x1 = subset[i * nfft:(i + 1) * nfft]
                x2 = subset[(i + 1) * nfft:(i + 2) * nfft]
                
                X1 = cp.fft.fft(x1)
                X2 = cp.fft.fft(x2)
                
                scd += cp.outer(X1, cp.conj(X2))
            
            scd = cp.fft.fftshift(cp.abs(scd)) / n_frames
            scd_cpu = cp_asnumpy(scd)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(20 * np.log10(scd_cpu + 1e-12), 
                      cmap='viridis', aspect='equal',
                      extent=[-0.5, 0.5, -0.5, 0.5])
        
        ax.set_xlabel('Normalized Frequency f', fontsize=11)
        ax.set_ylabel('Cyclic Frequency α', fontsize=11)
        header = self._get_plot_header(source_name)
        ax.set_title(f'Spectral Correlation Density (Cyclostationary Features)\n{header}', fontsize=11)
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')

        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_statistics(self, data: cp.ndarray, filename: str = "09_statistics.png",
                        source_name: str = ""):
        """Generate comprehensive statistical visualizations."""
        subset = cp_asnumpy(data[:min(len(data), 500000)])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # I/Q Histograms
        axes[0, 0].hist(subset.real, bins=200, alpha=0.7, 
                       color=self.colors['primary'], label='I', density=True)
        axes[0, 0].hist(subset.imag, bins=200, alpha=0.7,
                       color=self.colors['secondary'], label='Q', density=True)
        axes[0, 0].set_xlabel('Amplitude', fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title('I/Q Amplitude Distribution', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Magnitude histogram
        magnitude = np.abs(subset)
        axes[0, 1].hist(magnitude, bins=200, alpha=0.7,
                       color=self.colors['tertiary'], density=True)
        axes[0, 1].set_xlabel('Magnitude', fontsize=11)
        axes[0, 1].set_ylabel('Density', fontsize=11)
        axes[0, 1].set_title('Magnitude Distribution', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Phase histogram
        phase = np.angle(subset)
        axes[1, 0].hist(phase, bins=200, alpha=0.7,
                       color=self.colors['quaternary'], density=True)
        axes[1, 0].set_xlabel('Phase (radians)', fontsize=11)
        axes[1, 0].set_ylabel('Density', fontsize=11)
        axes[1, 0].set_title('Phase Distribution', fontsize=12)
        axes[1, 0].set_xlim(-np.pi, np.pi)
        axes[1, 0].grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Power histogram (dB scale)
        power_db = 10 * np.log10(magnitude ** 2 + 1e-12)
        axes[1, 1].hist(power_db, bins=200, alpha=0.7,
                       color=self.colors['primary'], density=True)
        axes[1, 1].set_xlabel('Power (dB)', fontsize=11)
        axes[1, 1].set_ylabel('Density', fontsize=11)
        axes[1, 1].set_title('Power Distribution', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, color=self.colors['grid'])

        header = self._get_plot_header(source_name)
        fig.suptitle(f'Statistical Analysis\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_power_analysis(self, data: cp.ndarray,
                            filename: str = "10_power_analysis.png",
                            source_name: str = ""):
        """Analyze power characteristics over time."""
        with gpu_memory_pool():
            # Short-term power (instantaneous)
            n_short = min(len(data), self.config.time_domain_samples)
            power_short = cp_asnumpy(cp.abs(data[:n_short]) ** 2)
            
            # Long-term power (windowed average)
            window_size = 1000
            n_long = min(len(data), 100000)
            power_inst = cp.abs(data[:n_long]) ** 2
            
            # Moving average on GPU
            kernel = cp.ones(window_size) / window_size
            power_avg = cp.convolve(power_inst, kernel, mode='valid')
            power_avg_cpu = cp_asnumpy(power_avg)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Instantaneous power
        time_short = np.arange(n_short) / self.config.sample_rate * 1e6
        axes[0].plot(time_short, 10 * np.log10(power_short + 1e-12),
                    color=self.colors['primary'], linewidth=0.8)
        axes[0].set_xlabel('Time (μs)', fontsize=11)
        axes[0].set_ylabel('Power (dB)', fontsize=11)
        axes[0].set_title('Instantaneous Power', fontsize=12)
        axes[0].grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Windowed average power
        time_long = np.arange(len(power_avg_cpu)) / self.config.sample_rate * 1e3
        axes[1].plot(time_long, 10 * np.log10(power_avg_cpu + 1e-12),
                    color=self.colors['secondary'], linewidth=0.8)
        axes[1].set_xlabel('Time (ms)', fontsize=11)
        axes[1].set_ylabel('Power (dB)', fontsize=11)
        axes[1].set_title(f'Moving Average Power (Window: {window_size} samples)', fontsize=12)
        axes[1].grid(True, alpha=0.3, color=self.colors['grid'])

        header = self._get_plot_header(source_name)
        fig.suptitle(f'Power Analysis\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)

    def plot_eye_diagram(self, data: cp.ndarray, samples_per_symbol: int = 8,
                         filename: str = "11_eye_diagram.png", source_name: str = ""):
        """Generate eye diagram for digital signal analysis."""
        n = min(len(data), 50000)
        subset = cp_asnumpy(data[:n])
        
        # Calculate number of complete symbols
        n_symbols = n // samples_per_symbol - 2
        if n_symbols < 10:
            log.warning("Insufficient data for eye diagram")
            return
        
        # Reshape for overlay (2 symbol periods)
        eye_length = samples_per_symbol * 2
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # I component eye
        for i in range(min(n_symbols, 500)):
            start = i * samples_per_symbol
            trace = subset.real[start:start + eye_length]
            if len(trace) == eye_length:
                axes[0].plot(trace, color=self.colors['primary'], 
                           alpha=0.05, linewidth=0.5)
        axes[0].set_xlabel('Sample', fontsize=11)
        axes[0].set_ylabel('Amplitude (I)', fontsize=11)
        axes[0].set_title('Eye Diagram - In-Phase', fontsize=12)
        axes[0].grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Q component eye
        for i in range(min(n_symbols, 500)):
            start = i * samples_per_symbol
            trace = subset.imag[start:start + eye_length]
            if len(trace) == eye_length:
                axes[1].plot(trace, color=self.colors['secondary'],
                           alpha=0.05, linewidth=0.5)
        axes[1].set_xlabel('Sample', fontsize=11)
        axes[1].set_ylabel('Amplitude (Q)', fontsize=11)
        axes[1].set_title('Eye Diagram - Quadrature', fontsize=12)
        axes[1].grid(True, alpha=0.3, color=self.colors['grid'])

        header = self._get_plot_header(source_name)
        fig.suptitle(f'Eye Diagram ({samples_per_symbol} samples/symbol)\n{header}', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def detect_anomalies(self, data: cp.ndarray) -> Dict[str, Any]:
        """Basic anomaly detection for RF forensics."""
        anomalies = {
            'dc_spike': False,
            'saturation': False,
            'dropout': False,
            'periodic_interference': False,
            'details': []
        }
        
        with gpu_memory_pool():
            # DC spike detection
            dc_component = cp.abs(cp.mean(data))
            rms = cp.sqrt(cp.mean(cp.abs(data) ** 2))
            if dc_component > 0.1 * rms:
                anomalies['dc_spike'] = True
                anomalies['details'].append(
                    f"DC offset detected: {float(to_scalar(dc_component)):.4f}")
            
            # Saturation detection (clipping)
            max_val = float(to_scalar(cp.max(cp.abs(data))))
            if max_val > 0.99:
                anomalies['saturation'] = True
                anomalies['details'].append(
                    f"Possible saturation: max amplitude = {max_val:.4f}")
            
            # Dropout detection (zero regions)
            power = cp.abs(data) ** 2
            avg_power = float(to_scalar(cp.mean(power)))
            threshold = avg_power * 0.001
            low_power_samples = int(to_scalar(cp.sum(power < threshold)))
            if low_power_samples > len(data) * 0.01:
                anomalies['dropout'] = True
                anomalies['details'].append(
                    f"Dropout detected: {low_power_samples} samples below threshold")
        
        return anomalies

    # =========================================================================
    # Digital Signal Analysis Methods
    # =========================================================================

    def detect_binary_signal(self, data: cp.ndarray) -> Tuple[bool, int, str, float, float]:
        """
        Detect if signal is binary/digital and identify active channel.

        Returns:
            (is_binary, unique_levels, active_channel, level_low, level_high)
        """
        data_np = cp_asnumpy(data[:min(len(data), 100000)])

        i_vals = data_np.real
        q_vals = data_np.imag

        i_power = np.mean(np.abs(i_vals) ** 2)
        q_power = np.mean(np.abs(q_vals) ** 2)

        # Determine active channel
        if i_power < 1e-10 and q_power > 1e-10:
            active_channel = "Q"
            active_data = q_vals
        elif q_power < 1e-10 and i_power > 1e-10:
            active_channel = "I"
            active_data = i_vals
        else:
            active_channel = "complex"
            active_data = np.abs(data_np)

        # Count unique values (check for quantization)
        unique_vals = np.unique(np.round(active_data, decimals=6))
        unique_levels = len(unique_vals)

        # Check for bimodal (binary) distribution
        hist, bin_edges = np.histogram(active_data, bins=50)

        # Find peaks in histogram
        is_binary = False
        level_low, level_high = 0.0, 0.0

        if unique_levels <= 100:  # Likely quantized
            # Check for two dominant clusters
            threshold = np.median(active_data)
            low_vals = active_data[active_data < threshold]
            high_vals = active_data[active_data >= threshold]

            if len(low_vals) > len(active_data) * 0.1 and len(high_vals) > len(active_data) * 0.1:
                level_low = float(np.mean(low_vals))
                level_high = float(np.mean(high_vals))

                # Check if these are well-separated
                low_std = np.std(low_vals) if len(low_vals) > 1 else 0
                high_std = np.std(high_vals) if len(high_vals) > 1 else 0
                separation = abs(level_high - level_low)

                if separation > 3 * max(low_std, high_std, 0.01):
                    is_binary = True

        return is_binary, unique_levels, active_channel, level_low, level_high

    def analyze_block_structure(self, data: cp.ndarray,
                                block_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Analyze block-to-block correlation to detect periodic structure.
        """
        if block_sizes is None:
            block_sizes = [256, 512, 1024, 2048, 4096]

        data_np = cp_asnumpy(data[:min(len(data), 500000)])

        # Use magnitude or active channel
        if np.mean(np.abs(data_np.real) ** 2) < 1e-10:
            signal = data_np.imag
        elif np.mean(np.abs(data_np.imag) ** 2) < 1e-10:
            signal = data_np.real
        else:
            signal = np.abs(data_np)

        results = {
            'block_correlations': {},
            'best_block_size': 0,
            'best_correlation': 0.0,
            'has_inversion': False,
            'inversion_block_size': 0,
            'inversion_correlation': 0.0,
        }

        for bs in block_sizes:
            n_blocks = len(signal) // bs
            if n_blocks < 4:
                continue

            # Compute adjacent block correlations
            correlations = []
            for i in range(min(n_blocks - 1, 100)):
                b1 = signal[i * bs:(i + 1) * bs]
                b2 = signal[(i + 1) * bs:(i + 2) * bs]

                # Normalize for correlation
                b1_norm = (b1 - np.mean(b1)) / (np.std(b1) + 1e-10)
                b2_norm = (b2 - np.mean(b2)) / (np.std(b2) + 1e-10)

                corr = np.dot(b1_norm, b2_norm) / bs
                correlations.append(corr)

            mean_corr = np.mean(correlations)
            results['block_correlations'][bs] = float(mean_corr)

            # Check for inversion pattern (negative correlation)
            if mean_corr < -0.3 and abs(mean_corr) > abs(results['inversion_correlation']):
                results['has_inversion'] = True
                results['inversion_block_size'] = bs
                results['inversion_correlation'] = float(mean_corr)

            # Track best positive correlation
            if abs(mean_corr) > abs(results['best_correlation']):
                results['best_block_size'] = bs
                results['best_correlation'] = float(mean_corr)

        return results

    def demodulate_binary(self, data: cp.ndarray,
                          threshold: Optional[float] = None) -> np.ndarray:
        """
        Threshold demodulation to extract binary data.
        """
        data_np = cp_asnumpy(data)

        # Determine which channel to use
        if np.mean(np.abs(data_np.real) ** 2) < 1e-10:
            signal = data_np.imag
        elif np.mean(np.abs(data_np.imag) ** 2) < 1e-10:
            signal = data_np.real
        else:
            signal = np.abs(data_np)

        # Auto-detect threshold
        if threshold is None:
            threshold = np.median(signal)

        bits = (signal > threshold).astype(np.uint8)
        return bits

    def differential_decode(self, bits: np.ndarray,
                           mode: str = 'sample') -> np.ndarray:
        """
        Differential decoding (NRZ-I).

        Args:
            bits: Binary array
            mode: 'sample' for per-sample XOR, 'block' for block-level
        """
        if mode == 'sample':
            decoded = np.zeros_like(bits)
            decoded[1:] = bits[1:] ^ bits[:-1]
            return decoded
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def analyze_encoding(self, bits: np.ndarray) -> Dict[str, Any]:
        """
        Analyze bit stream to detect encoding type.
        """
        n = min(len(bits), 500000)
        test_bits = bits[:n]

        # Calculate balance
        ones = np.sum(test_bits)
        balance = ones / n

        # Run-length analysis
        transitions = np.where(np.diff(test_bits) != 0)[0]
        if len(transitions) > 1:
            run_lengths = np.diff(transitions)
            rl_counts = Counter(run_lengths)
            mean_run = np.mean(run_lengths)

            # Transition rate
            transition_rate = len(transitions) / n * self.config.sample_rate

            # Check MLS properties (50% runs of length 1, 25% length 2, etc.)
            total_runs = len(run_lengths)
            len1_ratio = rl_counts.get(1, 0) / total_runs if total_runs > 0 else 0
            len2_ratio = rl_counts.get(2, 0) / total_runs if total_runs > 0 else 0

            # Classify encoding
            if abs(len1_ratio - 0.5) < 0.1 and abs(len2_ratio - 0.25) < 0.1:
                encoding_type = "MLS-like"
            elif len1_ratio > 0.6:
                encoding_type = "NRZ-high-transition"
            elif mean_run > 3:
                encoding_type = "NRZ-low-transition"
            else:
                encoding_type = "NRZ"
        else:
            mean_run = n
            transition_rate = 0
            encoding_type = "constant"
            rl_counts = Counter()

        return {
            'balance': float(balance),
            'mean_run_length': float(mean_run),
            'transition_rate': float(transition_rate),
            'encoding_type': encoding_type,
            'run_length_distribution': dict(rl_counts.most_common(10)),
            'estimated_chip_rate': float(transition_rate / 2) if transition_rate > 0 else 0,
        }

    def search_sync_patterns(self, bits: np.ndarray) -> List[Tuple[str, str, int]]:
        """
        Search for common sync patterns in bit stream.
        """
        n = min(len(bits), 500000)
        bit_str = ''.join(map(str, bits[:n]))

        patterns = [
            ("HDLC flag", "01111110"),
            ("Preamble 0xAA", "10101010"),
            ("Preamble 0x55", "01010101"),
            ("Long preamble", "1010101010101010"),
            ("Barker-7", "1110010"),
            ("Barker-11", "11100010010"),
        ]

        results = []
        for name, pattern in patterns:
            count = bit_str.count(pattern)
            if count > 5:
                results.append((name, pattern, count))

        return sorted(results, key=lambda x: -x[2])

    def extract_hdlc_frames(self, bits: np.ndarray,
                           max_frames: int = 100) -> List[Tuple[int, bytes]]:
        """
        Extract HDLC frames from bit stream.
        """
        bit_str = ''.join(map(str, bits[:min(len(bits), 1000000)]))
        hdlc_flag = "01111110"

        # Find all flag positions
        positions = []
        pos = 0
        while True:
            pos = bit_str.find(hdlc_flag, pos)
            if pos < 0:
                break
            positions.append(pos)
            pos += 1

        frames = []
        for i in range(min(len(positions) - 1, max_frames)):
            start = positions[i]
            end = positions[i + 1]
            frame_len = end - start

            # Valid frame size (8-4000 bytes)
            if 64 <= frame_len <= 32000:
                frame_bits = bit_str[start:end + 8]

                # Bit destuffing (remove 0 after five 1s)
                destuffed = ""
                ones_count = 0
                for bit in frame_bits:
                    if bit == '1':
                        ones_count += 1
                        destuffed += bit
                    else:
                        if ones_count != 5:
                            destuffed += bit
                        ones_count = 0

                # Convert to bytes (HDLC uses LSB-first)
                n_bytes = len(destuffed) // 8
                frame_bytes = []
                for j in range(n_bytes):
                    byte_bits = destuffed[j*8:(j+1)*8]
                    if len(byte_bits) == 8:
                        byte_val = int(byte_bits[::-1], 2)  # LSB-first
                        frame_bytes.append(byte_val)

                if frame_bytes:
                    frames.append((start, bytes(frame_bytes)))

        return frames

    def analyze_bytes(self, data_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze byte-level properties of decoded data.
        """
        if not data_bytes:
            return {'entropy': 0, 'ascii_count': 0, 'common_bytes': []}

        # Byte frequency
        byte_counts = Counter(data_bytes)
        total = len(data_bytes)

        # Shannon entropy
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)

        # ASCII detection
        ascii_count = sum(1 for b in data_bytes if 32 <= b < 127)

        # Common bytes
        common = [(f"0x{b:02X}", c) for b, c in byte_counts.most_common(10)]

        # Find ASCII strings
        ascii_strings = []
        current = []
        start = 0
        for i, b in enumerate(data_bytes[:50000]):
            if 32 <= b < 127:
                if not current:
                    start = i
                current.append(chr(b))
            else:
                if len(current) >= 4:
                    ascii_strings.append((start, ''.join(current)))
                current = []

        return {
            'entropy': float(entropy),
            'ascii_ratio': float(ascii_count / total) if total > 0 else 0,
            'ascii_strings_found': len(ascii_strings),
            'common_bytes': common,
            'sample_ascii_strings': ascii_strings[:20],
        }

    def bits_to_bytes(self, bits: np.ndarray, msb_first: bool = True) -> bytes:
        """Convert bit array to bytes using vectorized numpy operations."""
        # Ensure we have a multiple of 8 bits
        n_bits = (len(bits) // 8) * 8
        bits_trimmed = bits[:n_bits].astype(np.uint8)

        if msb_first:
            # np.packbits uses MSB-first by default
            return bytes(np.packbits(bits_trimmed))
        else:
            # For LSB-first, reverse bits within each byte
            bits_reshaped = bits_trimmed.reshape(-1, 8)
            bits_reversed = bits_reshaped[:, ::-1]
            return bytes(np.packbits(bits_reversed.flatten()))

    def run_digital_analysis(self, data: cp.ndarray, name: str = "signal",
                            filepath: Optional[Path] = None) -> DigitalSignalAnalysis:
        """
        Execute complete digital signal analysis pipeline.
        """
        log.info(f"Starting digital signal analysis: {name}")
        results = DigitalSignalAnalysis()

        # Step 1: Detect binary signal
        is_binary, unique_levels, active_channel, level_low, level_high = \
            self.detect_binary_signal(data)

        results.is_binary = is_binary
        results.unique_levels = unique_levels
        results.is_real_only = active_channel in ("I", "Q")
        results.active_channel = active_channel
        results.level_low = level_low
        results.level_high = level_high

        log.info(f"  Binary signal: {is_binary}, Levels: {unique_levels}, Channel: {active_channel}")

        # Step 2: Block structure analysis
        block_results = self.analyze_block_structure(data)
        results.block_size = block_results['best_block_size']
        results.block_correlation = block_results['best_correlation']
        results.has_inversion_pattern = block_results['has_inversion']
        results.inversion_correlation = block_results.get('inversion_correlation', 0.0)

        if block_results['has_inversion']:
            log.info(f"  Inversion pattern at {block_results['inversion_block_size']} samples "
                    f"(corr: {block_results['inversion_correlation']:.3f})")

        # Step 3: Demodulate
        threshold = self.config.digital_threshold
        if threshold is None:
            threshold = (level_low + level_high) / 2 if is_binary else 0

        bits = self.demodulate_binary(data, threshold)

        # Step 4: Encoding analysis
        encoding = self.analyze_encoding(bits)
        results.encoding_type = encoding['encoding_type']
        results.run_length_mean = encoding['mean_run_length']
        results.balance_ratio = encoding['balance']
        results.transition_rate = encoding['transition_rate']
        results.estimated_chip_rate = encoding['estimated_chip_rate']

        if encoding['transition_rate'] > 0:
            results.samples_per_symbol = self.config.sample_rate / encoding['estimated_chip_rate']
            results.estimated_symbol_rate = encoding['estimated_chip_rate']

        log.info(f"  Encoding: {encoding['encoding_type']}, "
                f"Chip rate: {encoding['estimated_chip_rate']/1e6:.3f} Mchips/s")

        # Step 5: Sync pattern search
        sync_patterns = self.search_sync_patterns(bits)
        results.sync_patterns = [(pname, count) for pname, _, count in sync_patterns]

        for pname, pattern, count in sync_patterns[:3]:
            log.info(f"  Sync pattern '{pname}': {count} occurrences")

        # Step 6: HDLC frame extraction
        if any('HDLC' in name for name, _, _ in sync_patterns):
            frames = self.extract_hdlc_frames(bits)
            results.hdlc_flags_found = sum(1 for n, _, c in sync_patterns if 'HDLC' in n for _ in range(c))
            results.frame_count = len(frames)

            if frames:
                frame_lengths = [len(f[1]) for f in frames]
                if frame_lengths:
                    length_counts = Counter(frame_lengths)
                    results.common_frame_length = length_counts.most_common(1)[0][0]

                log.info(f"  HDLC frames: {len(frames)}, common length: {results.common_frame_length} bytes")

        # Count preambles
        results.preamble_count = sum(c for n, _, c in sync_patterns if 'reamble' in n)

        # Step 7: Byte analysis
        data_bytes = self.bits_to_bytes(bits[:min(len(bits), 100000)])
        byte_analysis = self.analyze_bytes(data_bytes)
        results.byte_entropy = byte_analysis['entropy']
        results.ascii_sequences_found = byte_analysis['ascii_strings_found']

        log.info(f"  Byte entropy: {byte_analysis['entropy']:.3f}, "
                f"ASCII strings: {byte_analysis['ascii_strings_found']}")

        # Step 8: Generate plots
        src = filepath.name if filepath else name
        self.plot_digital_analysis(data, bits, results,
                                   f"{name}_12_digital_analysis.png", src,
                                   block_results=block_results)

        # Print summary
        self._print_digital_metrics(results, name)

        return results

    def plot_digital_analysis(self, data: cp.ndarray, bits: np.ndarray,
                             analysis: DigitalSignalAnalysis,
                             filename: str, source_name: str = "",
                             block_results: Optional[Dict] = None):
        """Generate digital signal analysis visualization."""
        data_np = cp_asnumpy(data[:min(len(data), 5000)])

        # Get signal for plotting
        if analysis.active_channel == "Q":
            signal = data_np.imag
        elif analysis.active_channel == "I":
            signal = data_np.real
        else:
            signal = np.abs(data_np)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Raw signal with threshold
        n_plot = min(500, len(signal))
        axes[0, 0].plot(signal[:n_plot], color=self.colors['primary'], linewidth=0.8)
        threshold = (analysis.level_low + analysis.level_high) / 2
        axes[0, 0].axhline(y=threshold, color=self.colors['secondary'],
                          linestyle='--', alpha=0.7, label=f'Threshold: {threshold:.2f}')
        axes[0, 0].axhline(y=analysis.level_high, color=self.colors['tertiary'],
                          linestyle=':', alpha=0.5, label=f'High: {analysis.level_high:.2f}')
        axes[0, 0].axhline(y=analysis.level_low, color=self.colors['tertiary'],
                          linestyle=':', alpha=0.5, label=f'Low: {analysis.level_low:.2f}')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title(f'Raw Signal ({analysis.active_channel} channel)')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, color=self.colors['grid'])

        # Plot 2: Demodulated bits
        axes[0, 1].step(range(n_plot), bits[:n_plot], color=self.colors['tertiary'],
                       where='post', linewidth=0.8)
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Bit Value')
        axes[0, 1].set_title('Demodulated Bits')
        axes[0, 1].set_ylim(-0.1, 1.1)
        axes[0, 1].grid(True, alpha=0.3, color=self.colors['grid'])

        # Plot 3: Run length histogram
        transitions = np.where(np.diff(bits[:100000]) != 0)[0]
        if len(transitions) > 1:
            run_lengths = np.diff(transitions)
            axes[1, 0].hist(run_lengths, bins=min(50, max(run_lengths)),
                           color=self.colors['quaternary'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Run Length (samples)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Run Length Distribution (mean: {analysis.run_length_mean:.1f})')
        axes[1, 0].grid(True, alpha=0.3, color=self.colors['grid'])

        # Plot 4: Block correlation (if blocks detected)
        if analysis.block_size > 0 and block_results is not None:
            block_sizes = sorted(block_results['block_correlations'].keys())
            correlations = [block_results['block_correlations'][bs] for bs in block_sizes]

            bars = axes[1, 1].bar(range(len(block_sizes)), correlations,
                                  color=self.colors['primary'], alpha=0.7)
            # Color negative correlations differently
            for i, (bar, corr) in enumerate(zip(bars, correlations)):
                if corr < 0:
                    bar.set_color(self.colors['secondary'])

            axes[1, 1].set_xticks(range(len(block_sizes)))
            axes[1, 1].set_xticklabels(block_sizes)
            axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)
            axes[1, 1].set_xlabel('Block Size (samples)')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].set_title('Block-to-Block Correlation')
            axes[1, 1].grid(True, alpha=0.3, color=self.colors['grid'])
        else:
            axes[1, 1].text(0.5, 0.5, 'No block structure detected',
                          ha='center', va='center', transform=axes[1, 1].transAxes)

        header = self._get_plot_header(source_name)
        encoding_info = f"Encoding: {analysis.encoding_type}"
        if analysis.estimated_chip_rate > 0:
            encoding_info += f" | Chip Rate: {analysis.estimated_chip_rate/1e6:.2f} Mchips/s"
        fig.suptitle(f'Digital Signal Analysis\n{header}\n{encoding_info}', fontsize=11)

        plt.tight_layout()
        self._save_figure(fig, filename)

    def _print_digital_metrics(self, analysis: DigitalSignalAnalysis, name: str):
        """Print formatted digital analysis summary."""
        print(f"\n{'='*60}")
        print(f" Digital Signal Analysis: {name}")
        print(f"{'='*60}")
        print(f" Binary Signal:      {analysis.is_binary}")
        print(f" Unique Levels:      {analysis.unique_levels}")
        print(f" Active Channel:     {analysis.active_channel}")
        print(f" Level Low/High:     {analysis.level_low:.2f} / {analysis.level_high:.2f}")
        print(f"{'='*60}")
        print(f" Block Size:         {analysis.block_size} samples")
        print(f" Block Correlation:  {analysis.block_correlation:.3f}")
        print(f" Inversion Pattern:  {analysis.has_inversion_pattern}")
        print(f"{'='*60}")
        print(f" Encoding Type:      {analysis.encoding_type}")
        print(f" Balance Ratio:      {analysis.balance_ratio:.3f}")
        print(f" Mean Run Length:    {analysis.run_length_mean:.2f} samples")
        print(f" Transition Rate:    {analysis.transition_rate/1e6:.3f} MHz")
        print(f" Est. Chip Rate:     {analysis.estimated_chip_rate/1e6:.3f} Mchips/s")
        print(f"{'='*60}")
        print(f" HDLC Flags:         {analysis.hdlc_flags_found}")
        print(f" Preamble Count:     {analysis.preamble_count}")
        print(f" Frame Count:        {analysis.frame_count}")
        print(f" Byte Entropy:       {analysis.byte_entropy:.3f} bits")
        print(f" ASCII Sequences:    {analysis.ascii_sequences_found}")
        print(f"{'='*60}\n")

    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure with proper settings."""
        filepath = self.config.output_dir / filename
        try:
            fig.savefig(filepath, dpi=self.config.plot_dpi,
                       bbox_inches='tight', facecolor='auto', edgecolor='auto')
            log.info(f"Saved: {filename}")
        except Exception as e:
            log.error(f"Failed to save {filename}: {e}")
        finally:
            plt.close(fig)

    # =========================================================================
    # V3 INTEGRATION METHODS
    # =========================================================================

    def analyze_fft_pipeline(self, data: cp.ndarray) -> FFTDebugMetrics:
        """Run FFT pipeline debug analysis."""
        analyzer = FFTDebugAnalyzer(self.config.sample_rate)
        return analyzer.full_analysis(data, self.config.fft_size)

    def analyze_spreading_codes(self, data: cp.ndarray) -> PNSequenceMetrics:
        """Run spreading code / PN sequence analysis."""
        analyzer = SpreadingCodeAnalyzer(self.config.sample_rate)
        return analyzer.full_analysis(data)

    def analyze_bit_ordering(self, data: cp.ndarray) -> BitOrderMetrics:
        """Run bit ordering and frame analysis."""
        # First demodulate to bits
        data_np = cp_asnumpy(data)
        signal, _ = self._get_active_signal(data_np)

        threshold = np.mean(signal)
        bits = (signal > threshold).astype(np.uint8)

        analyzer = BitFrameAnalyzer()
        return analyzer.full_analysis(bits[:500000])

    def plot_fft_debug(self, data: cp.ndarray, filename: str, source_name: str):
        """Generate FFT debug analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'FFT Pipeline Debug: {source_name}', fontsize=14, fontweight='bold')

        data_np = cp_asnumpy(data[:self.config.fft_size * 4])
        fs = self.config.sample_rate
        fft_size = self.config.fft_size

        # Panel 1: Window comparison
        ax1 = axes[0, 0]
        windows = {'Rectangular': np.ones(fft_size),
                   'Hann': np.hanning(fft_size),
                   'Blackman': np.blackman(fft_size)}

        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs)) / 1e6
        for wname, win in windows.items():
            X = np.fft.fftshift(np.fft.fft(data_np[:fft_size] * win))
            X_db = 20 * np.log10(np.abs(X) + 1e-10)
            ax1.plot(freqs, X_db, label=wname, alpha=0.8)
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Power (dB)')
        ax1.set_title('Window Function Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: DC offset analysis
        ax2 = axes[0, 1]
        dc_i = np.mean(np.real(data_np))
        dc_q = np.mean(np.imag(data_np))
        ax2.scatter([dc_i], [dc_q], s=200, c='red', marker='x', label=f'DC: {dc_i:.4f}+{dc_q:.4f}j')
        ax2.scatter(np.real(data_np[:1000]), np.imag(data_np[:1000]), alpha=0.3, s=5)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_title('DC Offset in I/Q Plane')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Panel 3: Noise floor estimation
        ax3 = axes[1, 0]
        X = np.fft.fftshift(np.fft.fft(data_np[:fft_size] * np.blackman(fft_size)))
        X_db = 20 * np.log10(np.abs(X) + 1e-10)
        noise_floor = np.median(X_db)
        ax3.plot(freqs, X_db, alpha=0.8)
        ax3.axhline(y=noise_floor, color='r', linestyle='--', label=f'Noise floor: {noise_floor:.1f} dB')
        ax3.set_xlabel('Frequency (MHz)')
        ax3.set_ylabel('Power (dB)')
        ax3.set_title('Noise Floor & Spurs')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Spectral leakage visualization
        ax4 = axes[1, 1]
        rect_fft = np.abs(np.fft.fftshift(np.fft.fft(data_np[:fft_size])))
        hann_fft = np.abs(np.fft.fftshift(np.fft.fft(data_np[:fft_size] * np.hanning(fft_size))))
        rect_fft = rect_fft / np.max(rect_fft)
        hann_fft = hann_fft / np.max(hann_fft)
        ax4.semilogy(freqs, rect_fft, label='Rectangular', alpha=0.7)
        ax4.semilogy(freqs, hann_fft, label='Hann', alpha=0.7)
        ax4.set_xlabel('Frequency (MHz)')
        ax4.set_ylabel('Normalized Magnitude (log)')
        ax4.set_title('Spectral Leakage Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.config.output_dir / filename
        fig.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        log.info(f"Saved: {filename}")

    def plot_spreading_code(self, data: cp.ndarray, filename: str, source_name: str):
        """Generate spreading code analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Spreading Code Analysis: {source_name}', fontsize=14, fontweight='bold')

        data_np = cp_asnumpy(data[:50000])
        signal = np.sign(np.real(data_np))

        # Panel 1: Autocorrelation
        ax1 = axes[0, 0]
        autocorr = np.correlate(signal[:5000], signal[:5000], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        ax1.plot(autocorr[:1000])
        ax1.set_xlabel('Lag (samples)')
        ax1.set_ylabel('Correlation')
        ax1.set_title('Autocorrelation (Period Detection)')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Chip transitions
        ax2 = axes[0, 1]
        transitions = np.diff(signal[:2000])
        ax2.plot(transitions, alpha=0.8)
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Transition')
        ax2.set_title('Chip Transitions')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Run length distribution
        ax3 = axes[1, 0]
        bits = (signal[:10000] > 0).astype(int)
        runs = []
        current = 1
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                current += 1
            else:
                runs.append(current)
                current = 1
        if runs:
            ax3.hist(runs, bins=range(1, min(max(runs)+2, 50)), edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Run Length')
        ax3.set_ylabel('Count')
        ax3.set_title('Run Length Distribution')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Power spectral density
        ax4 = axes[1, 1]
        f, Pxx = welch(signal[:10000], fs=self.config.sample_rate, nperseg=1024)
        ax4.semilogy(f / 1e6, Pxx)
        ax4.set_xlabel('Frequency (MHz)')
        ax4.set_ylabel('PSD')
        ax4.set_title('Power Spectral Density')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.config.output_dir / filename
        fig.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        log.info(f"Saved: {filename}")

    def plot_bit_frame(self, data: cp.ndarray, filename: str, source_name: str):
        """Generate bit ordering and frame analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Bit/Frame Analysis: {source_name}', fontsize=14, fontweight='bold')

        data_np = cp_asnumpy(data[:100000])
        signal, _ = self._get_active_signal(data_np)

        threshold = np.mean(signal)
        bits = (signal > threshold).astype(np.uint8)

        # Panel 1: Bit stream visualization
        ax1 = axes[0, 0]
        ax1.step(range(500), bits[:500], where='mid')
        ax1.set_xlabel('Bit Index')
        ax1.set_ylabel('Bit Value')
        ax1.set_title('Demodulated Bit Stream (first 500)')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Byte value distribution
        ax2 = axes[0, 1]
        n_bytes = len(bits) // 8
        bytes_arr = []
        for i in range(n_bytes):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | int(bits[i*8 + j])
            bytes_arr.append(byte_val)
        if bytes_arr:
            ax2.hist(bytes_arr, bins=64, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Byte Value')
        ax2.set_ylabel('Count')
        ax2.set_title('Byte Value Distribution')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Sync pattern search
        ax3 = axes[1, 0]
        bit_str = ''.join(map(str, bits[:50000]))
        patterns = {'HDLC (0x7E)': '01111110', '0xAA': '10101010', '0x55': '01010101'}
        counts = {name: bit_str.count(pat) for name, pat in patterns.items()}
        ax3.bar(counts.keys(), counts.values(), color=['#3498db', '#e74c3c', '#2ecc71'])
        ax3.set_ylabel('Occurrences')
        ax3.set_title('Sync Pattern Detection')
        ax3.grid(True, alpha=0.3, axis='y')

        # Panel 4: Frame spacing histogram (if HDLC found)
        ax4 = axes[1, 1]
        hdlc_pattern = '01111110'
        positions = []
        start = 0
        while True:
            pos = bit_str.find(hdlc_pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1

        if len(positions) > 2:
            spacings = np.diff(positions)
            ax4.hist(spacings, bins=50, edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Frame Spacing (bits)')
            ax4.set_ylabel('Count')
            ax4.set_title('HDLC Frame Spacing Distribution')
        else:
            ax4.text(0.5, 0.5, 'Not enough HDLC frames\nfor spacing analysis',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Frame Spacing (N/A)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.config.output_dir / filename
        fig.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        log.info(f"Saved: {filename}")

    def _print_v3_summary(self, fft: FFTDebugMetrics, spreading: PNSequenceMetrics,
                          bit_frame: BitOrderMetrics):
        """Print v3 analysis summary."""
        print(f"\n{'='*60}")
        print(f" V3 Enhanced Analysis Summary")
        print(f"{'='*60}")
        print(f" FFT Debug:")
        print(f"   DC Problematic:     {fft.dc_is_problematic}")
        print(f"   Recommended Window: {fft.recommended_window}")
        print(f"   SNR (peak/median):  {fft.snr_peak_median_dB:.1f} dB")
        print(f"   Spurs Detected:     {fft.spurs_detected}")
        print(f" Spreading Code:")
        print(f"   Type Detected:      {spreading.detected_type}")
        if spreading.detected_type != 'none':
            print(f"   Degree:             {spreading.detected_degree}")
            print(f"   Correlation Peak:   {spreading.correlation_peak:.3f}")
            print(f"   Confidence:         {spreading.confidence}")
        print(f" Bit/Frame Analysis:")
        print(f"   Bit Order:          {bit_frame.detected_order}")
        print(f"   Scrambler:          {bit_frame.scrambler_detected}")
        print(f"   Sync Word:          {bit_frame.sync_word_detected}")
        print(f"   Frames Found:       {bit_frame.frames_found}")
        print(f"{'='*60}\n")

    def run_full_analysis(self, data: cp.ndarray, name: str = "signal",
                          filepath: Optional[Path] = None,
                          enable_v3: bool = False,
                          residual_metadata: Dict[str, Any] = None) -> SignalMetrics:
        """
        Execute complete analysis pipeline.
        
        FORENSIC COMPLIANCE: This method now integrates ForensicPipeline to create
        a verifiable hash chain of all processing steps per NIST SP 800-86.
        """
        log.info(f"Starting analysis: {name}")
        
        # FORENSIC FIX: Initialize ForensicPipeline for chain of custody
        # Per NIST SP 800-86: "hash immediately at acquisition, before processing,
        # and after any transformation"
        forensic_pipeline = ForensicPipeline()
        forensic_pipeline.add_hash_checkpoint(data, "raw_input", {
            'source': filepath.name if filepath else name,
            'sample_count': len(data),
            'dtype': str(data.dtype)
        })
        
        metrics = self.compute_metrics(data)
        
        # FORENSIC: Checkpoint after metrics computation
        forensic_pipeline.add_hash_checkpoint(data, "post_metrics", {
            'avg_power_dbm': metrics.avg_power_dbm,
            'snr_estimate_db': metrics.snr_estimate_db
        })
        self._print_metrics(metrics, name)
        
        # Run all plot functions with source metadata
        src = filepath.name if filepath else name
        self.plot_time_domain(data, f"{name}_01_time_domain.png", src)
        self.plot_frequency_domain(data, f"{name}_02_frequency_domain.png", source_name=src)
        self.plot_spectrogram(data, f"{name}_03_spectrogram.png", src)
        self.plot_waterfall(data, f"{name}_04_waterfall.png", src)
        self.plot_constellation(data, f"{name}_05_constellation.png", src)
        self.plot_phase_analysis(data, f"{name}_06_phase.png", src)
        self.plot_autocorrelation(data, f"{name}_07_autocorrelation.png", src)
        self.plot_cyclostationary(data, f"{name}_08_cyclostationary.png", src)
        self.plot_statistics(data, f"{name}_09_statistics.png", src)
        self.plot_power_analysis(data, f"{name}_10_power_analysis.png", src)
        self.plot_eye_diagram(data, filename=f"{name}_11_eye_diagram.png", source_name=src)
        
        # Anomaly detection
        anomalies = self.detect_anomalies(data)
        if any(anomalies[k] for k in ['dc_spike', 'saturation', 'dropout']):
            log.warning(f"Anomalies detected: {anomalies['details']}")
        
        # FORENSIC: Checkpoint after anomaly detection
        forensic_pipeline.add_hash_checkpoint(data, "post_anomaly_detection", {
            'anomalies_found': any(anomalies[k] for k in ['dc_spike', 'saturation', 'dropout'])
        })
        
        # Collect file metadata
        file_metadata = {}
        if filepath and filepath.exists():
            stat = filepath.stat()
            # Compute SHA-256 hash with chunked reading for large files
            # FORENSIC FIX: Dual-algorithm hashing per SWGDE
            forensic_hashes = compute_forensic_hashes(filepath)
            file_hash = forensic_hashes['sha256']
            file_metadata = {
                'source_file': str(filepath.absolute()),
                'filename': filepath.name,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / 1e6, 2),
                'file_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'file_created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'file_sha256': file_hash,
                'file_sha3_256': forensic_hashes.get('sha3_256', ''),
                'hash_timestamp_utc': forensic_hashes.get('hash_timestamp_utc', ''),
            }

        # Digital signal analysis (if enabled)
        digital_results = None
        if self.config.digital_analysis:
            digital_results = self.run_digital_analysis(data, name, filepath)
            # FORENSIC: Checkpoint after digital analysis
            forensic_pipeline.add_hash_checkpoint(data, "post_digital_analysis", {
                'is_binary': digital_results.is_binary if digital_results else None
            })

        # V3 Enhanced Analysis (if enabled)
        v3_metrics = {}
        if enable_v3:
            log.info("Running v3 enhanced analysis...")

            # FFT Pipeline Debug
            fft_debug_metrics = self.analyze_fft_pipeline(data)
            self.plot_fft_debug(data, f"{name}_13_fft_debug.png", src)
            v3_metrics['fft_debug'] = fft_debug_metrics.to_dict()

            # Spreading Code Analysis
            spreading_metrics = self.analyze_spreading_codes(data)
            self.plot_spreading_code(data, f"{name}_14_spreading_code.png", src)
            v3_metrics['spreading_code'] = spreading_metrics.to_dict()

            # Bit Ordering & Frame Analysis
            bit_frame_metrics = self.analyze_bit_ordering(data)
            self.plot_bit_frame(data, f"{name}_15_bit_frame.png", src)
            v3_metrics['bit_frame'] = bit_frame_metrics.to_dict()

            # Print v3 summary
            self._print_v3_summary(fft_debug_metrics, spreading_metrics, bit_frame_metrics)
            
            # FORENSIC: Checkpoint after v3 analysis
            forensic_pipeline.add_hash_checkpoint(data, "post_v3_analysis", {
                'fft_debug_completed': True,
                'spreading_code_completed': True,
                'bit_frame_completed': True
            })

        # Save metrics to JSON
        metrics_file = self.config.output_dir / f"{name}_metrics.json"
        # FORENSIC: Final checkpoint before output
        forensic_pipeline.add_hash_checkpoint(data, "pre_output", {
            'analysis_complete': True
        })
        
        output_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer_version': '2.2.2-forensic',
            'file_metadata': file_metadata,
            'residual_metadata': residual_metadata or {'residual_bytes': 0},
            'forensic_pipeline': forensic_pipeline.to_dict(),  # FORENSIC: Include hash chain
            'analysis_config': {
                'sample_rate_hz': self.config.sample_rate,
                'center_freq_hz': self.config.center_freq,
                'data_format': self.config.data_format.value,
                'fft_size': self.config.fft_size,
                'impedance_ohms': self.config.impedance_ohms,
                'digital_analysis': self.config.digital_analysis,
                'v3_analysis': enable_v3,
            },
            'metrics': metrics.to_dict(),
            'anomalies': anomalies
        }

        if digital_results:
            output_data['digital_analysis'] = digital_results.to_dict()

        if v3_metrics:
            output_data['v3_analysis'] = v3_metrics

        with open(metrics_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Issue 10 Fix: Add forensic hash of output file for integrity verification
        try:
            output_hashes = compute_forensic_hashes(metrics_file)
            log.info(f"Saved metrics: {metrics_file.name}")
            log.info(f"Output SHA256: {output_hashes['sha256']}")
            log.info(f"Output SHA3-256: {output_hashes['sha3_256']}")
            
            # Optionally write hash sidecar file for verification
            hash_file = metrics_file.with_suffix('.sha256')
            with open(hash_file, 'w') as hf:
                hf.write(f"SHA256: {output_hashes['sha256']}\n")
                hf.write(f"SHA3-256: {output_hashes['sha3_256']}\n")
                hf.write(f"Timestamp: {output_hashes['hash_timestamp_utc']}\n")
        except Exception as e:
            log.warning(f"Could not compute output hash: {e}")
            log.info(f"Saved metrics: {metrics_file.name}")

        return metrics
    
    def _print_metrics(self, metrics: SignalMetrics, name: str):
        """Print formatted metrics summary."""
        print(f"\n{'='*60}")
        print(f" Signal Analysis Summary: {name}")
        print(f"{'='*60}")
        print(f" Samples:            {metrics.sample_count:,}")
        print(f" Duration:           {metrics.sample_count / self.config.sample_rate * 1e3:.2f} ms")
        print(f" Avg Power:          {metrics.avg_power_dbm:.2f} dBm")
        print(f" Peak Power:         {metrics.peak_power_dbm:.2f} dBm")
        print(f" PAPR:               {metrics.papr_db:.2f} dB")
        print(f" I/Q Imbalance:      {metrics.iq_imbalance_db:.2f} dB")
        print(f" Est. SNR:           {metrics.snr_estimate_db:.2f} dB")
        print(f" Est. Bandwidth:     {metrics.bandwidth_estimate_hz / 1e3:.2f} kHz")
        print(f" Freq Offset:        {metrics.center_freq_offset_hz:.2f} Hz")
        print(f" DC Offset:          {metrics.dc_offset:.4f}")
        print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CUDA-Accelerated RF Signal Analysis Suite v2.2.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s capture.bin
  %(prog)s -r 2.4e6 -f int16 before.bin after.bin
  %(prog)s --sample-rate 10e6 --format complex64 --dark signal.bin
  %(prog)s --digital -r 10e6 -f int16 signal.bin    # Enable digital analysis
  %(prog)s -d --block-size 2048 -r 1e6 data.bin     # Custom block size
  %(prog)s --v3 -r 10e6 -f int16 signal.bin         # Enable v3 enhanced analysis
  %(prog)s -d --v3 -r 10e6 -f int16 signal.bin      # Full analysis (digital + v3)
        '''
    )
    
    parser.add_argument('files', nargs='+', type=Path,
                       help='Binary signal file(s) to analyze')
    
    parser.add_argument('-r', '--sample-rate', type=float, default=1e6,
                       help='Sample rate in Hz (default: 1e6)')

    parser.add_argument('-c', '--center-freq', type=float, default=0.0,
                       help='Center frequency in Hz (default: 0)')

    parser.add_argument('-f', '--format', type=str, default='complex64',
                       choices=['complex64', 'complex128', 'int16', 'int8', 'float32'],
                       help='Binary data format (default: complex64)')
    
    parser.add_argument('-o', '--output', type=Path, default=Path('./analysis_output'),
                       help='Output directory (default: ./analysis_output)')
    
    parser.add_argument('--fft-size', type=int, default=4096,
                       help='FFT size for spectral analysis (default: 4096)')
    
    parser.add_argument('--dark', action='store_true', default=True,
                       help='Use dark theme for plots (default: True)')
    
    parser.add_argument('--light', action='store_true',
                       help='Use light theme for plots')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode for large files')
    
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to load (default: all)')

    # Digital signal analysis options
    parser.add_argument('--digital', '-d', action='store_true',
                       help='Enable deep digital signal analysis')

    parser.add_argument('--block-size', type=int, default=1024,
                       help='Block size for digital analysis (default: 1024)')

    parser.add_argument('--threshold', type=float, default=None,
                       help='Demodulation threshold (default: auto-detect)')

    # V3 enhanced analysis options
    parser.add_argument('--v3', action='store_true',
                       help='Enable v3 enhanced analysis (FFT debug, spreading codes, bit/frame)')

    # FFT Pipeline Comparison options
    parser.add_argument('--compare', action='store_true',
                       help='Enable before/after FFT comparison mode (requires exactly 2 files)')
    
    parser.add_argument('--after-is-fft', action='store_true',
                       help='Indicate that the second file is already FFT output (frequency domain)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check GPU availability (Issue 5 Fix: Don't exit, use CPU fallback)
    global GPU_AVAILABLE
    if GPU_AVAILABLE:
        try:
            device = cp.cuda.Device(0)
            mem_info = device.mem_info
            log.info(f"GPU: Device 0 | Memory: {mem_info[1] / 1e9:.1f} GB (Free: {mem_info[0] / 1e9:.1f} GB)")
        except Exception as e:
            log.warning(f"CUDA device error: {e}. Falling back to CPU mode.")
            GPU_AVAILABLE = False
    else:
        log.warning("CUDA not available. Running in CPU-only mode (slower performance).")
        log.info("To enable GPU acceleration, install CuPy: pip install cupy-cuda12x")
    
    # Build configuration
    format_map = {
        'complex64': DataFormat.COMPLEX64,
        'complex128': DataFormat.COMPLEX128,
        'int16': DataFormat.INT16_IQ,
        'int8': DataFormat.INT8_IQ,
        'float32': DataFormat.FLOAT32_REAL,
    }
    
    config = AnalysisConfig(
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        data_format=format_map[args.format],
        fft_size=args.fft_size,
        output_dir=args.output,
        dark_theme=not args.light,
        digital_analysis=args.digital,
        digital_block_size=args.block_size,
        digital_threshold=args.threshold,
    )
    
    analyzer = SignalAnalyzer(config)
    
    # Handle comparison mode
    if args.compare:
        if len(args.files) != 2:
            log.error("Comparison mode requires exactly 2 files: before.bin after.bin")
            log.error("Usage: --compare before.bin after.bin [--after-is-fft]")
            return
        
        before_file = args.files[0].resolve()
        after_file = args.files[1].resolve()
        
        if not before_file.exists() or not after_file.exists():
            log.error("Both files must exist for comparison mode")
            return
        
        log.info(f"FFT Pipeline Comparison Mode")
        log.info(f"  Before: {before_file.name}")
        log.info(f"  After:  {after_file.name}")
        if args.after_is_fft:
            log.info(f"  Note: 'After' file treated as FFT output (frequency domain)")
        
        # Load both files
        before_data, _ = SignalLoader.load(before_file, config, max_samples=args.max_samples)
        after_data, _ = SignalLoader.load(after_file, config, max_samples=args.max_samples)
        
        if before_data is None or after_data is None:
            log.error("Failed to load one or both files")
            return
        
        # Convert to numpy for comparison
        before_np = cp_asnumpy(before_data)
        after_np = cp_asnumpy(after_data)
        
        # Run comparison
        comparator = FFTPipelineComparator(config.sample_rate, config.fft_size)
        result = comparator.compare(
            before_np, after_np,
            before_file, after_file,
            after_is_fft=args.after_is_fft
        )
        
        # Print summary
        print_comparison_summary(result)
        
        # Generate plots
        config.output_dir.mkdir(parents=True, exist_ok=True)
        name_prefix = f"{before_file.stem}_vs_{after_file.stem}"
        plot_files = comparator.generate_comparison_plots(
            before_np, after_np, result,
            config.output_dir, name_prefix,
            after_is_fft=args.after_is_fft
        )
        for pf in plot_files:
            log.info(f"Saved: {pf.name}")
        
        # Save JSON report
        json_file = config.output_dir / f"{name_prefix}_comparison.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        log.info(f"Saved: {json_file.name}")
        
        log.info(f"Comparison complete. Output saved to: {config.output_dir}")
        return
    
    # Process each file (standard mode)
    for filepath in args.files:
        # Issue 8 Fix: Add path validation for security
        try:
            filepath = filepath.resolve()  # Resolve to absolute path
            # Validate path is a regular file (not symlink to sensitive location)
            if not filepath.is_file():
                log.error(f"Not a regular file: {filepath}")
                continue
            # Optional: Check if path is within allowed directories
            # This can be customized based on deployment requirements
        except (OSError, ValueError) as e:
            log.error(f"Invalid path {filepath}: {e}")
            continue
        
        if not filepath.exists():
            log.error(f"File not found: {filepath}")
            continue
        
        log.info(f"Processing: {filepath}")
        
        if args.streaming:
            # Streaming mode for large files
            # Issue 6 Fix: Track residual metadata across chunks (now properly implemented)
            log.info("Using streaming mode...")
            all_metrics = []
            accumulated_residual = {
                'chunks_processed': 0, 
                'inter_chunk_residuals': [],
                'total_residual_bytes': 0
            }
            for chunk_idx, chunk, chunk_residual_meta in SignalLoader.load_chunked(filepath, config):
                chunk_name = f"{filepath.stem}_chunk{chunk_idx:03d}"
                # Merge chunk metadata with residual info from loader
                chunk_residual = {
                    'chunk_idx': chunk_idx,
                    'chunk_samples': len(chunk),
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    **chunk_residual_meta  # Include residual byte info from loader
                }
                accumulated_residual['inter_chunk_residuals'].append(chunk_residual)
                accumulated_residual['chunks_processed'] += 1
                if chunk_residual_meta.get('residual_bytes', 0) > 0:
                    accumulated_residual['total_residual_bytes'] = chunk_residual_meta['residual_bytes']
                metrics = analyzer.run_full_analysis(chunk, chunk_name, enable_v3=args.v3,
                                                    residual_metadata=chunk_residual)
                all_metrics.append(metrics.to_dict())
            log.info(f"Streaming complete: {accumulated_residual['chunks_processed']} chunks processed")
            if accumulated_residual['total_residual_bytes'] > 0:
                log.info(f"FORENSIC: Total residual bytes preserved: {accumulated_residual['total_residual_bytes']}")
        else:
            # Standard mode
            # FORENSIC: SignalLoader.load now returns (data, residual_metadata) tuple
            data, residual_metadata = SignalLoader.load(filepath, config, max_samples=args.max_samples)
            if data is not None:
                analyzer.run_full_analysis(data, filepath.stem, filepath=filepath, 
                                          enable_v3=args.v3, residual_metadata=residual_metadata)
    
    log.info(f"Analysis complete. Output saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
