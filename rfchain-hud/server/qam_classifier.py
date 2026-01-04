#!/usr/bin/env python3
"""
Enhanced QAM Modulation Classifier
Uses multiple techniques to accurately identify QAM constellation order
"""

import numpy as np
import json
import sys
from scipy import signal
from scipy.ndimage import gaussian_filter
from collections import Counter
import matplotlib.pyplot as plt

def load_signal(filepath, data_format='complex64'):
    """Load IQ signal from binary file"""
    dtype_map = {
        'complex64': np.complex64,
        'complex128': np.complex128,
        'int16': np.int16,
        'int8': np.int8,
        'float32': np.float32,
    }
    dtype = dtype_map.get(data_format, np.complex64)
    data = np.fromfile(filepath, dtype=dtype)
    
    if dtype in [np.int16, np.int8]:
        data = data[::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)
        data = data / np.max(np.abs(data))
    
    return data

def count_constellation_points(signal_data, num_samples=100000, resolution=256):
    """Count unique constellation points using 2D histogram"""
    data = signal_data[:num_samples]
    
    # Normalize to unit power
    data = data / np.std(data) * 0.5
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(
        data.real, data.imag, 
        bins=resolution, 
        range=[[-2, 2], [-2, 2]]
    )
    
    # Smooth histogram to reduce noise
    hist_smooth = gaussian_filter(hist, sigma=1)
    
    # Find peaks (constellation points)
    threshold = np.max(hist_smooth) * 0.02
    peaks = hist_smooth > threshold
    
    # Count connected regions
    from scipy.ndimage import label
    labeled, num_features = label(peaks)
    
    return num_features, hist_smooth

def analyze_grid_pattern(signal_data, num_samples=100000):
    """Analyze if constellation forms a regular grid (QAM characteristic)"""
    data = signal_data[:num_samples]
    data = data / np.std(data)
    
    # Quantize I and Q separately
    i_vals = data.real
    q_vals = data.imag
    
    # Find clusters in I dimension
    i_hist, i_bins = np.histogram(i_vals, bins=200)
    i_smooth = gaussian_filter(i_hist.astype(float), sigma=2)
    i_peaks, _ = signal.find_peaks(i_smooth, height=np.max(i_smooth)*0.05, distance=5)
    
    # Find clusters in Q dimension
    q_hist, q_bins = np.histogram(q_vals, bins=200)
    q_smooth = gaussian_filter(q_hist.astype(float), sigma=2)
    q_peaks, _ = signal.find_peaks(q_smooth, height=np.max(q_smooth)*0.05, distance=5)
    
    # Number of levels in each dimension
    i_levels = len(i_peaks)
    q_levels = len(q_peaks)
    
    # Calculate level spacing regularity
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

def estimate_symbol_rate_spectral(signal_data, sample_rate, num_samples=65536):
    """Estimate symbol rate using spectral analysis of envelope"""
    data = signal_data[:num_samples]
    
    # Compute envelope squared (cyclostationary feature)
    envelope_sq = np.abs(data) ** 2
    envelope_sq = envelope_sq - np.mean(envelope_sq)
    
    # FFT of squared envelope
    fft = np.abs(np.fft.fft(envelope_sq))
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    
    # Only look at positive frequencies, skip DC
    pos_mask = (freqs > 1000) & (freqs < sample_rate/2)
    pos_freqs = freqs[pos_mask]
    pos_fft = fft[pos_mask]
    
    # Find peaks
    peaks, properties = signal.find_peaks(pos_fft, height=np.max(pos_fft)*0.1, distance=10)
    
    if len(peaks) > 0:
        # Highest peak is likely symbol rate or harmonic
        peak_idx = peaks[np.argmax(pos_fft[peaks])]
        symbol_rate = pos_freqs[peak_idx]
        
        # Check if this might be a harmonic
        for divisor in [2, 3, 4]:
            candidate = symbol_rate / divisor
            if candidate > 10000:  # Minimum reasonable symbol rate
                # Check if there's energy at this frequency
                candidate_idx = np.argmin(np.abs(pos_freqs - candidate))
                if pos_fft[candidate_idx] > np.max(pos_fft) * 0.05:
                    symbol_rate = candidate
                    break
    else:
        symbol_rate = 0
    
    samples_per_symbol = sample_rate / symbol_rate if symbol_rate > 0 else 0
    
    return symbol_rate, samples_per_symbol

def classify_qam_order(grid_analysis, constellation_points):
    """Determine QAM order from analysis results"""
    # Standard QAM orders
    qam_orders = {
        4: (2, 2),    # QPSK/4-QAM
        16: (4, 4),   # 16-QAM
        32: (6, 6),   # 32-QAM (cross)
        64: (8, 8),   # 64-QAM
        128: (12, 12), # 128-QAM (cross)
        256: (16, 16), # 256-QAM
    }
    
    i_levels = grid_analysis['i_levels']
    q_levels = grid_analysis['q_levels']
    estimated_points = grid_analysis['estimated_points']
    
    # Find closest standard QAM order
    best_order = 4
    best_diff = float('inf')
    
    for order, (i, q) in qam_orders.items():
        # Check both grid dimensions and total points
        grid_diff = abs(i_levels - i) + abs(q_levels - q)
        point_diff = abs(estimated_points - order)
        total_diff = grid_diff + point_diff * 0.1
        
        if total_diff < best_diff:
            best_diff = total_diff
            best_order = order
    
    # Also consider constellation point count
    point_based_order = min(qam_orders.keys(), key=lambda x: abs(x - constellation_points))
    
    # Confidence based on grid regularity
    confidence = grid_analysis['grid_regularity'] * 100
    
    # If grid analysis and point count agree, high confidence
    if best_order == point_based_order:
        confidence = min(confidence + 20, 100)
    
    return {
        'qam_order': best_order,
        'point_based_estimate': point_based_order,
        'confidence_percent': confidence,
        'bits_per_symbol': int(np.log2(best_order)),
    }

def calculate_data_rate(symbol_rate, bits_per_symbol):
    """Calculate theoretical data rate"""
    return symbol_rate * bits_per_symbol

def main():
    if len(sys.argv) < 2:
        print("Usage: python qam_classifier.py <signal_file> [sample_rate] [data_format]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
    data_format = sys.argv[3] if len(sys.argv) > 3 else 'complex64'
    
    print(f"Loading signal from {filepath}...")
    signal_data = load_signal(filepath, data_format)
    print(f"Loaded {len(signal_data):,} samples")
    print(f"Sample rate: {sample_rate/1e6:.3f} MHz")
    
    # Count constellation points
    print("\nCounting constellation points...")
    num_points, hist = count_constellation_points(signal_data)
    print(f"Detected approximately {num_points} constellation points")
    
    # Analyze grid pattern
    print("Analyzing grid pattern...")
    grid = analyze_grid_pattern(signal_data)
    print(f"I-axis levels: {grid['i_levels']}, Q-axis levels: {grid['q_levels']}")
    print(f"Grid regularity: {grid['grid_regularity']*100:.1f}%")
    
    # Classify QAM order
    print("\nClassifying modulation...")
    classification = classify_qam_order(grid, num_points)
    
    # Estimate symbol rate
    print("Estimating symbol rate...")
    symbol_rate, sps = estimate_symbol_rate_spectral(signal_data, sample_rate)
    
    # Calculate data rate
    data_rate = calculate_data_rate(symbol_rate, classification['bits_per_symbol'])
    
    # Compile results
    results = {
        'modulation': {
            'type': 'QAM',
            'order': classification['qam_order'],
            'name': f"{classification['qam_order']}-QAM",
            'bits_per_symbol': classification['bits_per_symbol'],
            'confidence_percent': round(classification['confidence_percent'], 1),
        },
        'constellation': {
            'detected_points': num_points,
            'i_levels': grid['i_levels'],
            'q_levels': grid['q_levels'],
            'grid_regularity_percent': round(grid['grid_regularity'] * 100, 1),
            'is_square': grid['is_square_qam'],
        },
        'timing': {
            'symbol_rate_hz': round(symbol_rate, 2),
            'symbol_rate_ksps': round(symbol_rate / 1000, 2),
            'samples_per_symbol': round(sps, 2),
            'symbol_period_us': round(1e6 / symbol_rate, 3) if symbol_rate > 0 else 0,
        },
        'data_rate': {
            'raw_rate_bps': round(data_rate, 2),
            'raw_rate_kbps': round(data_rate / 1000, 2),
            'raw_rate_mbps': round(data_rate / 1e6, 4),
        },
        'signal_info': {
            'sample_rate_hz': sample_rate,
            'total_samples': len(signal_data),
            'duration_ms': round(len(signal_data) / sample_rate * 1000, 2),
        }
    }
    
    # Output results
    print("\n" + "="*70)
    print("                    MODULATION CLASSIFICATION RESULTS")
    print("="*70)
    print(f"\n  Modulation Type:     {results['modulation']['name']}")
    print(f"  Bits per Symbol:     {results['modulation']['bits_per_symbol']}")
    print(f"  Confidence:          {results['modulation']['confidence_percent']}%")
    print(f"\n  Symbol Rate:         {results['timing']['symbol_rate_ksps']:.2f} ksps")
    print(f"  Samples per Symbol:  {results['timing']['samples_per_symbol']:.2f}")
    print(f"  Symbol Period:       {results['timing']['symbol_period_us']:.3f} µs")
    print(f"\n  Raw Data Rate:       {results['data_rate']['raw_rate_kbps']:.2f} kbps")
    print(f"                       ({results['data_rate']['raw_rate_mbps']:.4f} Mbps)")
    print(f"\n  Constellation:")
    print(f"    - Detected Points: {results['constellation']['detected_points']}")
    print(f"    - I Levels:        {results['constellation']['i_levels']}")
    print(f"    - Q Levels:        {results['constellation']['q_levels']}")
    print(f"    - Grid Regularity: {results['constellation']['grid_regularity_percent']}%")
    print("="*70)
    
    # Also output JSON for programmatic use
    print("\nJSON Output:")
    print(json.dumps(results, indent=2))
    
    return results

if __name__ == "__main__":
    main()
