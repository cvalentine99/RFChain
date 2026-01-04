#!/usr/bin/env python3
"""
Advanced Modulation Classification for QAM signals
Analyzes constellation patterns to identify modulation order and symbol rate
"""

import numpy as np
import json
import sys
from scipy import signal
from scipy.cluster.hierarchy import fclusterdata
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
        # Convert interleaved I/Q to complex
        data = data[::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)
        data = data / np.max(np.abs(data))
    
    return data

def estimate_symbol_rate(signal_data, sample_rate):
    """Estimate symbol rate using cyclostationary analysis"""
    # Compute instantaneous amplitude
    envelope = np.abs(signal_data)
    
    # Compute autocorrelation of squared envelope (cyclostationary feature)
    squared_env = envelope ** 2
    autocorr = np.correlate(squared_env[:10000], squared_env[:10000], mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks in autocorrelation (symbol period)
    peaks, properties = signal.find_peaks(autocorr[1:], height=autocorr[0]*0.1, distance=5)
    
    if len(peaks) > 0:
        # First significant peak indicates symbol period
        symbol_period_samples = peaks[0] + 1
        symbol_rate = sample_rate / symbol_period_samples
        samples_per_symbol = symbol_period_samples
    else:
        # Fallback: use spectral analysis
        fft_env = np.abs(np.fft.fft(squared_env[:65536]))
        freqs = np.fft.fftfreq(len(fft_env), 1/sample_rate)
        
        # Find dominant frequency (symbol rate)
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_env[:len(fft_env)//2]
        
        # Ignore DC and very low frequencies
        start_idx = int(len(pos_freqs) * 0.01)
        peak_idx = start_idx + np.argmax(pos_fft[start_idx:])
        symbol_rate = pos_freqs[peak_idx]
        samples_per_symbol = sample_rate / symbol_rate if symbol_rate > 0 else 0
    
    return symbol_rate, samples_per_symbol

def classify_qam_order(signal_data, num_samples=50000):
    """Classify QAM order based on constellation clustering"""
    # Normalize the signal
    data = signal_data[:num_samples]
    data = data / np.std(data)
    
    # Create 2D array for clustering (I, Q)
    points = np.column_stack([data.real, data.imag])
    
    # Try different cluster counts and find best fit
    cluster_candidates = [4, 8, 16, 32, 64, 128, 256]  # QAM orders
    best_order = 4
    best_score = float('inf')
    
    results = {}
    
    for n_clusters in cluster_candidates:
        if n_clusters > len(points) // 10:
            continue
            
        try:
            # Use hierarchical clustering
            labels = fclusterdata(points, t=n_clusters, criterion='maxclust')
            
            # Calculate cluster quality (within-cluster variance)
            unique_labels = np.unique(labels)
            total_variance = 0
            cluster_centers = []
            
            for label in unique_labels:
                cluster_points = points[labels == label]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
                variance = np.mean(np.sum((cluster_points - center)**2, axis=1))
                total_variance += variance
            
            avg_variance = total_variance / len(unique_labels)
            
            # Check if clusters form a grid pattern (QAM characteristic)
            centers = np.array(cluster_centers)
            
            # Calculate grid regularity score
            if len(centers) >= 4:
                # Check for regular spacing
                i_values = np.sort(np.unique(np.round(centers[:, 0], 2)))
                q_values = np.sort(np.unique(np.round(centers[:, 1], 2)))
                
                grid_score = len(i_values) * len(q_values)
                regularity = abs(grid_score - n_clusters) / n_clusters
            else:
                regularity = 1.0
            
            # Combined score (lower is better)
            score = avg_variance * (1 + regularity)
            
            results[n_clusters] = {
                'variance': avg_variance,
                'regularity': regularity,
                'score': score,
                'actual_clusters': len(unique_labels)
            }
            
            if score < best_score and len(unique_labels) >= n_clusters * 0.8:
                best_score = score
                best_order = n_clusters
                
        except Exception as e:
            continue
    
    return best_order, results

def analyze_constellation_geometry(signal_data, num_samples=50000):
    """Analyze constellation geometry for modulation identification"""
    data = signal_data[:num_samples]
    data = data / np.std(data)
    
    # Calculate amplitude histogram
    amplitudes = np.abs(data)
    amp_hist, amp_bins = np.histogram(amplitudes, bins=100)
    
    # Count amplitude levels (rings in constellation)
    amp_peaks, _ = signal.find_peaks(amp_hist, height=np.max(amp_hist)*0.1, distance=5)
    num_amplitude_levels = len(amp_peaks)
    
    # Calculate phase histogram
    phases = np.angle(data)
    phase_hist, phase_bins = np.histogram(phases, bins=360)
    
    # Count phase levels
    phase_peaks, _ = signal.find_peaks(phase_hist, height=np.max(phase_hist)*0.05, distance=5)
    num_phase_levels = len(phase_peaks)
    
    # Determine modulation type based on geometry
    if num_amplitude_levels <= 2 and num_phase_levels >= 4:
        mod_type = "PSK"
        if num_phase_levels <= 4:
            order = 4  # QPSK
        elif num_phase_levels <= 8:
            order = 8  # 8-PSK
        else:
            order = 16  # Higher order PSK
    elif num_amplitude_levels >= 3:
        mod_type = "QAM"
        # Estimate QAM order from amplitude and phase levels
        estimated_points = num_amplitude_levels * num_phase_levels
        
        # Map to standard QAM orders
        qam_orders = [4, 16, 64, 256, 1024]
        order = min(qam_orders, key=lambda x: abs(x - estimated_points))
    else:
        mod_type = "Unknown"
        order = 0
    
    return {
        'modulation_type': mod_type,
        'estimated_order': order,
        'amplitude_levels': num_amplitude_levels,
        'phase_levels': num_phase_levels,
        'amplitude_peaks': amp_peaks.tolist(),
        'phase_peaks': phase_peaks.tolist()
    }

def estimate_evm(signal_data, qam_order, num_samples=10000):
    """Estimate Error Vector Magnitude"""
    data = signal_data[:num_samples]
    data = data / np.std(data)
    
    # Generate ideal QAM constellation
    sqrt_order = int(np.sqrt(qam_order))
    if sqrt_order ** 2 != qam_order:
        sqrt_order = int(np.ceil(np.sqrt(qam_order)))
    
    # Create ideal constellation points
    levels = np.arange(-(sqrt_order-1), sqrt_order, 2)
    ideal_points = []
    for i in levels:
        for q in levels:
            ideal_points.append(complex(i, q))
    ideal_points = np.array(ideal_points[:qam_order])
    
    # Normalize ideal constellation
    ideal_points = ideal_points / np.std(ideal_points)
    
    # Find nearest ideal point for each sample
    errors = []
    for sample in data:
        distances = np.abs(sample - ideal_points)
        min_dist = np.min(distances)
        errors.append(min_dist)
    
    # Calculate EVM
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    rms_signal = np.sqrt(np.mean(np.abs(data)**2))
    evm_percent = (rms_error / rms_signal) * 100
    
    return evm_percent

def main():
    if len(sys.argv) < 2:
        print("Usage: python modulation_classifier.py <signal_file> [sample_rate] [data_format]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
    data_format = sys.argv[3] if len(sys.argv) > 3 else 'complex64'
    
    print(f"Loading signal from {filepath}...")
    signal_data = load_signal(filepath, data_format)
    print(f"Loaded {len(signal_data)} samples")
    
    # Estimate symbol rate
    print("\nEstimating symbol rate...")
    symbol_rate, samples_per_symbol = estimate_symbol_rate(signal_data, sample_rate)
    
    # Analyze constellation geometry
    print("Analyzing constellation geometry...")
    geometry = analyze_constellation_geometry(signal_data)
    
    # Classify QAM order
    print("Classifying modulation order...")
    qam_order, cluster_results = classify_qam_order(signal_data)
    
    # Estimate EVM
    print("Estimating EVM...")
    evm = estimate_evm(signal_data, qam_order)
    
    # Compile results
    results = {
        'modulation_classification': {
            'detected_type': geometry['modulation_type'],
            'detected_order': qam_order,
            'geometry_estimate': geometry['estimated_order'],
            'amplitude_levels': geometry['amplitude_levels'],
            'phase_levels': geometry['phase_levels'],
        },
        'symbol_timing': {
            'estimated_symbol_rate_hz': symbol_rate,
            'samples_per_symbol': samples_per_symbol,
            'baud_rate': symbol_rate,
        },
        'signal_quality': {
            'evm_percent': evm,
            'evm_db': 20 * np.log10(evm/100) if evm > 0 else -100,
        },
        'cluster_analysis': cluster_results,
        'sample_rate_hz': sample_rate,
        'total_samples': len(signal_data),
    }
    
    # Output as JSON
    print("\n" + "="*60)
    print("MODULATION CLASSIFICATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))
    
    return results

if __name__ == "__main__":
    main()
