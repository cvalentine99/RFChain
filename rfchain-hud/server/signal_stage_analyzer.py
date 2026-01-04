#!/usr/bin/env python3
"""
Signal Stage Analyzer
Determines the processing stage of a captured signal and recommends DSP steps
"""

import numpy as np
import json
import sys
from scipy import signal as sig
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
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
    
    return data

def analyze_time_domain(signal_data):
    """Analyze time-domain characteristics"""
    i_data = signal_data.real
    q_data = signal_data.imag
    magnitude = np.abs(signal_data)
    phase = np.angle(signal_data)
    
    return {
        'sample_count': len(signal_data),
        'i_channel': {
            'mean': float(np.mean(i_data)),
            'std': float(np.std(i_data)),
            'min': float(np.min(i_data)),
            'max': float(np.max(i_data)),
            'peak_to_peak': float(np.max(i_data) - np.min(i_data)),
            'rms': float(np.sqrt(np.mean(i_data**2))),
            'kurtosis': float(kurtosis(i_data)),
            'skewness': float(skew(i_data)),
        },
        'q_channel': {
            'mean': float(np.mean(q_data)),
            'std': float(np.std(q_data)),
            'min': float(np.min(q_data)),
            'max': float(np.max(q_data)),
            'peak_to_peak': float(np.max(q_data) - np.min(q_data)),
            'rms': float(np.sqrt(np.mean(q_data**2))),
            'kurtosis': float(kurtosis(q_data)),
            'skewness': float(skew(q_data)),
        },
        'magnitude': {
            'mean': float(np.mean(magnitude)),
            'std': float(np.std(magnitude)),
            'min': float(np.min(magnitude)),
            'max': float(np.max(magnitude)),
            'dynamic_range_db': float(20 * np.log10(np.max(magnitude) / (np.min(magnitude) + 1e-10))),
        },
        'phase': {
            'mean': float(np.mean(phase)),
            'std': float(np.std(phase)),
            'unwrapped_range': float(np.max(np.unwrap(phase)) - np.min(np.unwrap(phase))),
        },
        'dc_offset': {
            'i': float(np.mean(i_data)),
            'q': float(np.mean(q_data)),
            'magnitude': float(np.abs(np.mean(signal_data))),
            'relative_to_signal': float(np.abs(np.mean(signal_data)) / np.std(signal_data)),
        },
        'iq_balance': {
            'amplitude_ratio': float(np.std(i_data) / np.std(q_data)),
            'amplitude_imbalance_db': float(20 * np.log10(np.std(i_data) / np.std(q_data))),
            'correlation': float(np.corrcoef(i_data[:10000], q_data[:10000])[0, 1]),
        }
    }

def analyze_frequency_domain(signal_data, sample_rate):
    """Analyze frequency-domain characteristics"""
    n_fft = min(65536, len(signal_data))
    
    # Compute PSD using Welch's method
    freqs, psd = sig.welch(signal_data, fs=sample_rate, nperseg=4096, 
                           noverlap=2048, return_onesided=False)
    
    # Sort by frequency
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    psd = psd[sort_idx]
    
    # Find spectral peaks
    psd_db = 10 * np.log10(psd + 1e-20)
    peaks, properties = sig.find_peaks(psd_db, height=-50, distance=10, prominence=5)
    
    # Analyze DC component
    dc_idx = np.argmin(np.abs(freqs))
    dc_power = psd[dc_idx]
    total_power = np.sum(psd)
    dc_ratio = dc_power / total_power
    
    # Find occupied bandwidth (3dB bandwidth)
    max_psd = np.max(psd)
    threshold = max_psd / 2  # -3dB
    occupied_mask = psd > threshold
    if np.any(occupied_mask):
        occupied_freqs = freqs[occupied_mask]
        bandwidth_3db = np.max(occupied_freqs) - np.min(occupied_freqs)
    else:
        bandwidth_3db = 0
    
    # Find 99% power bandwidth
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    low_idx = np.searchsorted(cumsum, 0.005 * total)
    high_idx = np.searchsorted(cumsum, 0.995 * total)
    bandwidth_99 = freqs[high_idx] - freqs[low_idx]
    
    # Spectral flatness (measure of noise-like vs tonal)
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-20)))
    arithmetic_mean = np.mean(psd)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    # Find dominant frequencies
    peak_freqs = freqs[peaks]
    peak_powers = psd_db[peaks]
    sorted_peaks = np.argsort(peak_powers)[::-1]
    
    dominant_freqs = []
    for i in sorted_peaks[:10]:
        dominant_freqs.append({
            'frequency_hz': float(peak_freqs[i]),
            'power_db': float(peak_powers[i]),
        })
    
    return {
        'dc_component': {
            'power': float(dc_power),
            'power_db': float(10 * np.log10(dc_power + 1e-20)),
            'ratio_to_total': float(dc_ratio),
            'ratio_percent': float(dc_ratio * 100),
        },
        'bandwidth': {
            '3db_hz': float(bandwidth_3db),
            '99_percent_hz': float(bandwidth_99),
            'occupied_ratio': float(bandwidth_3db / sample_rate),
        },
        'spectral_characteristics': {
            'flatness': float(spectral_flatness),
            'flatness_db': float(10 * np.log10(spectral_flatness + 1e-20)),
            'peak_to_average_db': float(10 * np.log10(np.max(psd) / np.mean(psd))),
            'num_significant_peaks': len(peaks),
        },
        'dominant_frequencies': dominant_freqs,
        'noise_floor_estimate_db': float(np.percentile(psd_db, 10)),
        'peak_power_db': float(np.max(psd_db)),
    }

def analyze_signal_structure(signal_data, sample_rate):
    """Analyze signal structure for periodicity and patterns"""
    # Autocorrelation analysis
    n_samples = min(50000, len(signal_data))
    data = signal_data[:n_samples]
    
    # Compute autocorrelation
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = np.abs(autocorr) / autocorr[0]
    
    # Find periodicity
    peaks, _ = sig.find_peaks(autocorr[100:], height=0.3, distance=50)
    peaks = peaks + 100  # Adjust for offset
    
    if len(peaks) > 0:
        primary_period = peaks[0]
        period_frequency = sample_rate / primary_period
    else:
        primary_period = 0
        period_frequency = 0
    
    # Cyclostationary analysis (squared envelope)
    envelope_sq = np.abs(data) ** 2
    env_autocorr = np.correlate(envelope_sq - np.mean(envelope_sq), 
                                 envelope_sq - np.mean(envelope_sq), mode='full')
    env_autocorr = env_autocorr[len(env_autocorr)//2:]
    env_autocorr = env_autocorr / env_autocorr[0]
    
    env_peaks, _ = sig.find_peaks(env_autocorr[10:], height=0.1, distance=5)
    
    if len(env_peaks) > 0:
        symbol_period_samples = env_peaks[0] + 10
        estimated_symbol_rate = sample_rate / symbol_period_samples
    else:
        symbol_period_samples = 0
        estimated_symbol_rate = 0
    
    # Check for OFDM-like structure (cyclic prefix)
    # Look for correlation at typical CP lengths
    cp_candidates = []
    for cp_ratio in [1/4, 1/8, 1/16, 1/32]:
        for fft_size in [64, 128, 256, 512, 1024, 2048, 4096]:
            cp_length = int(fft_size * cp_ratio)
            symbol_length = fft_size + cp_length
            
            if symbol_length < len(autocorr):
                corr_at_symbol = autocorr[symbol_length]
                if corr_at_symbol > 0.5:
                    cp_candidates.append({
                        'fft_size': fft_size,
                        'cp_length': cp_length,
                        'symbol_length': symbol_length,
                        'correlation': float(corr_at_symbol),
                    })
    
    return {
        'periodicity': {
            'primary_period_samples': int(primary_period),
            'period_frequency_hz': float(period_frequency),
            'num_periodic_peaks': len(peaks),
        },
        'symbol_timing': {
            'estimated_period_samples': int(symbol_period_samples),
            'estimated_symbol_rate_hz': float(estimated_symbol_rate),
        },
        'ofdm_candidates': sorted(cp_candidates, key=lambda x: -x['correlation'])[:5],
        'has_cyclic_structure': len(cp_candidates) > 0,
    }

def determine_processing_stage(filename, time_analysis, freq_analysis, structure_analysis):
    """Determine the signal processing stage based on analysis"""
    
    indicators = {
        'raw_adc': 0,
        'after_ddc': 0,
        'after_filtering': 0,
        'after_agc': 0,
        'baseband': 0,
        'pre_fft': 0,
        'frequency_domain': 0,
    }
    
    evidence = []
    
    # Filename analysis
    filename_lower = filename.lower()
    if 'beforefft' in filename_lower or 'pre_fft' in filename_lower or 'prefft' in filename_lower:
        indicators['pre_fft'] += 3
        evidence.append("Filename contains 'beforeFFT' - indicates pre-FFT time-domain samples")
    
    if 'dump' in filename_lower:
        indicators['raw_adc'] += 1
        evidence.append("Filename contains 'dump' - suggests raw capture point")
    
    # DC offset analysis
    dc_ratio = time_analysis['dc_offset']['relative_to_signal']
    if dc_ratio > 0.1:
        indicators['raw_adc'] += 2
        evidence.append(f"Significant DC offset ({dc_ratio:.3f}) - typical of raw ADC output")
    elif dc_ratio < 0.01:
        indicators['after_filtering'] += 1
        evidence.append("Minimal DC offset - suggests DC removal filtering applied")
    
    # I/Q balance analysis
    iq_imbalance = abs(time_analysis['iq_balance']['amplitude_imbalance_db'])
    if iq_imbalance > 1.0:
        indicators['raw_adc'] += 1
        evidence.append(f"I/Q imbalance ({iq_imbalance:.2f} dB) - uncorrected receiver")
    elif iq_imbalance < 0.1:
        indicators['after_agc'] += 1
        evidence.append("Well-balanced I/Q - suggests calibration applied")
    
    # Spectral analysis
    spectral_flatness = freq_analysis['spectral_characteristics']['flatness']
    if spectral_flatness > 0.5:
        indicators['after_filtering'] += 1
        evidence.append("High spectral flatness - noise-like or spread spectrum")
    
    num_peaks = freq_analysis['spectral_characteristics']['num_significant_peaks']
    if num_peaks > 5:
        indicators['pre_fft'] += 1
        evidence.append(f"Multiple spectral peaks ({num_peaks}) - multi-carrier or pre-FFT OFDM")
    
    # OFDM structure check
    if structure_analysis['has_cyclic_structure']:
        indicators['pre_fft'] += 2
        evidence.append("Cyclic prefix structure detected - OFDM time-domain samples")
    
    # Bandwidth analysis
    bw_ratio = freq_analysis['bandwidth']['occupied_ratio']
    if bw_ratio > 0.8:
        indicators['baseband'] += 1
        evidence.append("High bandwidth occupation - baseband signal")
    
    # Determine most likely stage
    most_likely = max(indicators, key=indicators.get)
    confidence = indicators[most_likely] / max(sum(indicators.values()), 1) * 100
    
    return {
        'most_likely_stage': most_likely,
        'confidence_percent': float(confidence),
        'indicator_scores': indicators,
        'evidence': evidence,
    }

def recommend_dsp_pipeline(stage_analysis, freq_analysis, structure_analysis):
    """Recommend DSP processing steps based on signal stage"""
    
    stage = stage_analysis['most_likely_stage']
    recommendations = []
    
    # Common preprocessing
    recommendations.append({
        'step': 1,
        'name': 'DC Offset Removal',
        'description': 'Remove DC component to center signal at baseband',
        'method': 'Subtract mean or use high-pass filter with very low cutoff',
        'priority': 'HIGH',
    })
    
    if stage in ['raw_adc', 'pre_fft']:
        recommendations.append({
            'step': 2,
            'name': 'I/Q Imbalance Correction',
            'description': 'Correct amplitude and phase imbalance between I and Q channels',
            'method': 'Blind estimation using signal statistics or known training sequence',
            'priority': 'MEDIUM',
        })
    
    # Check for OFDM
    if structure_analysis['has_cyclic_structure'] and structure_analysis['ofdm_candidates']:
        best_ofdm = structure_analysis['ofdm_candidates'][0]
        recommendations.append({
            'step': 3,
            'name': 'OFDM Symbol Synchronization',
            'description': f"Synchronize to OFDM symbols (FFT size: {best_ofdm['fft_size']}, CP: {best_ofdm['cp_length']})",
            'method': 'Use cyclic prefix correlation for timing, then apply FFT',
            'priority': 'HIGH',
        })
        recommendations.append({
            'step': 4,
            'name': 'FFT Processing',
            'description': f"Apply {best_ofdm['fft_size']}-point FFT to extract subcarriers",
            'method': 'Remove CP, apply FFT, extract data subcarriers',
            'priority': 'HIGH',
        })
        recommendations.append({
            'step': 5,
            'name': 'Channel Estimation & Equalization',
            'description': 'Estimate and correct channel response per subcarrier',
            'method': 'Use pilot subcarriers or preamble for channel estimation',
            'priority': 'HIGH',
        })
        recommendations.append({
            'step': 6,
            'name': 'Subcarrier Demodulation',
            'description': 'Demodulate each subcarrier (likely QAM)',
            'method': 'Apply per-subcarrier QAM demodulation after equalization',
            'priority': 'HIGH',
        })
    else:
        # Single-carrier processing
        recommendations.append({
            'step': 3,
            'name': 'Carrier Frequency Offset Estimation',
            'description': 'Estimate and correct any residual carrier frequency offset',
            'method': 'Use 4th power method for QAM or decision-directed PLL',
            'priority': 'HIGH',
        })
        recommendations.append({
            'step': 4,
            'name': 'Matched Filtering',
            'description': 'Apply pulse-shaping matched filter (root raised cosine)',
            'method': 'Convolve with RRC filter matched to transmitter',
            'priority': 'MEDIUM',
        })
        recommendations.append({
            'step': 5,
            'name': 'Symbol Timing Recovery',
            'description': 'Recover optimal sampling instants',
            'method': 'Gardner or Mueller-Muller timing error detector',
            'priority': 'HIGH',
        })
        recommendations.append({
            'step': 6,
            'name': 'Adaptive Equalization',
            'description': 'Compensate for channel distortion',
            'method': 'CMA (blind) or LMS (decision-directed) equalizer',
            'priority': 'MEDIUM',
        })
    
    return recommendations

def main():
    if len(sys.argv) < 2:
        print("Usage: python signal_stage_analyzer.py <signal_file> [sample_rate]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
    
    # Extract filename
    import os
    filename = os.path.basename(filepath)
    
    print("="*70)
    print("                    SIGNAL STAGE ANALYZER")
    print("="*70)
    print(f"\nFile: {filename}")
    print(f"Sample Rate: {sample_rate/1e6:.3f} MHz")
    
    # Load signal
    print("\nLoading signal...")
    signal_data = load_signal(filepath, 'complex64')
    print(f"Loaded {len(signal_data):,} samples ({len(signal_data)/sample_rate*1000:.2f} ms)")
    
    # Time domain analysis
    print("\nAnalyzing time domain characteristics...")
    time_analysis = analyze_time_domain(signal_data)
    
    # Frequency domain analysis
    print("Analyzing frequency domain characteristics...")
    freq_analysis = analyze_frequency_domain(signal_data, sample_rate)
    
    # Structure analysis
    print("Analyzing signal structure...")
    structure_analysis = analyze_signal_structure(signal_data, sample_rate)
    
    # Determine processing stage
    print("Determining processing stage...")
    stage_analysis = determine_processing_stage(filename, time_analysis, freq_analysis, structure_analysis)
    
    # Generate recommendations
    print("Generating DSP recommendations...")
    recommendations = recommend_dsp_pipeline(stage_analysis, freq_analysis, structure_analysis)
    
    # Output results
    print("\n" + "="*70)
    print("                    ANALYSIS RESULTS")
    print("="*70)
    
    print("\n[TIME DOMAIN CHARACTERISTICS]")
    print(f"  I Channel:  mean={time_analysis['i_channel']['mean']:.6f}, std={time_analysis['i_channel']['std']:.6f}")
    print(f"  Q Channel:  mean={time_analysis['q_channel']['mean']:.6f}, std={time_analysis['q_channel']['std']:.6f}")
    print(f"  DC Offset:  {time_analysis['dc_offset']['magnitude']:.6f} ({time_analysis['dc_offset']['relative_to_signal']*100:.2f}% of signal)")
    print(f"  I/Q Imbalance: {time_analysis['iq_balance']['amplitude_imbalance_db']:.2f} dB")
    print(f"  I/Q Correlation: {time_analysis['iq_balance']['correlation']:.4f}")
    print(f"  I Kurtosis: {time_analysis['i_channel']['kurtosis']:.2f} (Gaussian=0)")
    print(f"  Q Kurtosis: {time_analysis['q_channel']['kurtosis']:.2f} (Gaussian=0)")
    
    print("\n[FREQUENCY DOMAIN CHARACTERISTICS]")
    print(f"  DC Power: {freq_analysis['dc_component']['power_db']:.1f} dB ({freq_analysis['dc_component']['ratio_percent']:.2f}% of total)")
    print(f"  3dB Bandwidth: {freq_analysis['bandwidth']['3db_hz']/1e3:.2f} kHz")
    print(f"  99% Bandwidth: {freq_analysis['bandwidth']['99_percent_hz']/1e3:.2f} kHz")
    print(f"  Spectral Flatness: {freq_analysis['spectral_characteristics']['flatness']:.4f} ({freq_analysis['spectral_characteristics']['flatness_db']:.1f} dB)")
    print(f"  Significant Peaks: {freq_analysis['spectral_characteristics']['num_significant_peaks']}")
    print(f"  Peak Power: {freq_analysis['peak_power_db']:.1f} dB")
    print(f"  Noise Floor: {freq_analysis['noise_floor_estimate_db']:.1f} dB")
    
    print("\n  Dominant Frequencies:")
    for i, peak in enumerate(freq_analysis['dominant_frequencies'][:5]):
        print(f"    {i+1}. {peak['frequency_hz']/1e3:+.2f} kHz @ {peak['power_db']:.1f} dB")
    
    print("\n[SIGNAL STRUCTURE]")
    print(f"  Primary Period: {structure_analysis['periodicity']['primary_period_samples']} samples")
    print(f"  Period Frequency: {structure_analysis['periodicity']['period_frequency_hz']:.2f} Hz")
    print(f"  Est. Symbol Rate: {structure_analysis['symbol_timing']['estimated_symbol_rate_hz']/1e3:.2f} kHz")
    print(f"  OFDM Structure Detected: {'Yes' if structure_analysis['has_cyclic_structure'] else 'No'}")
    
    if structure_analysis['ofdm_candidates']:
        print("\n  OFDM Candidates:")
        for i, cand in enumerate(structure_analysis['ofdm_candidates'][:3]):
            print(f"    {i+1}. FFT={cand['fft_size']}, CP={cand['cp_length']}, Corr={cand['correlation']:.3f}")
    
    print("\n" + "="*70)
    print("                    PROCESSING STAGE DETERMINATION")
    print("="*70)
    
    stage_names = {
        'raw_adc': 'Raw ADC Output',
        'after_ddc': 'After Digital Down-Conversion',
        'after_filtering': 'After Channel Filtering',
        'after_agc': 'After AGC',
        'baseband': 'Baseband Signal',
        'pre_fft': 'Pre-FFT (OFDM Time Domain)',
        'frequency_domain': 'Frequency Domain',
    }
    
    print(f"\n  Most Likely Stage: {stage_names.get(stage_analysis['most_likely_stage'], stage_analysis['most_likely_stage'])}")
    print(f"  Confidence: {stage_analysis['confidence_percent']:.1f}%")
    
    print("\n  Evidence:")
    for ev in stage_analysis['evidence']:
        print(f"    • {ev}")
    
    print("\n" + "="*70)
    print("                    RECOMMENDED DSP PIPELINE")
    print("="*70)
    
    for rec in recommendations:
        print(f"\n  Step {rec['step']}: {rec['name']} [{rec['priority']}]")
        print(f"    Description: {rec['description']}")
        print(f"    Method: {rec['method']}")
    
    print("\n" + "="*70)
    
    # JSON summary
    summary = {
        'filename': filename,
        'sample_rate_hz': sample_rate,
        'duration_ms': len(signal_data) / sample_rate * 1000,
        'processing_stage': {
            'determined_stage': stage_analysis['most_likely_stage'],
            'stage_name': stage_names.get(stage_analysis['most_likely_stage'], stage_analysis['most_likely_stage']),
            'confidence_percent': stage_analysis['confidence_percent'],
            'evidence': stage_analysis['evidence'],
        },
        'signal_characteristics': {
            'dc_offset_percent': time_analysis['dc_offset']['relative_to_signal'] * 100,
            'iq_imbalance_db': time_analysis['iq_balance']['amplitude_imbalance_db'],
            'bandwidth_3db_khz': freq_analysis['bandwidth']['3db_hz'] / 1e3,
            'spectral_flatness': freq_analysis['spectral_characteristics']['flatness'],
            'is_ofdm': structure_analysis['has_cyclic_structure'],
        },
        'ofdm_parameters': structure_analysis['ofdm_candidates'][0] if structure_analysis['ofdm_candidates'] else None,
        'recommended_pipeline': [
            {'step': r['step'], 'name': r['name'], 'priority': r['priority']} 
            for r in recommendations
        ],
    }
    
    print("\nJSON Summary:")
    print(json.dumps(summary, indent=2))
    
    return summary

if __name__ == "__main__":
    main()
