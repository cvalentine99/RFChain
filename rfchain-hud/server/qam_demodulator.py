#!/usr/bin/env python3
"""
128-QAM Demodulator
Recovers bit stream from QAM modulated signal using symbol timing recovery
"""

import numpy as np
import json
import sys
from scipy import signal
from scipy.ndimage import gaussian_filter

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

def generate_128qam_constellation():
    """Generate standard 128-QAM constellation (cross pattern)"""
    # 128-QAM uses a cross constellation to reduce peak power
    # It's typically 12x12 with corners removed, or similar arrangement
    
    # For simplicity, use a rectangular approximation: 11x12 = 132 points, remove 4
    # Or use 8x16 = 128 points exactly
    
    # Standard approach: use levels that give 128 points
    # 128 = 2^7, so we need 7 bits per symbol
    
    # Create 16x8 rectangular constellation
    i_levels = np.arange(-15, 16, 2)  # -15, -13, ..., 13, 15 (16 levels)
    q_levels = np.arange(-7, 8, 2)    # -7, -5, ..., 5, 7 (8 levels)
    
    # Alternative: 12x12 cross (more common for 128-QAM)
    # Use all points from 12x12 grid except corners
    i_levels = np.arange(-11, 12, 2)  # 12 levels
    q_levels = np.arange(-11, 12, 2)  # 12 levels
    
    constellation = []
    for i in i_levels:
        for q in q_levels:
            # Remove corner points to get 128 from 144
            if abs(i) == 11 and abs(q) == 11:
                continue
            if abs(i) == 11 and abs(q) == 9:
                continue
            if abs(i) == 9 and abs(q) == 11:
                continue
            constellation.append(complex(i, q))
    
    # If we have more than 128, trim
    constellation = constellation[:128]
    
    # Normalize
    constellation = np.array(constellation)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    
    return constellation

def generate_gray_mapping(num_bits):
    """Generate Gray code mapping for symbols"""
    n = 2 ** num_bits
    gray_codes = []
    for i in range(n):
        gray = i ^ (i >> 1)
        gray_codes.append(format(gray, f'0{num_bits}b'))
    return gray_codes

def symbol_timing_recovery(signal_data, samples_per_symbol, num_symbols=1000):
    """
    Gardner timing error detector for symbol timing recovery
    Returns optimal sampling instants
    """
    sps = int(round(samples_per_symbol))
    
    # Use Mueller-Muller or Gardner algorithm
    # Simplified: find optimal phase by maximizing eye opening
    
    best_phase = 0
    best_metric = 0
    
    for phase in range(sps):
        # Sample at this phase
        samples = signal_data[phase::sps][:num_symbols]
        
        # Metric: variance of samples (higher = better eye opening)
        metric = np.var(np.abs(samples))
        
        if metric > best_metric:
            best_metric = metric
            best_phase = phase
    
    return best_phase

def matched_filter(signal_data, samples_per_symbol):
    """Apply matched filter (root raised cosine)"""
    sps = int(round(samples_per_symbol))
    
    # Simple moving average as approximation
    # For proper implementation, use RRC filter
    num_taps = sps * 4
    alpha = 0.35  # Roll-off factor
    
    # Generate RRC filter
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps
    
    # Avoid division by zero
    t[t == 0] = 1e-10
    t_4alpha = t * 4 * alpha
    t_4alpha[t_4alpha == 1] = 1 + 1e-10
    t_4alpha[t_4alpha == -1] = -1 - 1e-10
    
    # RRC impulse response
    h = (np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))) / \
        (np.pi * t * (1 - (4 * alpha * t)**2))
    
    # Handle special cases
    h[np.isnan(h)] = 0
    h[np.isinf(h)] = 0
    
    # Normalize
    h = h / np.sqrt(np.sum(h**2))
    
    # Apply filter
    filtered = np.convolve(signal_data, h, mode='same')
    
    return filtered

def agc(signal_data, target_power=1.0):
    """Automatic Gain Control"""
    current_power = np.mean(np.abs(signal_data)**2)
    gain = np.sqrt(target_power / current_power)
    return signal_data * gain

def carrier_recovery(signal_data, num_iterations=100):
    """
    Simple carrier frequency/phase recovery using decision-directed PLL
    """
    # For QAM, use 4th power method to estimate frequency offset
    signal_4th = signal_data ** 4
    
    # FFT to find frequency offset
    fft = np.fft.fft(signal_4th[:10000])
    freqs = np.fft.fftfreq(len(fft))
    
    # Find peak (frequency offset * 4)
    peak_idx = np.argmax(np.abs(fft))
    freq_offset = freqs[peak_idx] / 4
    
    # Correct frequency offset
    t = np.arange(len(signal_data))
    corrected = signal_data * np.exp(-1j * 2 * np.pi * freq_offset * t)
    
    # Phase correction using pilot or decision-directed
    # Simple approach: rotate to align with constellation
    phase_est = np.angle(np.mean(corrected[:1000]**4)) / 4
    corrected = corrected * np.exp(-1j * phase_est)
    
    return corrected

def nearest_neighbor_decode(symbols, constellation):
    """Map received symbols to nearest constellation points"""
    decoded_indices = []
    
    for sym in symbols:
        distances = np.abs(sym - constellation)
        nearest_idx = np.argmin(distances)
        decoded_indices.append(nearest_idx)
    
    return np.array(decoded_indices)

def symbols_to_bits(symbol_indices, bits_per_symbol):
    """Convert symbol indices to bit stream"""
    bits = []
    
    for idx in symbol_indices:
        # Convert index to binary (Gray coded ideally)
        gray = idx ^ (idx >> 1)  # Gray encode
        bit_string = format(idx, f'0{bits_per_symbol}b')
        bits.extend([int(b) for b in bit_string])
    
    return np.array(bits)

def demodulate_qam(signal_data, sample_rate, symbol_rate, qam_order=128):
    """
    Main demodulation function
    """
    samples_per_symbol = sample_rate / symbol_rate
    bits_per_symbol = int(np.log2(qam_order))
    
    print(f"Demodulating {qam_order}-QAM signal")
    print(f"Sample rate: {sample_rate/1e6:.3f} MHz")
    print(f"Symbol rate: {symbol_rate/1e3:.2f} ksps")
    print(f"Samples per symbol: {samples_per_symbol:.2f}")
    print(f"Bits per symbol: {bits_per_symbol}")
    
    # Step 1: AGC
    print("\nApplying AGC...")
    signal_agc = agc(signal_data)
    
    # Step 2: Carrier recovery
    print("Performing carrier recovery...")
    signal_cr = carrier_recovery(signal_agc)
    
    # Step 3: Matched filter
    print("Applying matched filter...")
    signal_mf = matched_filter(signal_cr, samples_per_symbol)
    
    # Step 4: Symbol timing recovery
    print("Recovering symbol timing...")
    optimal_phase = symbol_timing_recovery(signal_mf, samples_per_symbol)
    print(f"Optimal sampling phase: {optimal_phase}")
    
    # Step 5: Sample at symbol rate
    sps = int(round(samples_per_symbol))
    symbols = signal_mf[optimal_phase::sps]
    print(f"Extracted {len(symbols)} symbols")
    
    # Step 6: Normalize symbols
    symbols = symbols / np.std(symbols) * 0.5
    
    # Step 7: Generate constellation
    constellation = generate_128qam_constellation()
    
    # Step 8: Decode symbols
    print("Decoding symbols...")
    symbol_indices = nearest_neighbor_decode(symbols, constellation)
    
    # Step 9: Convert to bits
    bits = symbols_to_bits(symbol_indices, bits_per_symbol)
    
    return {
        'symbols': symbols,
        'symbol_indices': symbol_indices,
        'bits': bits,
        'constellation': constellation,
        'samples_per_symbol': samples_per_symbol,
        'optimal_phase': optimal_phase,
    }

def format_bits(bits, group_size=8):
    """Format bits for display"""
    bit_string = ''.join(str(b) for b in bits)
    groups = [bit_string[i:i+group_size] for i in range(0, len(bit_string), group_size)]
    return ' '.join(groups)

def bits_to_hex(bits):
    """Convert bits to hexadecimal"""
    hex_str = ''
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i:i+8]
        byte_val = int(''.join(str(b) for b in byte), 2)
        hex_str += f'{byte_val:02X} '
    return hex_str.strip()

def bits_to_ascii(bits):
    """Try to convert bits to ASCII (if printable)"""
    ascii_str = ''
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i:i+8]
        byte_val = int(''.join(str(b) for b in byte), 2)
        if 32 <= byte_val <= 126:
            ascii_str += chr(byte_val)
        else:
            ascii_str += '.'
    return ascii_str

def main():
    if len(sys.argv) < 2:
        print("Usage: python qam_demodulator.py <signal_file> [sample_rate] [symbol_rate] [qam_order]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
    symbol_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 76760
    qam_order = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    
    print("="*70)
    print("                    128-QAM DEMODULATOR")
    print("="*70)
    
    # Load signal
    print(f"\nLoading signal from {filepath}...")
    signal_data = load_signal(filepath, 'complex64')
    print(f"Loaded {len(signal_data):,} samples")
    
    # Demodulate
    result = demodulate_qam(signal_data, sample_rate, symbol_rate, qam_order)
    
    bits = result['bits']
    
    # Display results
    print("\n" + "="*70)
    print("                    DEMODULATION RESULTS")
    print("="*70)
    
    print(f"\nTotal symbols decoded: {len(result['symbol_indices']):,}")
    print(f"Total bits recovered: {len(bits):,}")
    
    # First 100 bits
    first_100_bits = bits[:100]
    
    print(f"\n{'='*70}")
    print("FIRST 100 BITS OF RECOVERED DATA STREAM:")
    print("="*70)
    print(f"\nBinary (grouped by 8):")
    print(format_bits(first_100_bits, 8))
    
    print(f"\nBinary (grouped by 7 - symbol boundaries):")
    print(format_bits(first_100_bits, 7))
    
    # First 128 bits for hex display (16 bytes)
    first_128_bits = bits[:128]
    print(f"\nHexadecimal (first 16 bytes):")
    print(bits_to_hex(first_128_bits))
    
    print(f"\nASCII interpretation (first 16 bytes):")
    print(bits_to_ascii(first_128_bits))
    
    # Statistics
    print(f"\n{'='*70}")
    print("BIT STATISTICS:")
    print("="*70)
    ones = np.sum(first_100_bits)
    zeros = 100 - ones
    print(f"Zeros: {zeros} ({zeros}%)")
    print(f"Ones:  {ones} ({ones}%)")
    print(f"Balance: {abs(zeros-ones)}% deviation from 50/50")
    
    # Symbol distribution
    print(f"\n{'='*70}")
    print("SYMBOL STATISTICS (first 100 symbols):")
    print("="*70)
    first_100_symbols = result['symbol_indices'][:100]
    unique_symbols = len(np.unique(first_100_symbols))
    print(f"Unique symbols used: {unique_symbols} out of {qam_order}")
    
    # Output JSON summary
    summary = {
        'demodulation_params': {
            'qam_order': qam_order,
            'symbol_rate_hz': symbol_rate,
            'sample_rate_hz': sample_rate,
            'samples_per_symbol': float(result['samples_per_symbol']),
        },
        'results': {
            'total_symbols': len(result['symbol_indices']),
            'total_bits': len(bits),
            'first_100_bits': format_bits(first_100_bits, 8),
            'first_16_bytes_hex': bits_to_hex(first_128_bits),
            'first_16_bytes_ascii': bits_to_ascii(first_128_bits),
        },
        'statistics': {
            'bit_zeros': int(zeros),
            'bit_ones': int(ones),
            'unique_symbols_in_first_100': int(unique_symbols),
        }
    }
    
    print(f"\n{'='*70}")
    print("JSON SUMMARY:")
    print("="*70)
    print(json.dumps(summary, indent=2))
    
    return summary

if __name__ == "__main__":
    main()
