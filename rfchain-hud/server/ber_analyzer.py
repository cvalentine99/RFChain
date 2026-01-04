#!/usr/bin/env python3
"""
Bit Error Rate (BER) Analyzer
Tests recovered data against standard PRBS sequences to calculate BER
"""

import numpy as np
import json
import sys
from collections import defaultdict

class PRBSGenerator:
    """
    Pseudo-Random Binary Sequence Generator
    Implements standard PRBS patterns used in communications testing
    """
    
    # Standard PRBS polynomials (taps are 1-indexed positions from MSB)
    POLYNOMIALS = {
        'PRBS-7':  (7, [7, 6]),           # x^7 + x^6 + 1
        'PRBS-9':  (9, [9, 5]),           # x^9 + x^5 + 1
        'PRBS-11': (11, [11, 9]),         # x^11 + x^9 + 1
        'PRBS-15': (15, [15, 14]),        # x^15 + x^14 + 1
        'PRBS-20': (20, [20, 3]),         # x^20 + x^3 + 1
        'PRBS-23': (23, [23, 18]),        # x^23 + x^18 + 1
        'PRBS-31': (31, [31, 28]),        # x^31 + x^28 + 1
    }
    
    def __init__(self, prbs_type='PRBS-7'):
        if prbs_type not in self.POLYNOMIALS:
            raise ValueError(f"Unknown PRBS type: {prbs_type}")
        
        self.prbs_type = prbs_type
        self.order, self.taps = self.POLYNOMIALS[prbs_type]
        self.sequence_length = (2 ** self.order) - 1
        self.register = None
        self.reset()
    
    def reset(self, seed=None):
        """Reset the shift register"""
        if seed is None:
            # Initialize with all ones (standard)
            self.register = [1] * self.order
        else:
            # Initialize with provided seed
            self.register = [(seed >> i) & 1 for i in range(self.order)]
            if all(b == 0 for b in self.register):
                self.register = [1] * self.order  # Avoid all-zeros state
    
    def next_bit(self):
        """Generate next bit in sequence"""
        # Calculate feedback (XOR of tap positions)
        feedback = 0
        for tap in self.taps:
            feedback ^= self.register[tap - 1]
        
        # Output is the last bit
        output = self.register[-1]
        
        # Shift register
        self.register = [feedback] + self.register[:-1]
        
        return output
    
    def generate(self, length):
        """Generate a sequence of specified length"""
        return np.array([self.next_bit() for _ in range(length)])
    
    def generate_full_sequence(self):
        """Generate one complete PRBS cycle"""
        self.reset()
        return self.generate(self.sequence_length)


def find_sequence_alignment(received_bits, prbs_sequence, search_window=1000):
    """
    Find the best alignment between received bits and PRBS sequence
    Returns offset and correlation score
    """
    received = np.array(received_bits[:search_window])
    prbs = np.array(prbs_sequence)
    
    best_offset = 0
    best_correlation = 0
    best_errors = search_window
    
    # Try different offsets
    for offset in range(min(len(prbs) - search_window, search_window)):
        prbs_segment = prbs[offset:offset + search_window]
        
        if len(prbs_segment) < search_window:
            continue
        
        # Count matching bits
        matches = np.sum(received == prbs_segment)
        correlation = matches / search_window
        errors = search_window - matches
        
        if correlation > best_correlation:
            best_correlation = correlation
            best_offset = offset
            best_errors = errors
        
        # Also try inverted (in case of polarity inversion)
        inv_matches = np.sum(received == (1 - prbs_segment))
        inv_correlation = inv_matches / search_window
        
        if inv_correlation > best_correlation:
            best_correlation = inv_correlation
            best_offset = offset
            best_errors = search_window - inv_matches
    
    return best_offset, best_correlation, best_errors


def calculate_ber(received_bits, reference_bits):
    """Calculate Bit Error Rate between received and reference sequences"""
    min_len = min(len(received_bits), len(reference_bits))
    received = np.array(received_bits[:min_len])
    reference = np.array(reference_bits[:min_len])
    
    errors = np.sum(received != reference)
    ber = errors / min_len
    
    return ber, errors, min_len


def analyze_error_distribution(received_bits, reference_bits, block_size=1000):
    """Analyze how errors are distributed across the data"""
    min_len = min(len(received_bits), len(reference_bits))
    received = np.array(received_bits[:min_len])
    reference = np.array(reference_bits[:min_len])
    
    errors_per_block = []
    block_bers = []
    
    for i in range(0, min_len - block_size, block_size):
        block_received = received[i:i + block_size]
        block_reference = reference[i:i + block_size]
        block_errors = np.sum(block_received != block_reference)
        errors_per_block.append(block_errors)
        block_bers.append(block_errors / block_size)
    
    return {
        'errors_per_block': errors_per_block,
        'block_bers': block_bers,
        'mean_block_ber': np.mean(block_bers) if block_bers else 0,
        'std_block_ber': np.std(block_bers) if block_bers else 0,
        'max_block_ber': np.max(block_bers) if block_bers else 0,
        'min_block_ber': np.min(block_bers) if block_bers else 0,
    }


def find_error_bursts(received_bits, reference_bits, burst_threshold=3):
    """Identify burst errors (consecutive errors)"""
    min_len = min(len(received_bits), len(reference_bits))
    received = np.array(received_bits[:min_len])
    reference = np.array(reference_bits[:min_len])
    
    error_positions = np.where(received != reference)[0]
    
    bursts = []
    if len(error_positions) > 0:
        burst_start = error_positions[0]
        burst_length = 1
        
        for i in range(1, len(error_positions)):
            if error_positions[i] - error_positions[i-1] <= burst_threshold:
                burst_length += 1
            else:
                if burst_length > 1:
                    bursts.append({
                        'start': int(burst_start),
                        'length': int(burst_length),
                        'end': int(error_positions[i-1])
                    })
                burst_start = error_positions[i]
                burst_length = 1
        
        # Don't forget the last burst
        if burst_length > 1:
            bursts.append({
                'start': int(burst_start),
                'length': int(burst_length),
                'end': int(error_positions[-1])
            })
    
    return bursts


def test_all_prbs_sequences(received_bits, max_test_length=100000):
    """Test received bits against all standard PRBS sequences"""
    results = {}
    
    test_bits = received_bits[:max_test_length]
    
    for prbs_type in PRBSGenerator.POLYNOMIALS.keys():
        print(f"  Testing {prbs_type}...")
        
        gen = PRBSGenerator(prbs_type)
        
        # Generate enough PRBS bits
        prbs_bits = gen.generate(max_test_length + 1000)
        
        # Find best alignment
        offset, correlation, initial_errors = find_sequence_alignment(
            test_bits, prbs_bits, search_window=min(1000, len(test_bits))
        )
        
        # Calculate BER with alignment
        aligned_prbs = prbs_bits[offset:offset + len(test_bits)]
        if len(aligned_prbs) < len(test_bits):
            # Wrap around for longer sequences
            cycles_needed = (len(test_bits) // gen.sequence_length) + 2
            extended_prbs = np.tile(gen.generate_full_sequence(), cycles_needed)
            aligned_prbs = extended_prbs[offset:offset + len(test_bits)]
        
        ber, errors, total_bits = calculate_ber(test_bits, aligned_prbs)
        
        # Also test inverted
        inv_ber, inv_errors, _ = calculate_ber(test_bits, 1 - aligned_prbs)
        
        if inv_ber < ber:
            ber = inv_ber
            errors = inv_errors
            inverted = True
        else:
            inverted = False
        
        results[prbs_type] = {
            'ber': float(ber),
            'errors': int(errors),
            'total_bits': int(total_bits),
            'correlation': float(correlation),
            'offset': int(offset),
            'inverted': inverted,
            'sequence_length': gen.sequence_length,
        }
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python ber_analyzer.py <signal_file> [sample_rate] [symbol_rate]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
    symbol_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 76760
    
    print("="*70)
    print("                    BIT ERROR RATE (BER) ANALYZER")
    print("="*70)
    
    # First, demodulate the signal to get bits
    print("\nStep 1: Demodulating signal...")
    
    # Import and run demodulator
    from qam_demodulator import load_signal, demodulate_qam
    
    signal_data = load_signal(filepath, 'complex64')
    result = demodulate_qam(signal_data, sample_rate, symbol_rate, 128)
    received_bits = result['bits']
    
    print(f"Recovered {len(received_bits):,} bits from signal")
    
    # Test against PRBS sequences
    print("\nStep 2: Testing against standard PRBS sequences...")
    prbs_results = test_all_prbs_sequences(received_bits)
    
    # Find best matching PRBS
    best_prbs = min(prbs_results.keys(), key=lambda k: prbs_results[k]['ber'])
    best_result = prbs_results[best_prbs]
    
    print("\n" + "="*70)
    print("                    BER ANALYSIS RESULTS")
    print("="*70)
    
    print("\nPRBS Sequence Comparison:")
    print("-" * 70)
    print(f"{'PRBS Type':<12} {'BER':<15} {'Errors':<12} {'Correlation':<12} {'Match'}")
    print("-" * 70)
    
    for prbs_type, res in sorted(prbs_results.items(), key=lambda x: x[1]['ber']):
        match_indicator = "<<<< BEST" if prbs_type == best_prbs else ""
        ber_str = f"{res['ber']:.6f}" if res['ber'] > 0 else "0.000000"
        print(f"{prbs_type:<12} {ber_str:<15} {res['errors']:<12,} {res['correlation']:.4f}       {match_indicator}")
    
    print("-" * 70)
    
    # Detailed analysis for best match
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS FOR BEST MATCH: {best_prbs}")
    print("="*70)
    
    # Regenerate aligned PRBS for detailed analysis
    gen = PRBSGenerator(best_prbs)
    prbs_bits = gen.generate(len(received_bits) + 1000)
    
    if best_result['inverted']:
        aligned_prbs = 1 - prbs_bits[best_result['offset']:best_result['offset'] + len(received_bits)]
    else:
        aligned_prbs = prbs_bits[best_result['offset']:best_result['offset'] + len(received_bits)]
    
    # Error distribution analysis
    print("\nError Distribution Analysis:")
    dist = analyze_error_distribution(received_bits, aligned_prbs)
    print(f"  Mean Block BER:     {dist['mean_block_ber']:.6f}")
    print(f"  Std Dev Block BER:  {dist['std_block_ber']:.6f}")
    print(f"  Max Block BER:      {dist['max_block_ber']:.6f}")
    print(f"  Min Block BER:      {dist['min_block_ber']:.6f}")
    
    # Burst error analysis
    print("\nBurst Error Analysis:")
    bursts = find_error_bursts(received_bits, aligned_prbs)
    print(f"  Total Burst Events: {len(bursts)}")
    if bursts:
        burst_lengths = [b['length'] for b in bursts]
        print(f"  Average Burst Length: {np.mean(burst_lengths):.1f} bits")
        print(f"  Max Burst Length: {max(burst_lengths)} bits")
        print(f"  First 5 Bursts: {bursts[:5]}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"  Best Matching PRBS:    {best_prbs}")
    print(f"  Sequence Length:       {best_result['sequence_length']:,} bits")
    print(f"  Alignment Offset:      {best_result['offset']} bits")
    print(f"  Polarity Inverted:     {'Yes' if best_result['inverted'] else 'No'}")
    print(f"  ")
    print(f"  Total Bits Analyzed:   {best_result['total_bits']:,}")
    print(f"  Total Bit Errors:      {best_result['errors']:,}")
    print(f"  Bit Error Rate (BER):  {best_result['ber']:.2e}")
    print(f"  BER (percentage):      {best_result['ber']*100:.4f}%")
    print(f"  ")
    
    # Quality assessment
    if best_result['ber'] < 1e-6:
        quality = "EXCELLENT (Error-free or near error-free)"
    elif best_result['ber'] < 1e-4:
        quality = "GOOD (Acceptable for most applications)"
    elif best_result['ber'] < 1e-3:
        quality = "FAIR (May need FEC)"
    elif best_result['ber'] < 1e-2:
        quality = "POOR (Significant errors)"
    else:
        quality = "VERY POOR (High error rate, possible sync issues)"
    
    print(f"  Signal Quality:        {quality}")
    
    # Confidence assessment
    if best_result['correlation'] > 0.9:
        confidence = "HIGH - Strong PRBS pattern match"
    elif best_result['correlation'] > 0.7:
        confidence = "MEDIUM - Moderate PRBS pattern match"
    elif best_result['correlation'] > 0.5:
        confidence = "LOW - Weak PRBS pattern match"
    else:
        confidence = "VERY LOW - May not be PRBS data"
    
    print(f"  PRBS Match Confidence: {confidence}")
    print("="*70)
    
    # JSON output
    summary = {
        'best_match': {
            'prbs_type': best_prbs,
            'ber': float(best_result['ber']),
            'ber_scientific': f"{best_result['ber']:.2e}",
            'ber_percentage': float(best_result['ber'] * 100),
            'total_errors': int(best_result['errors']),
            'total_bits': int(best_result['total_bits']),
            'alignment_offset': int(best_result['offset']),
            'polarity_inverted': best_result['inverted'],
            'correlation': float(best_result['correlation']),
        },
        'error_distribution': {
            'mean_block_ber': float(dist['mean_block_ber']),
            'std_block_ber': float(dist['std_block_ber']),
            'max_block_ber': float(dist['max_block_ber']),
            'min_block_ber': float(dist['min_block_ber']),
        },
        'burst_analysis': {
            'total_bursts': len(bursts),
            'average_burst_length': float(np.mean([b['length'] for b in bursts])) if bursts else 0,
            'max_burst_length': max([b['length'] for b in bursts]) if bursts else 0,
        },
        'quality_assessment': quality,
        'all_prbs_results': {k: {
            'ber': float(v['ber']),
            'errors': int(v['errors']),
            'correlation': float(v['correlation'])
        } for k, v in prbs_results.items()}
    }
    
    print("\nJSON Summary:")
    print(json.dumps(summary, indent=2))
    
    return summary


if __name__ == "__main__":
    main()
