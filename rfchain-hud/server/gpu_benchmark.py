#!/usr/bin/env python3
"""
GPU Benchmark Script for RFChain HUD
Tests FFT, correlation, and PSD performance on CPU vs GPU
"""

import json
import time
import sys
import numpy as np

# Check for CuPy availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def benchmark_fft(size: int, iterations: int = 100):
    """Benchmark FFT performance on CPU and GPU."""
    # Generate test signal
    np.random.seed(42)
    signal_real = np.random.randn(size).astype(np.float32)
    signal_imag = np.random.randn(size).astype(np.float32)
    signal_cpu = signal_real + 1j * signal_imag
    
    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = np.fft.fft(signal_cpu)
        cpu_times.append(time.perf_counter() - start)
    
    cpu_avg = np.mean(cpu_times) * 1000  # Convert to ms
    cpu_std = np.std(cpu_times) * 1000
    
    result = {
        'operation': 'FFT',
        'size': size,
        'iterations': iterations,
        'cpu': {
            'avg_ms': round(cpu_avg, 3),
            'std_ms': round(cpu_std, 3),
        }
    }
    
    # GPU benchmark if available
    if GPU_AVAILABLE:
        signal_gpu = cp.asarray(signal_cpu)
        
        # Warmup
        for _ in range(10):
            _ = cp.fft.fft(signal_gpu)
        cp.cuda.Stream.null.synchronize()
        
        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = cp.fft.fft(signal_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - start)
        
        gpu_avg = np.mean(gpu_times) * 1000
        gpu_std = np.std(gpu_times) * 1000
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        result['gpu'] = {
            'avg_ms': round(gpu_avg, 3),
            'std_ms': round(gpu_std, 3),
        }
        result['speedup'] = round(speedup, 2)
    
    return result

def benchmark_correlation(size: int, iterations: int = 50):
    """Benchmark cross-correlation performance."""
    np.random.seed(42)
    signal1 = np.random.randn(size).astype(np.float32)
    signal2 = np.random.randn(size // 4).astype(np.float32)
    
    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = np.correlate(signal1, signal2, mode='valid')
        cpu_times.append(time.perf_counter() - start)
    
    cpu_avg = np.mean(cpu_times) * 1000
    cpu_std = np.std(cpu_times) * 1000
    
    result = {
        'operation': 'Correlation',
        'size': size,
        'iterations': iterations,
        'cpu': {
            'avg_ms': round(cpu_avg, 3),
            'std_ms': round(cpu_std, 3),
        }
    }
    
    if GPU_AVAILABLE:
        signal1_gpu = cp.asarray(signal1)
        signal2_gpu = cp.asarray(signal2)
        
        # Warmup
        for _ in range(5):
            _ = cp.correlate(signal1_gpu, signal2_gpu, mode='valid')
        cp.cuda.Stream.null.synchronize()
        
        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = cp.correlate(signal1_gpu, signal2_gpu, mode='valid')
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - start)
        
        gpu_avg = np.mean(gpu_times) * 1000
        gpu_std = np.std(gpu_times) * 1000
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        result['gpu'] = {
            'avg_ms': round(gpu_avg, 3),
            'std_ms': round(gpu_std, 3),
        }
        result['speedup'] = round(speedup, 2)
    
    return result

def benchmark_psd(size: int, iterations: int = 50):
    """Benchmark Power Spectral Density calculation."""
    np.random.seed(42)
    signal = np.random.randn(size).astype(np.float32) + 1j * np.random.randn(size).astype(np.float32)
    
    # CPU benchmark (manual PSD calculation similar to the analysis script)
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fft_result = np.fft.fft(signal)
        psd = np.abs(fft_result) ** 2 / size
        cpu_times.append(time.perf_counter() - start)
    
    cpu_avg = np.mean(cpu_times) * 1000
    cpu_std = np.std(cpu_times) * 1000
    
    result = {
        'operation': 'PSD',
        'size': size,
        'iterations': iterations,
        'cpu': {
            'avg_ms': round(cpu_avg, 3),
            'std_ms': round(cpu_std, 3),
        }
    }
    
    if GPU_AVAILABLE:
        signal_gpu = cp.asarray(signal)
        
        # Warmup
        for _ in range(5):
            fft_result = cp.fft.fft(signal_gpu)
            psd = cp.abs(fft_result) ** 2 / size
        cp.cuda.Stream.null.synchronize()
        
        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fft_result = cp.fft.fft(signal_gpu)
            psd = cp.abs(fft_result) ** 2 / size
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - start)
        
        gpu_avg = np.mean(gpu_times) * 1000
        gpu_std = np.std(gpu_times) * 1000
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        result['gpu'] = {
            'avg_ms': round(gpu_avg, 3),
            'std_ms': round(gpu_std, 3),
        }
        result['speedup'] = round(speedup, 2)
    
    return result

def benchmark_array_ops(size: int, iterations: int = 100):
    """Benchmark common array operations."""
    np.random.seed(42)
    arr1 = np.random.randn(size).astype(np.float32)
    arr2 = np.random.randn(size).astype(np.float32)
    
    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = np.abs(arr1)
        _ = np.sum(arr1 ** 2)
        _ = arr1 * arr2
        _ = np.mean(arr1)
        cpu_times.append(time.perf_counter() - start)
    
    cpu_avg = np.mean(cpu_times) * 1000
    cpu_std = np.std(cpu_times) * 1000
    
    result = {
        'operation': 'Array Ops',
        'size': size,
        'iterations': iterations,
        'cpu': {
            'avg_ms': round(cpu_avg, 3),
            'std_ms': round(cpu_std, 3),
        }
    }
    
    if GPU_AVAILABLE:
        arr1_gpu = cp.asarray(arr1)
        arr2_gpu = cp.asarray(arr2)
        
        # Warmup
        for _ in range(10):
            _ = cp.abs(arr1_gpu)
            _ = cp.sum(arr1_gpu ** 2)
            _ = arr1_gpu * arr2_gpu
            _ = cp.mean(arr1_gpu)
        cp.cuda.Stream.null.synchronize()
        
        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = cp.abs(arr1_gpu)
            _ = cp.sum(arr1_gpu ** 2)
            _ = arr1_gpu * arr2_gpu
            _ = cp.mean(arr1_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - start)
        
        gpu_avg = np.mean(gpu_times) * 1000
        gpu_std = np.std(gpu_times) * 1000
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        result['gpu'] = {
            'avg_ms': round(gpu_avg, 3),
            'std_ms': round(gpu_std, 3),
        }
        result['speedup'] = round(speedup, 2)
    
    return result

def run_full_benchmark():
    """Run complete benchmark suite."""
    results = {
        'success': True,
        'gpu_available': GPU_AVAILABLE,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': [],
        'summary': {}
    }
    
    # Get GPU info if available
    if GPU_AVAILABLE:
        try:
            results['gpu_info'] = {
                'name': cp.cuda.Device(0).name,
                'compute_capability': cp.cuda.Device(0).compute_capability,
                'total_memory_gb': round(cp.cuda.Device(0).mem_info[1] / (1024**3), 2),
            }
        except Exception as e:
            results['gpu_info'] = {'error': str(e)}
    
    # Test sizes (typical signal analysis sizes)
    test_sizes = [
        (65536, 'Small (64K samples)'),      # ~0.5 MB
        (262144, 'Medium (256K samples)'),   # ~2 MB
        (1048576, 'Large (1M samples)'),     # ~8 MB
        (4194304, 'XLarge (4M samples)'),    # ~32 MB
    ]
    
    total_speedup = []
    
    for size, size_name in test_sizes:
        try:
            # FFT benchmark
            fft_result = benchmark_fft(size)
            fft_result['size_name'] = size_name
            results['benchmarks'].append(fft_result)
            if 'speedup' in fft_result:
                total_speedup.append(fft_result['speedup'])
            
            # Correlation benchmark
            corr_result = benchmark_correlation(size)
            corr_result['size_name'] = size_name
            results['benchmarks'].append(corr_result)
            if 'speedup' in corr_result:
                total_speedup.append(corr_result['speedup'])
            
            # PSD benchmark
            psd_result = benchmark_psd(size)
            psd_result['size_name'] = size_name
            results['benchmarks'].append(psd_result)
            if 'speedup' in psd_result:
                total_speedup.append(psd_result['speedup'])
            
            # Array ops benchmark
            array_result = benchmark_array_ops(size)
            array_result['size_name'] = size_name
            results['benchmarks'].append(array_result)
            if 'speedup' in array_result:
                total_speedup.append(array_result['speedup'])
                
        except Exception as e:
            results['benchmarks'].append({
                'operation': 'Error',
                'size': size,
                'size_name': size_name,
                'error': str(e)
            })
    
    # Calculate summary
    if total_speedup:
        results['summary'] = {
            'avg_speedup': round(np.mean(total_speedup), 2),
            'max_speedup': round(np.max(total_speedup), 2),
            'min_speedup': round(np.min(total_speedup), 2),
        }
    
    return results

if __name__ == '__main__':
    try:
        results = run_full_benchmark()
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': str(e),
            'gpu_available': GPU_AVAILABLE
        }))
        sys.exit(1)
