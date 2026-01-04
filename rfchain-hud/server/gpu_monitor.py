#!/usr/bin/env python3
"""
GPU Monitoring Script for RFChain HUD
Provides real-time NVIDIA GPU metrics via nvidia-smi
"""

import subprocess
import json
import sys

def get_gpu_stats():
    """
    Query NVIDIA GPU statistics using nvidia-smi.
    Returns JSON with GPU memory, utilization, temperature, and power usage.
    """
    try:
        # Query nvidia-smi for GPU metrics
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': 'nvidia-smi command failed',
                'stderr': result.stderr
            }
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 10:
                gpu = {
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory': {
                        'total_mb': float(parts[2]),
                        'used_mb': float(parts[3]),
                        'free_mb': float(parts[4]),
                        'used_percent': round(float(parts[3]) / float(parts[2]) * 100, 1) if float(parts[2]) > 0 else 0
                    },
                    'utilization': {
                        'gpu_percent': float(parts[5]) if parts[5] != '[N/A]' else 0,
                        'memory_percent': float(parts[6]) if parts[6] != '[N/A]' else 0
                    },
                    'temperature_c': float(parts[7]) if parts[7] != '[N/A]' else 0,
                    'power': {
                        'draw_w': float(parts[8]) if parts[8] != '[N/A]' else 0,
                        'limit_w': float(parts[9]) if parts[9] != '[N/A]' else 0
                    }
                }
                gpus.append(gpu)
        
        return {
            'success': True,
            'gpu_count': len(gpus),
            'gpus': gpus
        }
        
    except FileNotFoundError:
        return {
            'success': False,
            'error': 'nvidia-smi not found - NVIDIA drivers may not be installed',
            'gpu_count': 0,
            'gpus': []
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'nvidia-smi timed out',
            'gpu_count': 0,
            'gpus': []
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'gpu_count': 0,
            'gpus': []
        }

def get_cuda_info():
    """
    Get CUDA version and CuPy availability info.
    """
    cuda_info = {
        'cuda_available': False,
        'cuda_version': None,
        'cupy_available': False,
        'cupy_version': None
    }
    
    try:
        # Check CUDA version from nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            cuda_info['cuda_available'] = True
            cuda_info['driver_version'] = result.stdout.strip()
    except:
        pass
    
    try:
        # Check CuPy availability
        import cupy as cp
        cuda_info['cupy_available'] = True
        cuda_info['cupy_version'] = cp.__version__
        cuda_info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
    except ImportError:
        pass
    except Exception as e:
        cuda_info['cupy_error'] = str(e)
    
    return cuda_info

if __name__ == '__main__':
    # Get GPU stats
    stats = get_gpu_stats()
    
    # Add CUDA info
    stats['cuda_info'] = get_cuda_info()
    
    # Output as JSON
    print(json.dumps(stats, indent=2))
