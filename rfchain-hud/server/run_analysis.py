#!/usr/bin/env python3
"""
RF Signal Analysis Runner
Wrapper script to execute the analyze_signal_v2.2.2_forensic.py script
and return results as JSON for the web application.
"""

import sys
import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

def run_analysis(
    input_file: str,
    output_dir: str = None,
    sample_rate: float = 1e6,
    center_freq: float = 0.0,
    data_format: str = "complex64",
    fft_size: int = 4096,
    enable_digital: bool = False,
    enable_v3: bool = False
) -> dict:
    """
    Run the RF signal analysis script on an input file.
    
    Args:
        input_file: Path to the signal file (.bin, .raw, .iq)
        output_dir: Directory for output files (default: temp dir)
        sample_rate: Sample rate in Hz
        center_freq: Center frequency in Hz
        data_format: Data format (complex64, int16, int8, float32)
        fft_size: FFT size for spectral analysis
        enable_digital: Enable digital signal analysis
        enable_v3: Enable v3 enhanced analysis
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        return {
            "success": False,
            "error": f"Input file not found: {input_file}",
            "timestamp": datetime.now().isoformat()
        }
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="rf_analysis_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = Path(output_dir)
    
    # Build command
    script_path = Path(__file__).parent / "analyze_signal.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        str(input_path),
        "-r", str(sample_rate),
        "-c", str(center_freq),
        "-f", data_format,
        "--fft-size", str(fft_size),
        "-o", str(output_path),
        "--dark"
    ]
    
    if enable_digital:
        cmd.append("--digital")
    
    if enable_v3:
        cmd.append("--v3")
    
    try:
        # Run the analysis
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(script_path.parent)
        )
        
        # Check for errors
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Analysis failed: {result.stderr}",
                "stdout": result.stdout,
                "timestamp": datetime.now().isoformat()
            }
        
        # Find the metrics JSON file
        metrics_files = list(output_path.glob("*_metrics.json"))
        if not metrics_files:
            return {
                "success": False,
                "error": "No metrics file generated",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        
        # Load the metrics
        metrics_file = metrics_files[0]
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # Find generated plot files
        plot_files = list(output_path.glob("*.png"))
        plot_paths = [str(p) for p in plot_files]
        
        # Find hash sidecar file if exists
        hash_file = metrics_file.with_suffix('.sha256')
        hash_data = None
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                hash_data = f.read()
        
        return {
            "success": True,
            "metrics_file": str(metrics_file),
            "metrics": metrics_data,
            "plot_files": plot_paths,
            "hash_verification": hash_data,
            "output_dir": str(output_path),
            "stdout": result.stdout,
            "timestamp": datetime.now().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Analysis timed out after 5 minutes",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def main():
    """CLI interface for the analysis runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RF Signal Analysis Runner")
    parser.add_argument("input_file", help="Path to signal file")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-r", "--sample-rate", type=float, default=1e6, help="Sample rate (Hz)")
    parser.add_argument("-c", "--center-freq", type=float, default=0.0, help="Center frequency (Hz)")
    parser.add_argument("-f", "--format", default="complex64", help="Data format")
    parser.add_argument("--fft-size", type=int, default=4096, help="FFT size")
    parser.add_argument("-d", "--digital", action="store_true", help="Enable digital analysis")
    parser.add_argument("--v3", action="store_true", help="Enable v3 analysis")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    result = run_analysis(
        input_file=args.input_file,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        data_format=args.format,
        fft_size=args.fft_size,
        enable_digital=args.digital,
        enable_v3=args.v3
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["success"]:
            print(f"Analysis complete!")
            print(f"Metrics file: {result['metrics_file']}")
            print(f"Plot files: {len(result['plot_files'])}")
            if result.get('hash_verification'):
                print(f"Hash verification available")
        else:
            print(f"Analysis failed: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
