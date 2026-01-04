#!/usr/bin/env python3
"""
Batch Signal Processor
Processes multiple signal files in parallel using GPU acceleration.
Supports queue-based processing with progress tracking.
"""

import os
import sys
import json
import argparse
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import traceback

# Try to import CuPy for GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import numpy as np

@dataclass
class BatchItem:
    """Represents a single item in the batch queue"""
    id: int
    filename: str
    local_path: str
    status: str = 'queued'
    error_message: Optional[str] = None
    analysis_result_id: Optional[int] = None

@dataclass
class BatchJob:
    """Represents a batch processing job"""
    id: int
    name: str
    total_files: int
    completed_files: int = 0
    failed_files: int = 0
    status: str = 'pending'
    sample_rate: float = 1e6
    data_format: str = 'complex64'
    digital_analysis: bool = False
    v3_analysis: bool = False

class BatchProcessor:
    """Processes batch jobs with parallel GPU execution"""
    
    def __init__(self, 
                 max_parallel: int = 2,
                 output_base_dir: str = '/tmp/rfchain-batch',
                 analysis_script: Optional[str] = None):
        self.max_parallel = max_parallel
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Find analysis script
        if analysis_script:
            self.analysis_script = Path(analysis_script)
        else:
            # Look for script in same directory
            script_dir = Path(__file__).parent
            candidates = [
                script_dir / 'analyze_signal.py',
                script_dir / 'analyze_signal_v2.2.2_forensic.py',
            ]
            self.analysis_script = None
            for c in candidates:
                if c.exists():
                    self.analysis_script = c
                    break
        
        if not self.analysis_script or not self.analysis_script.exists():
            raise FileNotFoundError("Analysis script not found")
        
        self.results: Dict[int, Dict[str, Any]] = {}
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def _report_progress(self, item_id: int, status: str, message: str = '', result: Dict = None):
        """Report progress to callback"""
        if self.progress_callback:
            self.progress_callback({
                'item_id': item_id,
                'status': status,
                'message': message,
                'result': result,
            })
        print(f"[{status.upper()}] Item {item_id}: {message}", file=sys.stderr)
    
    def process_single(self, item: BatchItem, job: BatchJob) -> Dict[str, Any]:
        """Process a single signal file"""
        import subprocess
        
        start_time = time.time()
        output_dir = self.output_base_dir / f"job_{job.id}" / f"item_{item.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._report_progress(item.id, 'processing', f'Starting analysis of {item.filename}')
        
        try:
            # Build command
            cmd = [
                sys.executable,
                str(self.analysis_script),
                item.local_path,
                '-o', str(output_dir),
                '-s', str(job.sample_rate),
                '-f', job.data_format,
            ]
            
            if job.digital_analysis:
                cmd.append('--digital')
            if job.v3_analysis:
                cmd.append('--v3')
            
            # Run analysis
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per file
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else 'Unknown error'
                self._report_progress(item.id, 'failed', f'Analysis failed: {error_msg}')
                return {
                    'success': False,
                    'item_id': item.id,
                    'error': error_msg,
                    'elapsed_seconds': elapsed,
                }
            
            # Find metrics file
            metrics_file = output_dir / f"{Path(item.filename).stem}_metrics.json"
            metrics = {}
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
            
            # Find plot files
            plot_files = list(output_dir.glob('*.png'))
            plot_urls = [str(p) for p in plot_files]
            
            self._report_progress(item.id, 'completed', 
                f'Analysis complete in {elapsed:.1f}s, {len(plot_files)} plots generated')
            
            return {
                'success': True,
                'item_id': item.id,
                'metrics': metrics,
                'plot_urls': plot_urls,
                'output_dir': str(output_dir),
                'elapsed_seconds': elapsed,
            }
            
        except subprocess.TimeoutExpired:
            self._report_progress(item.id, 'failed', 'Analysis timed out after 10 minutes')
            return {
                'success': False,
                'item_id': item.id,
                'error': 'Analysis timed out',
                'elapsed_seconds': 600,
            }
        except Exception as e:
            self._report_progress(item.id, 'failed', f'Exception: {str(e)}')
            return {
                'success': False,
                'item_id': item.id,
                'error': str(e),
                'elapsed_seconds': time.time() - start_time,
            }
    
    def process_batch_sequential(self, job: BatchJob, items: List[BatchItem]) -> List[Dict[str, Any]]:
        """Process batch items sequentially (safer for GPU memory)"""
        results = []
        for item in items:
            result = self.process_single(item, job)
            results.append(result)
            self.results[item.id] = result
        return results
    
    def process_batch_parallel(self, job: BatchJob, items: List[BatchItem]) -> List[Dict[str, Any]]:
        """Process batch items in parallel using thread pool"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(self.process_single, item, job): item
                for item in items
            }
            
            for future in futures:
                item = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.results[item.id] = result
                except Exception as e:
                    error_result = {
                        'success': False,
                        'item_id': item.id,
                        'error': str(e),
                    }
                    results.append(error_result)
                    self.results[item.id] = error_result
        
        return results
    
    def process_batch(self, job: BatchJob, items: List[BatchItem], 
                      parallel: bool = False) -> Dict[str, Any]:
        """Process a complete batch job"""
        start_time = time.time()
        
        print(f"Starting batch job {job.id}: {job.name}", file=sys.stderr)
        print(f"  Total files: {job.total_files}", file=sys.stderr)
        print(f"  Parallel: {parallel} (max {self.max_parallel})", file=sys.stderr)
        print(f"  GPU available: {GPU_AVAILABLE}", file=sys.stderr)
        
        if parallel and self.max_parallel > 1:
            results = self.process_batch_parallel(job, items)
        else:
            results = self.process_batch_sequential(job, items)
        
        elapsed = time.time() - start_time
        
        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful
        
        summary = {
            'job_id': job.id,
            'total_files': len(items),
            'successful': successful,
            'failed': failed,
            'elapsed_seconds': elapsed,
            'avg_time_per_file': elapsed / len(items) if items else 0,
            'results': results,
        }
        
        print(f"\nBatch job {job.id} complete:", file=sys.stderr)
        print(f"  Successful: {successful}/{len(items)}", file=sys.stderr)
        print(f"  Failed: {failed}", file=sys.stderr)
        print(f"  Total time: {elapsed:.1f}s", file=sys.stderr)
        print(f"  Avg per file: {summary['avg_time_per_file']:.1f}s", file=sys.stderr)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Batch Signal Processor')
    parser.add_argument('--job-file', '-j', required=True,
                       help='JSON file containing job and items data')
    parser.add_argument('--output-dir', '-o', default='/tmp/rfchain-batch',
                       help='Base output directory')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max-parallel', '-m', type=int, default=2,
                       help='Maximum parallel processes')
    parser.add_argument('--analysis-script', '-a',
                       help='Path to analysis script')
    
    args = parser.parse_args()
    
    # Load job data
    with open(args.job_file) as f:
        data = json.load(f)
    
    job = BatchJob(**data['job'])
    items = [BatchItem(**item) for item in data['items']]
    
    # Create processor
    processor = BatchProcessor(
        max_parallel=args.max_parallel,
        output_base_dir=args.output_dir,
        analysis_script=args.analysis_script,
    )
    
    # Process batch
    summary = processor.process_batch(job
, items, parallel=args.parallel)
    
    # Output summary as JSON
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
