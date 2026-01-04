#!/usr/bin/env python3
"""
FAISS Vector Store for RFChain HUD RAG System

This module provides local/offline vector storage and similarity search
for signal analysis metadata and spectral results.

Uses FAISS (Facebook AI Similarity Search) for efficient vector operations.
Supports both CPU and GPU (CUDA) acceleration.
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np

# Try to import FAISS with GPU support, fall back to CPU
try:
    import faiss
    FAISS_AVAILABLE = True
    # Check for GPU support
    try:
        res = faiss.StandardGpuResources()
        FAISS_GPU = True
    except:
        FAISS_GPU = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False
    print("Warning: FAISS not installed. Run: pip install faiss-cpu (or faiss-gpu)", file=sys.stderr)

# Default paths
DEFAULT_INDEX_PATH = Path(__file__).parent.parent / "data" / "faiss_index"
DEFAULT_METADATA_PATH = Path(__file__).parent.parent / "data" / "faiss_metadata.json"

# Embedding dimensions (OpenAI text-embedding-3-small = 1536)
EMBEDDING_DIM = 1536


class SignalVectorStore:
    """
    FAISS-based vector store for signal analysis data.
    
    Stores embeddings with associated metadata for semantic search
    over signal characteristics, metrics, and analysis results.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        use_gpu: bool = True,
        embedding_dim: int = EMBEDDING_DIM
    ):
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and FAISS_GPU
        
        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load index
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_idx: Dict[str, int] = {}  # Maps analysis_id to index position
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        index_file = self.index_path.with_suffix('.index')
        
        if index_file.exists() and self.metadata_path.exists():
            # Load existing index
            self.index = faiss.read_index(str(index_file))
            
            # Move to GPU if available
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', [])
                self.id_to_idx = {str(m['analysis_id']): i for i, m in enumerate(self.metadata)}
            
            print(f"Loaded FAISS index with {self.index.ntotal} vectors", file=sys.stderr)
        else:
            # Create new index
            # Using IndexFlatIP (Inner Product) for cosine similarity
            # (vectors should be L2-normalized before adding)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            self.metadata = []
            self.id_to_idx = {}
            
            print(f"Created new FAISS index (dim={self.embedding_dim})", file=sys.stderr)
    
    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize vector for cosine similarity."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
    
    def add(
        self,
        analysis_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a signal analysis embedding to the index.
        
        Args:
            analysis_id: Unique identifier for the analysis
            embedding: Vector embedding (1536 dimensions)
            metadata: Associated metadata (filename, metrics, etc.)
        
        Returns:
            True if added successfully
        """
        if not FAISS_AVAILABLE:
            print("Error: FAISS not available", file=sys.stderr)
            return False
        
        # Check if already exists
        if str(analysis_id) in self.id_to_idx:
            # Update existing entry
            return self.update(analysis_id, embedding, metadata)
        
        # Convert and normalize embedding
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        vec = self._normalize_vector(vec)
        
        # Add to index
        self.index.add(vec)
        
        # Store metadata
        meta_entry = {
            'analysis_id': str(analysis_id),
            'added_at': datetime.now().isoformat(),
            **metadata
        }
        self.metadata.append(meta_entry)
        self.id_to_idx[str(analysis_id)] = len(self.metadata) - 1
        
        return True
    
    def update(
        self,
        analysis_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update an existing embedding.
        
        Note: FAISS doesn't support in-place updates, so we rebuild
        the index periodically. For now, we just update metadata.
        """
        if str(analysis_id) not in self.id_to_idx:
            return self.add(analysis_id, embedding, metadata)
        
        idx = self.id_to_idx[str(analysis_id)]
        self.metadata[idx] = {
            'analysis_id': str(analysis_id),
            'updated_at': datetime.now().isoformat(),
            **metadata
        }
        
        # TODO: For full update, need to rebuild index
        return True
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar signal analyses.
        
        Args:
            query_embedding: Query vector (1536 dimensions)
            k: Number of results to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of matches with metadata and similarity scores
        """
        if not FAISS_AVAILABLE or self.index.ntotal == 0:
            return []
        
        # Convert and normalize query
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query = self._normalize_vector(query)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            # Distance is cosine similarity (since we normalized)
            similarity = float(dist)
            
            if similarity >= threshold:
                results.append({
                    'similarity': similarity,
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def search_by_metrics(
        self,
        query_embedding: List[float],
        k: int = 5,
        threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with optional metadata filters.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            threshold: Minimum similarity
            filters: Optional filters like {'has_anomalies': True}
        
        Returns:
            Filtered search results
        """
        # Get more results than needed to allow for filtering
        results = self.search(query_embedding, k=k*3, threshold=threshold)
        
        if not filters:
            return results[:k]
        
        # Apply filters
        filtered = []
        for r in results:
            meta = r['metadata']
            match = True
            
            for key, value in filters.items():
                if key in meta and meta[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(r)
                if len(filtered) >= k:
                    break
        
        return filtered
    
    def get_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific analysis."""
        if str(analysis_id) in self.id_to_idx:
            idx = self.id_to_idx[str(analysis_id)]
            return self.metadata[idx]
        return None
    
    def delete(self, analysis_id: str) -> bool:
        """
        Mark an entry as deleted.
        
        Note: FAISS doesn't support deletion, so we mark in metadata.
        Periodic compaction should be run to rebuild the index.
        """
        if str(analysis_id) in self.id_to_idx:
            idx = self.id_to_idx[str(analysis_id)]
            self.metadata[idx]['deleted'] = True
            self.metadata[idx]['deleted_at'] = datetime.now().isoformat()
            return True
        return False
    
    def save(self) -> bool:
        """Save index and metadata to disk."""
        if not FAISS_AVAILABLE:
            return False
        
        try:
            # Move to CPU for saving if on GPU
            index_to_save = self.index
            if self.use_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            
            # Save index
            index_file = self.index_path.with_suffix('.index')
            faiss.write_index(index_to_save, str(index_file))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'embedding_dim': self.embedding_dim,
                    'total_vectors': self.index.ntotal,
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"Saved FAISS index ({self.index.ntotal} vectors) to {index_file}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Error saving index: {e}", file=sys.stderr)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal if FAISS_AVAILABLE else 0,
            'embedding_dim': self.embedding_dim,
            'gpu_enabled': self.use_gpu,
            'index_path': str(self.index_path),
            'metadata_count': len(self.metadata),
            'deleted_count': sum(1 for m in self.metadata if m.get('deleted', False))
        }
    
    def compact(self) -> bool:
        """
        Rebuild index without deleted entries.
        
        This is an expensive operation but necessary to reclaim space.
        """
        if not FAISS_AVAILABLE:
            return False
        
        # Filter out deleted entries
        active_metadata = [m for m in self.metadata if not m.get('deleted', False)]
        
        if len(active_metadata) == len(self.metadata):
            print("No deleted entries to compact", file=sys.stderr)
            return True
        
        # TODO: Would need to re-embed all entries or store embeddings in metadata
        # For now, just update metadata
        print(f"Compaction would remove {len(self.metadata) - len(active_metadata)} entries", file=sys.stderr)
        return True


def format_signal_document(
    analysis_id: str,
    filename: str,
    metrics: Dict[str, Any],
    anomalies: Optional[Dict[str, Any]] = None,
    forensic_data: Optional[Dict[str, Any]] = None,
    spectral_features: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format signal analysis data into a document for embedding.
    
    This creates a rich text representation that captures:
    - Signal identification
    - Core metrics (power, bandwidth, SNR, etc.)
    - Anomaly detection results
    - Spectral characteristics
    - Forensic chain-of-custody data
    
    Args:
        analysis_id: Unique analysis identifier
        filename: Original signal filename
        metrics: Core analysis metrics
        anomalies: Detected anomalies
        forensic_data: Chain-of-custody information
        spectral_features: Spectral analysis results
    
    Returns:
        Formatted document string for embedding
    """
    sections = []
    
    # Header
    sections.append(f"Signal Analysis Report: {filename}")
    sections.append(f"Analysis ID: {analysis_id}")
    sections.append("")
    
    # Core Metrics Section
    sections.append("=== SIGNAL METRICS ===")
    
    if metrics.get('avg_power_dbm') is not None:
        sections.append(f"Average Power: {metrics['avg_power_dbm']:.2f} dBm")
    if metrics.get('peak_power_dbm') is not None:
        sections.append(f"Peak Power: {metrics['peak_power_dbm']:.2f} dBm")
    if metrics.get('papr_db') is not None:
        sections.append(f"PAPR: {metrics['papr_db']:.2f} dB")
    if metrics.get('snr_estimate_db') is not None:
        sections.append(f"SNR Estimate: {metrics['snr_estimate_db']:.2f} dB")
    if metrics.get('bandwidth_hz') is not None:
        bw_khz = metrics['bandwidth_hz'] / 1000
        sections.append(f"Bandwidth: {bw_khz:.2f} kHz")
    if metrics.get('freq_offset_hz') is not None:
        sections.append(f"Frequency Offset: {metrics['freq_offset_hz']:.2f} Hz")
    if metrics.get('iq_imbalance_db') is not None:
        sections.append(f"I/Q Imbalance: {metrics['iq_imbalance_db']:.2f} dB")
    if metrics.get('sample_count') is not None:
        sections.append(f"Sample Count: {metrics['sample_count']:,}")
    if metrics.get('duration_ms') is not None:
        sections.append(f"Duration: {metrics['duration_ms']:.2f} ms")
    
    sections.append("")
    
    # Anomaly Section
    if anomalies:
        sections.append("=== ANOMALY DETECTION ===")
        
        detected = []
        if anomalies.get('dc_spike'):
            detected.append("DC Spike detected")
        if anomalies.get('saturation'):
            detected.append("Signal Saturation/Clipping detected")
        if anomalies.get('dropout'):
            detected.append("Signal Dropout detected")
        if anomalies.get('periodic_interference'):
            detected.append("Periodic Interference detected")
        
        if detected:
            for d in detected:
                sections.append(f"- {d}")
        else:
            sections.append("No anomalies detected")
        
        # Include details if available
        if anomalies.get('details'):
            for detail in anomalies['details']:
                sections.append(f"  Detail: {detail}")
        
        sections.append("")
    
    # Spectral Features Section
    if spectral_features:
        sections.append("=== SPECTRAL CHARACTERISTICS ===")
        
        if spectral_features.get('modulation_type'):
            sections.append(f"Modulation Type: {spectral_features['modulation_type']}")
        if spectral_features.get('symbol_rate'):
            sections.append(f"Symbol Rate: {spectral_features['symbol_rate']} sps")
        if spectral_features.get('ofdm_detected'):
            sections.append("OFDM Signal Detected")
            if spectral_features.get('ofdm_fft_size'):
                sections.append(f"  FFT Size: {spectral_features['ofdm_fft_size']}")
            if spectral_features.get('ofdm_cp_length'):
                sections.append(f"  Cyclic Prefix: {spectral_features['ofdm_cp_length']}")
        if spectral_features.get('carrier_frequency'):
            sections.append(f"Carrier Frequency: {spectral_features['carrier_frequency']} Hz")
        if spectral_features.get('noise_floor_db'):
            sections.append(f"Noise Floor: {spectral_features['noise_floor_db']:.2f} dB")
        
        sections.append("")
    
    # Forensic Section
    if forensic_data:
        sections.append("=== FORENSIC CHAIN OF CUSTODY ===")
        
        if forensic_data.get('raw_input_hash'):
            sections.append(f"Input Hash (SHA-256): {forensic_data['raw_input_hash'][:16]}...")
        if forensic_data.get('output_hash'):
            sections.append(f"Output Hash (SHA-256): {forensic_data['output_hash'][:16]}...")
        if forensic_data.get('analysis_timestamp'):
            sections.append(f"Analysis Timestamp: {forensic_data['analysis_timestamp']}")
        if forensic_data.get('analyst_id'):
            sections.append(f"Analyst ID: {forensic_data['analyst_id']}")
        
        sections.append("")
    
    # Signal Classification Summary
    sections.append("=== SIGNAL CLASSIFICATION ===")
    
    # Infer signal type from metrics
    signal_type = "Unknown"
    if metrics.get('bandwidth_hz'):
        bw = metrics['bandwidth_hz']
        if bw < 10000:
            signal_type = "Narrowband"
        elif bw < 100000:
            signal_type = "Standard bandwidth"
        else:
            signal_type = "Wideband"
    
    sections.append(f"Signal Type: {signal_type}")
    
    # Quality assessment
    quality = "Unknown"
    snr = metrics.get('snr_estimate_db')
    if snr is not None:
        if snr > 20:
            quality = "High quality (SNR > 20 dB)"
        elif snr > 10:
            quality = "Medium quality (SNR 10-20 dB)"
        else:
            quality = "Low quality (SNR < 10 dB)"
    
    sections.append(f"Signal Quality: {quality}")
    
    return "\n".join(sections)


def main():
    """CLI interface for the vector store."""
    parser = argparse.ArgumentParser(description="FAISS Vector Store for RFChain HUD")
    parser.add_argument('command', choices=['add', 'search', 'stats', 'save', 'compact'],
                       help='Command to execute')
    parser.add_argument('--analysis-id', '-a', help='Analysis ID')
    parser.add_argument('--embedding', '-e', help='Embedding vector (JSON array)')
    parser.add_argument('--metadata', '-m', help='Metadata (JSON object)')
    parser.add_argument('--query', '-q', help='Query embedding (JSON array)')
    parser.add_argument('--k', '-k', type=int, default=5, help='Number of results')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--index-path', help='Path to FAISS index')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Initialize store
    index_path = Path(args.index_path) if args.index_path else None
    store = SignalVectorStore(index_path=index_path, use_gpu=not args.no_gpu)
    
    if args.command == 'add':
        if not args.analysis_id or not args.embedding:
            print("Error: --analysis-id and --embedding required for add", file=sys.stderr)
            sys.exit(1)
        
        embedding = json.loads(args.embedding)
        metadata = json.loads(args.metadata) if args.metadata else {}
        
        success = store.add(args.analysis_id, embedding, metadata)
        store.save()
        
        print(json.dumps({'success': success, 'total_vectors': store.index.ntotal}))
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query required for search", file=sys.stderr)
            sys.exit(1)
        
        query = json.loads(args.query)
        results = store.search(query, k=args.k, threshold=args.threshold)
        
        print(json.dumps({'results': results, 'count': len(results)}))
    
    elif args.command == 'stats':
        stats = store.get_stats()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'save':
        success = store.save()
        print(json.dumps({'success': success}))
    
    elif args.command == 'compact':
        success = store.compact()
        store.save()
        print(json.dumps({'success': success}))


if __name__ == '__main__':
    main()
