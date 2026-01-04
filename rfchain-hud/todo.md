# RFChain HUD - Jarvis Signal Analyzer TODO

## Core Features
- [x] Jarvis/Iron Man HUD theme with dark background, cyan/blue glowing accents
- [x] Animated holographic panels with glowing edges
- [x] Real-time signal visualization dashboard

## Signal Visualization
- [x] Frequency spectrum chart (interactive)
- [x] Waterfall display
- [x] Constellation diagram
- [x] Time-domain waveform

## File Upload System
- [x] Drag-and-drop file upload for .bin/.raw/.iq files
- [x] Upload progress indicators
- [x] File validation and size checks

## RF Script Integration ✅ COMPLETED
- [x] Integration with analyze_signal_v2.2.2_forensic.py - WORKING!
- [x] Process uploaded signals and extract metrics
- [x] Store analysis results in database

## Signal Metrics Display
- [x] Power levels (avg, peak)
- [x] PAPR (Peak-to-Average Power Ratio)
- [x] I/Q imbalance
- [x] Bandwidth estimation
- [x] Frequency offset
- [x] Anomaly detection results

## RAG Chat Assistant
- [x] Chat box in lower-right corner
- [x] Message history with persistence
- [x] LLM integration (local LLM or API key)
- [x] Signal analysis context for RAG queries

## Voice Features
- [x] Voice synthesis (Web Speech API) for Jarvis-style responses
- [x] Voice input using Whisper API for questions

## Forensic Chain Visualization
- [x] Display 6 hash checkpoints
- [x] SHA-256 and SHA3-256 verification display
- [x] Chain of custody timeline

## Forensic Report Viewer
- [x] Interactive JSON viewer
- [x] Collapsible sections for metrics, anomalies, extended features
- [ ] Export/download capabilities

## Storage
- [x] S3 storage for analysis results
- [x] S3 storage for signal visualizations
- [ ] Automatic cleanup of old files

## Database Schema
- [x] Signal uploads table
- [x] Analysis results table
- [x] Chat messages table
- [x] Forensic reports table
- [x] LLM configurations table

## Python Script Integration
- [x] Clone RFChain repo and locate analyze_signal_v2.2.2_forensic.py
- [x] Create server-side Python execution endpoint (/api/analyze)
- [x] Update upload flow to trigger analysis after file upload
- [x] Parse analysis results and store in database
- [x] Update Analysis page to display real data with plot images
- [x] Test end-to-end signal upload and analysis flow (20 tests passing)

## Modulation Classification
- [x] Run digital analysis with --digital flag to enable modulation classification
- [x] Identify QAM constellation order: 128-QAM detected
- [x] Determine symbol rate from timing analysis: 76.76 ksps

## QAM Demodulation
- [x] Create 128-QAM demodulator with symbol timing recovery
- [x] Implement Gray-coded symbol-to-bit mapping
- [x] Extract and display first 100 bits of recovered data

## BER Analysis
- [x] Create BER analyzer with standard PRBS generators (PRBS-7, PRBS-9, PRBS-15, PRBS-23, PRBS-31)
- [x] Implement sequence synchronization and alignment
- [x] Calculate BER and report error statistics

## Signal Stage Analysis
- [x] Analyze raw signal characteristics (DC offset, frequency content, sample statistics)
- [x] Determine processing stage from filename and signal properties: Pre-FFT OFDM
- [x] Identify required DSP preprocessing steps
- [x] Recommend proper demodulation pipeline: OFDM with 256-point FFT, CP=8

## Script Integration
- [x] Integrate modulation classification into analyze_signal_v2.2.2_forensic.py
- [x] Add OFDM detection and processing capabilities
- [x] Add QAM demodulation with symbol timing recovery
- [x] Add BER analysis with PRBS sequence detection
- [x] Add signal stage detection and DSP pipeline recommendations

## App Signal Processing (REQUIRED) ✅ COMPLETED
- [x] Fix server errors (multer, __dirname, UV env vars - Claude API helped)
- [x] Test signal upload through the web UI
- [x] Verify Python analysis executes correctly
- [x] Display analysis results in dashboard with all 11 visualizations


## Local Storage Migration ✅ COMPLETED
- [x] Remove S3 storage dependency
- [x] Update upload endpoint to save files locally
- [x] Update analysis endpoint to read from local files
- [x] Store analysis outputs locally (/analysis_output/)
- [x] Test end-to-end local file pipeline - WORKING!


## Whisper Voice Input Integration ✅ COMPLETED
- [x] Review existing voice transcription helper in server/_core
- [x] Add microphone recording UI to JarvisChat component (MediaRecorder API)
- [x] Create tRPC endpoint for audio transcription via Whisper API (voice.transcribe)
- [x] Connect voice input to chat message flow
- [x] Add visual feedback for recording state (pulsing mic icon, recording timer)
- [x] Test voice commands end-to-end (26 tests passing)


## RAG System Implementation ✅ COMPLETED
- [x] Design RAG architecture for signal analysis context
- [x] Add database schema for vector embeddings (analysis_embeddings table)
- [x] Create embedding generation service using LLM API (server/_core/embeddings.ts)
- [x] Implement vector similarity search (cosine similarity)
- [x] Update chat router to retrieve relevant past analyses
- [x] Add automatic embedding generation on analysis completion
- [x] Backfill embeddings for existing analyses (embedding.backfill procedure)
- [x] Test semantic search with signal-related queries (40 tests passing)


## PDF Export for Forensic Reports ✅ COMPLETED
- [x] Create PDF generation service using pdfkit
- [x] Design forensic report template with chain-of-custody layout
- [x] Include hash verification section (SHA-256, SHA3-256)
- [x] Add analysis metrics and signal characteristics
- [x] Create download endpoint for PDF generation (forensic.generatePdf)
- [x] Add export button to Forensics page UI

## RAG Settings Panel ✅ COMPLETED
- [x] Add RAG settings to database schema (rag_settings table)
- [x] Create RAG settings UI component in Settings page
- [x] Add toggle for enabling/disabling RAG in chat
- [x] Add slider for similarity threshold adjustment (0-100%)
- [x] Add backfill button with progress indicator
- [x] Display embedding status (indexed vs pending)


## Signal Comparison View ✅ COMPLETED
- [x] Create Compare page with side-by-side layout
- [x] Add signal selection dropdowns for two analyses
- [x] Build metric comparison table with difference highlighting
- [x] Add visualization comparison (overlay mode and side-by-side)
- [x] Highlight significant differences (>10% deviation)
- [x] Add comparison summary with key findings


## Batch Signal Processing ✅ COMPLETED
- [x] Add database schema for batch jobs and queue items
- [x] Create batch processing router with queue management procedures
- [x] Update Upload page to support multiple file selection
- [x] Add batch progress tracking UI with real-time queue status
- [x] Implement sequential processing with error handling (pause/resume/retry)
- [x] Test batch upload and analysis flow (40 tests passing)


## Local Deployment (Lubuntu 24.04 + Conda) ✅ COMPLETED
- [x] Create conda environment.yml for Python 3.12 with all dependencies
- [x] Configure application to run on port 3007
- [x] Create startup scripts (start.sh, stop.sh, dev.sh, install.sh)
- [x] Create .env.local template for local deployment
- [x] Write comprehensive deployment documentation (DEPLOY.md)
- [x] Package all necessary files for local installation
- [x] Add GPU acceleration config (CuPy/CUDA 12.x for RTX 4090)
- [x] Document GPU-accelerated operations (FFT, correlation, PSD, resampling)
- [x] Add NVIDIA driver installation instructions


## GPU Monitoring & Benchmarking ✅ COMPLETED
- [x] Create GPU monitoring backend endpoint (nvidia-smi parsing)
- [x] Update dashboard UI with real-time VRAM usage display
- [x] Create GPU benchmark Python script (FFT, correlation, PSD tests)
- [x] Add benchmark results display to Settings page
- [x] Test GPU monitoring and benchmark features

## Benchmark History Storage ✅ COMPLETED
- [x] Add benchmark_history table to database schema
- [x] Create database helpers for saving/retrieving benchmark results
- [x] Update GPU router to persist benchmark results
- [x] Add benchmark history UI with charts to Settings page
- [x] Write tests for benchmark history feature (4 new tests)

## Benchmark Analytics & Export ✅ COMPLETED
- [x] Add Chart.js performance trend visualization (speedup over time)
- [x] Implement CSV export for benchmark history
- [x] Implement PDF export for benchmark reports
- [x] Create side-by-side benchmark comparison view
- [x] Add benchmark selection UI for comparison
- [x] Write tests for new features (11 new tests)


## Bug Fixes (End-to-End Review)
- [x] Fixed Analysis detail page - was using getBySignalId instead of getById (caused "Analysis not found" error)


## Forensic Hash Verification Fix ✅ COMPLETED
- [x] Investigate why hash values show "Not available" in UI
- [x] Verify hash values are being stored correctly from Python analysis (in forensicPipeline JSON)
- [x] Fix hash display in Forensics detail page (now reads from hash_chain array)

## Systemd Service for Auto-Startup ✅ COMPLETED
- [x] Create systemd unit file (rfchain-hud.service)
- [x] Create installation script (install-service.sh)
- [x] Add installation instructions to DEPLOY.md
- [x] Document service management commands

## Service Health Endpoint ✅ COMPLETED
- [x] Create /api/health endpoint for monitoring tools
- [x] Check database connectivity
- [x] Check GPU availability and status
- [x] Check system resources (CPU, memory, disk)
- [x] Return structured JSON response with status codes
- [x] Add tests for health endpoint (11 new tests)

## Systemd Pre-Start Health Check ✅ COMPLETED
- [x] Create pre-start health check script (check-deps.sh)
- [x] Verify database connectivity before start
- [x] Verify NVIDIA driver availability
- [x] Verify Node.js and conda environment
- [x] Update systemd service with ExecStartPre
- [x] Update DEPLOY.md with troubleshooting info


## Critical Deployment Fixes (URGENT)
- [ ] Fix database dialect mismatch - MySQL in code vs SQLite in env template
- [ ] Add authentication to upload/analysis endpoints (currently public)
- [ ] Fix Python path - remove hardcoded /usr/bin/python3.11, use env vars
- [ ] Fix .env.local template - add all required env vars
- [ ] Fix start.sh - proper production mode with build
- [ ] Add rate limiting and file validation to uploads
- [ ] Fix port binding - don't auto-select, fail if busy
- [ ] Fix createSignalUpload null handling
- [ ] Fix storage helpers for missing env vars gracefully


## Self-Hosted Forensic Mode (Air-Gapped) ✅ COMPLETED
- [x] Replace Manus OAuth with local username/password authentication
- [x] Add bcrypt password hashing for secure local auth
- [x] Create local user registration and login pages (first user = admin)
- [x] Keep MySQL database (more robust than SQLite for production)
- [x] Add local filesystem storage fallback (no S3 required)
- [x] Multi-LLM backend support (Ollama for offline, Anthropic/OpenAI optional)
- [x] Make JARVIS AI work with local LLM (Ollama) for fully offline operation
- [x] Update .env.local template for standalone operation
- [x] Update deployment scripts for air-gapped installation
- [x] Create comprehensive DEPLOY.md for self-hosted deployment
- [x] Add tests for local authentication (9 tests passing)
- [x] All 99 tests passing

## RAG v2 - Real Vector Search for Signal Data
- [ ] Add FAISS vector store for local/offline vector search
- [ ] Create signal metadata document schema for embedding
- [ ] Implement chunking pipeline for spectral analysis results
- [ ] Generate embeddings for all signal metadata fields
- [ ] Store embeddings in FAISS index with signal ID mapping
- [ ] Implement vector similarity search function
- [ ] Update JARVIS chat to use RAG retrieval with context
- [ ] Auto-index new analyses on completion
- [ ] Add RAG search endpoint for direct queries
- [ ] Test with real signal data
