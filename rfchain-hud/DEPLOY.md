# RFChain HUD - Self-Hosted Deployment Guide

**Version:** 2.0 (Self-Hosted)  
**Target Platform:** Lubuntu 24.04 LTS  
**Author:** Manus AI  
**Last Updated:** January 2026

> **FORENSIC COMPLIANCE**: This version is designed for **fully self-hosted** operation with **no external cloud dependencies**. All data, authentication, and processing stays on your local machine.

---

## System Requirements

RFChain HUD is designed to run on high-performance workstations for RF signal analysis. The following specifications represent the target deployment environment:

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Operating System** | Lubuntu 24.04 LTS | Ubuntu-based distributions supported |
| **CPU** | Intel i9-13900K (24 cores) | Multi-core recommended for batch processing |
| **GPU** | NVIDIA RTX 4090 (24GB VRAM) | **Required for GPU acceleration** via CuPy/CUDA |
| **RAM** | 128 GB | Minimum 16 GB for basic operation |
| **Storage** | 4 TB NVMe SSD | Fast storage recommended for large signal files |
| **Python** | 3.12 | Managed via Conda environment |
| **Node.js** | 18.x or 20.x | LTS versions recommended |
| **CUDA** | 12.x+ | Required for GPU acceleration |

---

## Prerequisites Installation

### 1. Install Miniconda

Miniconda provides a lightweight Python environment manager. Download and install from the official repository:

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, accept license, and allow initialization
# Restart terminal or run:
source ~/.bashrc
```

### 2. Install Node.js

Node.js 18 or 20 LTS is required for the web application server:

```bash
# Using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version  # Should show v20.x.x
npm --version   # Should show 10.x.x
```

### 3. Install NVIDIA Drivers and CUDA (for GPU Acceleration)

For GPU-accelerated signal analysis with your RTX 4090:

```bash
# Add NVIDIA package repository
sudo apt update
sudo apt install -y nvidia-driver-545  # Or latest stable driver

# Reboot to load the driver
sudo reboot

# After reboot, verify driver installation
nvidia-smi
# Should show: Driver Version: 545.xx, CUDA Version: 12.x
```

The CUDA toolkit will be installed via conda (included in `environment.yml`).

### 4. Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential git curl wget
```

---

## Installation Steps

### Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/cvalentine99/RFChain.git
```

### Step 2: Set Up the Conda Environment

The conda environment includes all Python dependencies for signal analysis:

```bash
cd ~/RFChain/rfchain-hud

# Create the conda environment from the provided configuration
conda env create -f deploy/environment.yml

# Activate the environment
conda activate rfchain-hud

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Step 3: Install Node.js Dependencies

```bash
cd ~/RFChain/rfchain-hud

# Install npm packages
npm install
```

### Step 4: Configure Environment Variables

Copy the template configuration file and customize it for your environment:

```bash
cp deploy/.env.local.template .env.local
```

Edit `.env.local` with your preferred text editor:

```bash
nano .env.local
```

**Critical Configuration Options:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `PORT` | Server port | `3007` |
| `PYTHON_PATH` | Path to conda Python | `/home/youruser/miniconda3/envs/rfchain-hud/bin/python` |
| `ANALYSIS_SCRIPT_PATH` | Path to forensic analysis script | `/home/youruser/RFChain/analyze_signal_v2.2.2_forensic.py` |
| `JWT_SECRET` | Session encryption key | Generate with `openssl rand -base64 32` |
| `ANTHROPIC_API_KEY` | API key for JARVIS chat | Your Anthropic API key |

### Step 5: Set Up MySQL Database

RFChain HUD uses MySQL for robust data storage. Run the setup script:

```bash
./deploy/setup-mysql.sh
```

This script will:
- Install MySQL if not present
- Create the `rfchain_hud` database
- Create the `rfchain` user with a password you specify
- Generate `.env.local` with your configuration

Then run the database migration:

```bash
pnpm db:push
```

### Step 6: Build for Production

```bash
npm run build
```

---

## Running the Application

### Production Mode (Background)

Use the provided startup script to run RFChain HUD as a background service:

```bash
# Make scripts executable
chmod +x deploy/*.sh

# Start the server
./deploy/start.sh

# The application will be available at http://localhost:3007
```

To stop the server:

```bash
./deploy/stop.sh
```

### Development Mode (Foreground)

For development with hot-reload:

```bash
./deploy/dev.sh
```

Press `Ctrl+C` to stop the development server.

---

## Directory Structure

After installation, the project directory contains:

```
rfchain-hud/
├── deploy/                  # Deployment configuration
│   ├── environment.yml      # Conda environment specification
│   ├── .env.local.template  # Environment variable template
│   ├── start.sh             # Production startup script
│   ├── stop.sh              # Server shutdown script
│   └── dev.sh               # Development mode script
├── uploads/                 # Uploaded signal files (.bin, .raw, .iq)
├── analysis_output/         # Analysis results and visualizations
├── audio_uploads/           # Voice recordings for JARVIS
├── data/                    # SQLite database
├── logs/                    # Application logs
└── .env.local               # Local configuration (not in git)
```

---

## Feature Configuration

### JARVIS AI Assistant

The JARVIS chat assistant requires an LLM backend. For **forensic compliance**, we recommend using a local LLM.

**Option 1: Local LLM via Ollama (RECOMMENDED for Forensic Use)**

Ollama runs completely offline on your machine:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (llama3 recommended)
ollama pull llama3

# Add to .env.local:
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

**Option 2: Anthropic Claude (Cloud - requires internet)**
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Option 3: OpenAI GPT (Cloud - requires internet)**
```
OPENAI_API_KEY=sk-...
```

### Voice Input (Whisper)

Voice transcription uses the Whisper API. If you have an API key:
```
WHISPER_API_KEY=your-whisper-api-key
```

### RAG (Retrieval-Augmented Generation)

RAG is enabled by default. Configure thresholds in `.env.local`:
```
RAG_ENABLED=true
RAG_SIMILARITY_THRESHOLD=0.7
RAG_MAX_RESULTS=5
```

### GPU Acceleration (NVIDIA CUDA)

The signal analysis script (`analyze_signal_v2.2.2_forensic.py`) uses **CuPy** for GPU-accelerated DSP operations on your RTX 4090. The following operations are GPU-accelerated:

| Operation | Library | Speedup |
|-----------|---------|--------|
| FFT/IFFT computations | CuPy (`cp.fft.fft`) | 10-50x |
| Signal correlation | CuPy (`cp.correlate`) | 5-20x |
| Power spectral density | CuPy | 10-30x |
| Polyphase resampling | CuPy | 5-15x |
| Array operations | CuPy | 2-10x |

**Configuration in `.env.local`:**
```bash
# Enable GPU acceleration
GPU_ENABLED=true

# Select GPU device (0 = first GPU)
CUDA_VISIBLE_DEVICES=0

# CuPy accelerators for optimal performance
CUPY_ACCELERATORS=cub,cutensor

# GPU memory fraction (0.8 = 80% of 24GB = ~19GB)
GPU_MEMORY_FRACTION=0.8
```

**Verifying GPU Setup:**
```bash
# Activate conda environment
conda activate rfchain-hud

# Test CuPy installation
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA version: {cp.cuda.runtime.runtimeGetVersion()}'); print(f'GPU: {cp.cuda.Device(0).name}')"
```

Expected output:
```
CuPy version: 13.x.x
CUDA version: 12xxx
GPU: NVIDIA GeForce RTX 4090
```

**CPU Fallback:** If CuPy is not available or GPU initialization fails, the script automatically falls back to NumPy (CPU-only mode). This is indicated in the logs:
```
WARNING: GPU mode requested, but CuPy is not available. Falling back to CPU.
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Port 3007 already in use** | Run `./deploy/stop.sh` or `lsof -i :3007` to find the process |
| **Python module not found** | Ensure conda environment is activated: `conda activate rfchain-hud` |
| **Analysis script fails** | Verify `PYTHON_PATH` and `ANALYSIS_SCRIPT_PATH` in `.env.local` |
| **Database errors** | Delete `data/rfchain.db` and run `npm run db:push` to reinitialize |
| **Node.js version mismatch** | Use Node.js 18.x or 20.x LTS |
| **CuPy import error** | Ensure CUDA 12.x is installed: `nvidia-smi` should show driver version |
| **GPU out of memory** | Reduce `GPU_MEMORY_FRACTION` in `.env.local` or process smaller files |
| **CUDA version mismatch** | Reinstall CuPy: `conda install -c conda-forge cupy-cuda12x` |

### Checking Logs

Application logs are stored in the `logs/` directory:

```bash
# View recent logs
tail -f logs/rfchain-hud.log

# Search for errors
grep -i error logs/rfchain-hud.log
```

### Verifying the Installation

After starting the server, verify the installation:

1. Open a web browser and navigate to `http://localhost:3007`
2. The RFChain HUD dashboard should display with system status indicators
3. Upload a test `.bin` file to verify the analysis pipeline
4. Open JARVIS chat (Ctrl+J) to test the AI assistant

---

## Systemd Service (Auto-Start on Boot)

For production deployments, you can configure RFChain HUD to start automatically on system boot using systemd.

### Quick Installation

Run the provided installation script:

```bash
cd ~/rfchain-hud/deploy
sudo ./install-service.sh
```

This script will:
1. Create a dedicated `rfchain` service user
2. Copy project files to `/opt/rfchain-hud`
3. Install the systemd service unit
4. Set appropriate file permissions

### Manual Installation

If you prefer manual installation:

```bash
# Copy service file
sudo cp deploy/rfchain-hud.service /etc/systemd/system/

# Edit the service file to match your paths
sudo nano /etc/systemd/system/rfchain-hud.service

# Reload systemd
sudo systemctl daemon-reload
```

### Service Management Commands

| Command | Description |
|---------|-------------|
| `sudo systemctl start rfchain-hud` | Start the service |
| `sudo systemctl stop rfchain-hud` | Stop the service |
| `sudo systemctl restart rfchain-hud` | Restart the service |
| `sudo systemctl status rfchain-hud` | Check service status |
| `sudo systemctl enable rfchain-hud` | Enable auto-start on boot |
| `sudo systemctl disable rfchain-hud` | Disable auto-start |
| `sudo journalctl -u rfchain-hud -f` | View live logs |
| `sudo journalctl -u rfchain-hud --since "1 hour ago"` | View recent logs |

### Pre-Start Health Checks

The service includes automatic dependency verification before starting. The `check-deps.sh` script runs via `ExecStartPre` and verifies:

| Check | Type | Description |
|-------|------|-------------|
| Installation directory | Critical | Verifies `/opt/rfchain-hud` exists |
| Configuration file | Critical | Verifies `.env.local` is present |
| Node.js version | Critical | Requires Node.js 18+ |
| Node modules | Critical | Verifies dependencies are installed |
| Build artifacts | Warning | Checks for `dist/` directory |
| Conda environment | Warning | Verifies Python environment |
| GPU/NVIDIA driver | Warning | Checks for GPU acceleration |
| Database connectivity | Critical | Tests database connection |
| Port availability | Warning | Checks if port 3007 is free |
| Disk space | Critical/Warning | Alerts at 90%/95% usage |

**Critical failures** prevent the service from starting. **Warnings** allow startup with degraded functionality.

To run the health check manually:

```bash
sudo -u rfchain /opt/rfchain-hud/deploy/check-deps.sh
```

### Service Configuration

The service file (`/etc/systemd/system/rfchain-hud.service`) includes:

- **Pre-start health checks** via `ExecStartPre` (verifies dependencies)
- **Automatic restart** on failure (max 3 attempts per minute)
- **Security hardening** (NoNewPrivileges, ProtectSystem, PrivateTmp)
- **Resource limits** (4GB memory max, 400% CPU quota for 4 cores)
- **Logging** to systemd journal

To customize, edit the service file:

```bash
sudo systemctl edit rfchain-hud --full
sudo systemctl daemon-reload
sudo systemctl restart rfchain-hud
```

### Troubleshooting Service Issues

```bash
# Check service status and recent logs
sudo systemctl status rfchain-hud

# View detailed logs
sudo journalctl -u rfchain-hud -n 100 --no-pager

# Check if port 3007 is in use
sudo lsof -i :3007

# Verify environment file
sudo cat /opt/rfchain-hud/.env.local
```

---

## Updating the Application

To update to a newer version:

```bash
cd ~/RFChain
git pull origin main

cd rfchain-hud
npm install
npm run build

./deploy/stop.sh
./deploy/start.sh
```

---

## Security Considerations

### For Forensic Use

1. **Network Isolation**: Run on an air-gapped network if required for chain-of-custody
2. **Use Ollama**: Ensures all AI processing stays local (no data leaves your machine)
3. **Disable External Services**: Leave cloud API keys empty in `.env.local`
4. **Local Authentication**: The app uses local username/password authentication stored in MySQL (no external OAuth)
5. **First User = Admin**: The first user to register becomes the administrator

### General Security

1. **Generate a strong JWT secret** using `openssl rand -base64 32`
2. **Keep credentials secure** - never commit `.env.local` to version control
3. **Firewall configuration** - restrict access to localhost only:
   ```bash
   sudo ufw allow from 127.0.0.1 to any port 3007
   sudo ufw enable
   ```
4. **Regular updates** - keep Node.js, Python, and dependencies updated
5. **Database backups** - regularly backup MySQL and the `uploads/` directory

### Authentication

RFChain HUD uses **local authentication** by default:
- Username/password stored in MySQL with bcrypt hashing
- JWT session tokens (7-day expiry)
- No external OAuth dependencies
- First registered user automatically becomes admin
- Admins can create additional users

---

## Support

For issues and feature requests, please open an issue on the GitHub repository or contact the development team.

**Repository:** https://github.com/cvalentine99/RFChain
