#!/bin/bash
# RFChain HUD - Start Script for Local Deployment
# Runs on port 3007 with conda environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="rfchain-hud"
PORT=3007
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/.rfchain-hud.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RFChain HUD - Signal Intelligence System            ║"
echo "║                   Local Deployment Startup                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}[WARNING] RFChain HUD is already running (PID: $OLD_PID)${NC}"
        echo "Use ./stop.sh to stop the existing instance first."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Create necessary directories
echo -e "${CYAN}[1/6] Creating directories...${NC}"
mkdir -p "$PROJECT_DIR/uploads"
mkdir -p "$PROJECT_DIR/analysis_output"
mkdir -p "$PROJECT_DIR/audio_uploads"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$LOG_DIR"

# Check for .env.local
if [ ! -f "$PROJECT_DIR/.env.local" ]; then
    echo -e "${YELLOW}[WARNING] .env.local not found. Creating from template...${NC}"
    if [ -f "$PROJECT_DIR/deploy/.env.local.template" ]; then
        cp "$PROJECT_DIR/deploy/.env.local.template" "$PROJECT_DIR/.env.local"
        echo -e "${YELLOW}Please edit .env.local with your configuration before running again.${NC}"
        exit 1
    else
        echo -e "${RED}[ERROR] Template file not found. Please create .env.local manually.${NC}"
        exit 1
    fi
fi

# Initialize conda
echo -e "${CYAN}[2/6] Initializing conda environment...${NC}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    # Try common conda locations
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo -e "${RED}[ERROR] Conda not found. Please install Miniconda or Anaconda.${NC}"
        exit 1
    fi
fi

# Check if conda environment exists
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo -e "${YELLOW}[INFO] Creating conda environment '$CONDA_ENV_NAME'...${NC}"
    conda env create -f "$PROJECT_DIR/deploy/environment.yml"
fi

# Activate conda environment
echo -e "${CYAN}[3/6] Activating conda environment...${NC}"
conda activate "$CONDA_ENV_NAME"
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# Check Node.js
echo -e "${CYAN}[4/6] Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR] Node.js not found. Please install Node.js 18+ or 20+.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js: $(node --version)${NC}"

# Install/update npm dependencies
echo -e "${CYAN}[5/6] Installing dependencies...${NC}"
cd "$PROJECT_DIR"
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
    npm install --production
fi

# Build for production
echo -e "${CYAN}[6/6] Building application...${NC}"
npm run build 2>/dev/null || echo -e "${YELLOW}[INFO] Build step skipped (dev mode)${NC}"

# Export environment variables
export PORT=$PORT
export NODE_ENV=production
export DOTENV_CONFIG_PATH="$PROJECT_DIR/.env.local"

# Start the server
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Starting RFChain HUD on port $PORT              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Access the application at: ${GREEN}http://localhost:$PORT${NC}"
echo -e "${CYAN}Logs are saved to: ${GREEN}$LOG_DIR/rfchain-hud.log${NC}"
echo ""

# Start server in background and save PID
cd "$PROJECT_DIR"
nohup node dist/server/_core/index.js > "$LOG_DIR/rfchain-hud.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# Wait a moment and check if server started
sleep 3
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server started successfully (PID: $SERVER_PID)${NC}"
    echo -e "${CYAN}Use ${YELLOW}./deploy/stop.sh${CYAN} to stop the server${NC}"
else
    echo -e "${RED}[ERROR] Server failed to start. Check logs at: $LOG_DIR/rfchain-hud.log${NC}"
    rm -f "$PID_FILE"
    exit 1
fi
