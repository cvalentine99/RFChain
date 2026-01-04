#!/bin/bash
# ============================================================================
# RFChain HUD - Production Start Script
# Forensic Signal Intelligence System - Self-Hosted
# ============================================================================
# Runs on port 3007 (configurable via .env.local)
# Requires: Node.js 18+, MySQL, Conda (for GPU acceleration)
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="rfchain-hud"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/.rfchain-hud.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RFChain HUD - Signal Intelligence System            ║"
echo "║                   Self-Hosted Deployment                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}[WARNING] RFChain HUD is already running (PID: $OLD_PID)${NC}"
        echo "Use ./deploy/stop.sh to stop the existing instance first."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Create necessary directories
echo -e "${CYAN}[1/7] Creating directories...${NC}"
mkdir -p "$PROJECT_DIR/uploads"
mkdir -p "$PROJECT_DIR/analysis_output"
mkdir -p "$PROJECT_DIR/audio_uploads"
mkdir -p "$PROJECT_DIR/storage_data"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓ Directories created${NC}"

# Check for .env.local
echo -e "${CYAN}[2/7] Checking configuration...${NC}"
if [ ! -f "$PROJECT_DIR/.env.local" ]; then
    echo -e "${RED}[ERROR] .env.local not found!${NC}"
    echo -e "${YELLOW}Run ./deploy/setup-mysql.sh first to create the configuration.${NC}"
    exit 1
fi

# Source environment variables
set -a
source "$PROJECT_DIR/.env.local"
set +a

# Verify required variables
if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}[ERROR] DATABASE_URL not set in .env.local${NC}"
    exit 1
fi

if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "GENERATE_A_SECURE_SECRET_HERE" ]; then
    echo -e "${RED}[ERROR] JWT_SECRET not properly configured in .env.local${NC}"
    echo -e "${YELLOW}Generate one with: openssl rand -base64 32${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration valid${NC}"

# Initialize conda (optional for GPU support)
echo -e "${CYAN}[3/7] Checking Python/Conda environment...${NC}"
CONDA_AVAILABLE=false
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    CONDA_AVAILABLE=true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_AVAILABLE=true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_AVAILABLE=true
fi

if [ "$CONDA_AVAILABLE" = true ]; then
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        conda activate "$CONDA_ENV_NAME" 2>/dev/null && echo -e "${GREEN}✓ Conda environment '$CONDA_ENV_NAME' activated${NC}" || echo -e "${YELLOW}[INFO] Using system Python${NC}"
    else
        echo -e "${YELLOW}[INFO] Conda environment '$CONDA_ENV_NAME' not found${NC}"
        echo -e "${YELLOW}Create it with: conda env create -f deploy/environment.yml${NC}"
    fi
else
    echo -e "${YELLOW}[INFO] Conda not available, using system Python${NC}"
fi

if command -v python &> /dev/null; then
    echo -e "${GREEN}✓ Python: $(python --version 2>&1)${NC}"
fi

# Check Node.js
echo -e "${CYAN}[4/7] Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR] Node.js not found. Please install Node.js 18+ or 20+.${NC}"
    exit 1
fi
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}[ERROR] Node.js 18+ required, found $(node --version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js: $(node --version)${NC}"

# Install dependencies if needed
echo -e "${CYAN}[5/7] Checking dependencies...${NC}"
cd "$PROJECT_DIR"
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    if command -v pnpm &> /dev/null; then
        pnpm install --prod
    else
        npm install --production
    fi
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Build for production
echo -e "${CYAN}[6/7] Building application...${NC}"
if [ ! -d "dist" ] || [ "$(find server -name '*.ts' -newer dist 2>/dev/null | head -1)" ]; then
    echo -e "${YELLOW}Building...${NC}"
    if command -v pnpm &> /dev/null; then
        pnpm build
    else
        npm run build
    fi
fi
echo -e "${GREEN}✓ Build complete${NC}"

# Get port from config
PORT="${PORT:-3007}"

# Start the server
echo -e "${CYAN}[7/7] Starting server...${NC}"

export NODE_ENV=production
export DOTENV_CONFIG_PATH="$PROJECT_DIR/.env.local"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Starting RFChain HUD on port $PORT              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Start server in background and save PID
cd "$PROJECT_DIR"
# Run the built server (esbuild outputs to dist/index.js)
nohup node dist/index.js > "$LOG_DIR/rfchain-hud.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# Wait a moment and check if server started
sleep 3
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server started successfully (PID: $SERVER_PID)${NC}"
    echo ""
    echo -e "${CYAN}Access the application at: ${GREEN}http://localhost:$PORT${NC}"
    echo -e "${CYAN}Logs: ${GREEN}$LOG_DIR/rfchain-hud.log${NC}"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo -e "  Stop server:    ${YELLOW}./deploy/stop.sh${NC}"
    echo -e "  View logs:      ${YELLOW}tail -f $LOG_DIR/rfchain-hud.log${NC}"
    echo -e "  Check status:   ${YELLOW}curl http://localhost:$PORT/api/health${NC}"
    echo ""
    echo -e "${CYAN}First-time setup:${NC}"
    echo -e "  The first user to register will become the admin."
    echo ""
else
    echo -e "${RED}[ERROR] Server failed to start${NC}"
    echo -e "${YELLOW}Check logs at: $LOG_DIR/rfchain-hud.log${NC}"
    rm -f "$PID_FILE"
    exit 1
fi
