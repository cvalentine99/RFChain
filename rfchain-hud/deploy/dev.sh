#!/bin/bash
# RFChain HUD - Development Mode Script
# Runs in foreground with hot-reload on port 3007

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="rfchain-hud"
PORT=3007

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RFChain HUD - Development Mode (Port $PORT)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Create directories
mkdir -p "$PROJECT_DIR/uploads"
mkdir -p "$PROJECT_DIR/analysis_output"
mkdir -p "$PROJECT_DIR/audio_uploads"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/logs"

# Check for .env.local
if [ ! -f "$PROJECT_DIR/.env.local" ]; then
    echo -e "${YELLOW}[WARNING] .env.local not found. Creating from template...${NC}"
    if [ -f "$PROJECT_DIR/deploy/.env.local.template" ]; then
        cp "$PROJECT_DIR/deploy/.env.local.template" "$PROJECT_DIR/.env.local"
        echo -e "${YELLOW}Please edit .env.local with your configuration.${NC}"
    fi
fi

# Initialize conda
echo -e "${CYAN}Initializing conda...${NC}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo -e "${RED}[ERROR] Conda not found${NC}"
    exit 1
fi

# Create environment if needed
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo -e "${YELLOW}Creating conda environment...${NC}"
    conda env create -f "$PROJECT_DIR/deploy/environment.yml"
fi

conda activate "$CONDA_ENV_NAME"
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# Install dependencies
cd "$PROJECT_DIR"
if [ ! -d "node_modules" ]; then
    echo -e "${CYAN}Installing npm dependencies...${NC}"
    npm install
fi

# Export environment
export PORT=$PORT
export NODE_ENV=development
export DOTENV_CONFIG_PATH="$PROJECT_DIR/.env.local"

echo ""
echo -e "${GREEN}Starting development server on http://localhost:$PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Run in development mode
npm run dev
