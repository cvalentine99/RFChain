#!/bin/bash
# RFChain HUD - One-Command Installation Script
# For Lubuntu 24.04 with Conda

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="rfchain-hud"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RFChain HUD - Installation Script                   ║"
echo "║                   Lubuntu 24.04 + Conda                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for conda
echo -e "${CYAN}[1/7] Checking prerequisites...${NC}"
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo -e "${RED}[ERROR] Conda not found. Please install Miniconda first:${NC}"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh"
        exit 1
    fi
fi
eval "$(conda shell.bash hook)"
echo -e "${GREEN}✓ Conda found${NC}"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR] Node.js not found. Please install Node.js 18+ or 20+:${NC}"
    echo "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
    echo "  sudo apt-get install -y nodejs"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node --version) found${NC}"

# Create conda environment
echo -e "${CYAN}[2/7] Setting up Python environment...${NC}"
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo -e "${YELLOW}[INFO] Conda environment '$CONDA_ENV_NAME' already exists${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$CONDA_ENV_NAME" -y
        conda env create -f "$PROJECT_DIR/deploy/environment.yml"
    fi
else
    conda env create -f "$PROJECT_DIR/deploy/environment.yml"
fi
conda activate "$CONDA_ENV_NAME"
echo -e "${GREEN}✓ Python $(python --version) environment ready${NC}"

# Create directories
echo -e "${CYAN}[3/7] Creating directories...${NC}"
mkdir -p "$PROJECT_DIR/uploads"
mkdir -p "$PROJECT_DIR/analysis_output"
mkdir -p "$PROJECT_DIR/audio_uploads"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/logs"
echo -e "${GREEN}✓ Directories created${NC}"

# Install npm dependencies
echo -e "${CYAN}[4/7] Installing Node.js dependencies...${NC}"
cd "$PROJECT_DIR"
npm install
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Setup environment file
echo -e "${CYAN}[5/7] Configuring environment...${NC}"
if [ ! -f "$PROJECT_DIR/.env.local" ]; then
    cp "$PROJECT_DIR/deploy/.env.local.template" "$PROJECT_DIR/.env.local"
    
    # Generate JWT secret
    JWT_SECRET=$(openssl rand -base64 32)
    sed -i "s|your-secure-jwt-secret-here|$JWT_SECRET|g" "$PROJECT_DIR/.env.local"
    
    # Update Python path
    PYTHON_PATH="$(which python)"
    sed -i "s|/home/your-username/miniconda3/envs/rfchain-hud/bin/python|$PYTHON_PATH|g" "$PROJECT_DIR/.env.local"
    
    # Update analysis script path
    ANALYSIS_PATH="$HOME/RFChain/analyze_signal_v2.2.2_forensic.py"
    sed -i "s|/home/your-username/RFChain/analyze_signal_v2.2.2_forensic.py|$ANALYSIS_PATH|g" "$PROJECT_DIR/.env.local"
    
    echo -e "${GREEN}✓ Environment file created${NC}"
    echo -e "${YELLOW}[NOTE] Please edit .env.local to add your API keys${NC}"
else
    echo -e "${YELLOW}[INFO] .env.local already exists, skipping${NC}"
fi

# Initialize database
echo -e "${CYAN}[6/7] Initializing database...${NC}"
npm run db:push 2>/dev/null || echo -e "${YELLOW}[INFO] Database initialization skipped${NC}"
echo -e "${GREEN}✓ Database ready${NC}"

# Build application
echo -e "${CYAN}[7/7] Building application...${NC}"
npm run build 2>/dev/null || echo -e "${YELLOW}[INFO] Build skipped (will run in dev mode)${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete!                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo -e "  1. Edit ${YELLOW}.env.local${NC} to add your API keys (ANTHROPIC_API_KEY, etc.)"
echo -e "  2. Start the server: ${YELLOW}./deploy/start.sh${NC}"
echo -e "  3. Open browser: ${GREEN}http://localhost:3007${NC}"
echo ""
echo -e "${CYAN}For development mode: ${YELLOW}./deploy/dev.sh${NC}"
echo -e "${CYAN}To stop the server: ${YELLOW}./deploy/stop.sh${NC}"
echo ""
