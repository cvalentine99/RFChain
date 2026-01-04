#!/bin/bash
# RFChain HUD - Production Start Script
# Runs on port 3007

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT=3007

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RFChain HUD - Signal Intelligence System            ║"
echo "║                   Production Deployment                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$PROJECT_DIR"

# Create necessary directories
echo -e "${CYAN}[1/5] Creating directories...${NC}"
mkdir -p uploads analysis_output logs data

# Check for .env.local
if [ ! -f ".env.local" ]; then
    echo -e "${RED}[ERROR] .env.local not found!${NC}"
    echo -e "${YELLOW}Copy the template and configure it:${NC}"
    echo "  cp deploy/.env.local.template .env.local"
    echo "  nano .env.local"
    echo ""
    echo -e "${YELLOW}IMPORTANT: This app requires MySQL. SQLite is NOT supported.${NC}"
    exit 1
fi

# Verify MySQL DATABASE_URL (not SQLite)
if grep -q "file:.*\.db" .env.local 2>/dev/null; then
    echo -e "${RED}[ERROR] SQLite database detected in .env.local${NC}"
    echo -e "${YELLOW}This application requires MySQL/MariaDB.${NC}"
    echo ""
    echo "Update DATABASE_URL in .env.local to use MySQL:"
    echo '  DATABASE_URL="mysql://user:password@localhost:3306/rfchain_hud"'
    exit 1
fi

# Check Node.js
echo -e "${CYAN}[2/5] Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR] Node.js not found. Install Node.js 18+.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js: $(node --version)${NC}"

# Install dependencies
echo -e "${CYAN}[3/5] Installing dependencies...${NC}"
if command -v pnpm &> /dev/null; then
    pnpm install --prod=false
else
    npm install
fi

# Build production bundle
echo -e "${CYAN}[4/5] Building production bundle...${NC}"
if command -v pnpm &> /dev/null; then
    pnpm build
else
    npm run build
fi

# Verify build output
if [ ! -f "dist/index.js" ]; then
    echo -e "${RED}[ERROR] Build failed - dist/index.js not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Build complete${NC}"

# Check if port is available
if lsof -i :$PORT > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] Port $PORT is already in use${NC}"
    echo "Stop the existing process or change PORT in .env.local"
    exit 1
fi

# Start production server
echo -e "${CYAN}[5/5] Starting production server...${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              RFChain HUD running on port $PORT               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Access: ${GREEN}http://localhost:$PORT${NC}"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo ""

# Export production environment and start
export NODE_ENV=production
export PORT=$PORT
exec node dist/index.js
