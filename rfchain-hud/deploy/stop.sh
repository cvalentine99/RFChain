#!/bin/bash
# RFChain HUD - Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.rfchain-hud.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Stopping RFChain HUD...${NC}"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        kill "$PID"
        sleep 2
        
        # Force kill if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}[WARNING] Process still running, force killing...${NC}"
            kill -9 "$PID" 2>/dev/null
        fi
        
        rm -f "$PID_FILE"
        echo -e "${GREEN}✓ RFChain HUD stopped (PID: $PID)${NC}"
    else
        rm -f "$PID_FILE"
        echo -e "${YELLOW}[INFO] Server was not running (stale PID file removed)${NC}"
    fi
else
    echo -e "${YELLOW}[INFO] No PID file found. Server may not be running.${NC}"
    
    # Try to find and kill any running instance
    PIDS=$(pgrep -f "rfchain-hud" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo -e "${YELLOW}[INFO] Found running processes: $PIDS${NC}"
        echo -e "${YELLOW}Killing processes...${NC}"
        pkill -f "rfchain-hud" 2>/dev/null
        echo -e "${GREEN}✓ Processes terminated${NC}"
    fi
fi
