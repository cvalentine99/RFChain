#!/bin/bash
#
# RFChain HUD - Systemd Service Installation Script
# This script installs and configures the systemd service for automatic startup
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       RFChain HUD - Systemd Service Installation             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root (sudo)${NC}"
    exit 1
fi

# Configuration
INSTALL_DIR="/opt/rfchain-hud"
SERVICE_USER="rfchain"
SERVICE_GROUP="rfchain"
CONDA_PATH="/home/${SERVICE_USER}/miniconda3"

# Check if source directory exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_DIR/package.json" ]; then
    echo -e "${RED}Error: Cannot find project directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Creating service user...${NC}"
if id "$SERVICE_USER" &>/dev/null; then
    echo "  User '$SERVICE_USER' already exists"
else
    useradd -r -m -s /bin/bash "$SERVICE_USER"
    echo "  Created user '$SERVICE_USER'"
fi

echo -e "${YELLOW}Step 2: Creating installation directory...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/uploads"
mkdir -p "$INSTALL_DIR/analysis_output"
mkdir -p "$INSTALL_DIR/logs"

echo -e "${YELLOW}Step 3: Copying project files...${NC}"
rsync -av --exclude='node_modules' --exclude='.git' --exclude='uploads/*' --exclude='analysis_output/*' \
    "$PROJECT_DIR/" "$INSTALL_DIR/"

echo -e "${YELLOW}Step 4: Setting permissions...${NC}"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
chmod 750 "$INSTALL_DIR"
chmod 770 "$INSTALL_DIR/uploads"
chmod 770 "$INSTALL_DIR/analysis_output"
chmod 770 "$INSTALL_DIR/logs"

echo -e "${YELLOW}Step 5: Installing health check script...${NC}"
cp "$SCRIPT_DIR/check-deps.sh" "$INSTALL_DIR/deploy/"
chmod +x "$INSTALL_DIR/deploy/check-deps.sh"
chown "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR/deploy/check-deps.sh"

echo -e "${YELLOW}Step 6: Installing systemd service...${NC}"
cp "$SCRIPT_DIR/rfchain-hud.service" /etc/systemd/system/
chmod 644 /etc/systemd/system/rfchain-hud.service

# Update service file with actual paths if conda is in a different location
if [ -d "$CONDA_PATH" ]; then
    echo "  Conda found at: $CONDA_PATH"
else
    echo -e "${YELLOW}  Warning: Conda not found at default path${NC}"
    echo "  Please update /etc/systemd/system/rfchain-hud.service with correct conda path"
fi

echo -e "${YELLOW}Step 7: Reloading systemd...${NC}"
systemctl daemon-reload

echo -e "${YELLOW}Step 8: Checking for .env.local...${NC}"
if [ ! -f "$INSTALL_DIR/.env.local" ]; then
    if [ -f "$INSTALL_DIR/deploy/.env.local.template" ]; then
        cp "$INSTALL_DIR/deploy/.env.local.template" "$INSTALL_DIR/.env.local"
        chown "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR/.env.local"
        chmod 600 "$INSTALL_DIR/.env.local"
        echo -e "${YELLOW}  Created .env.local from template - PLEASE EDIT WITH YOUR SETTINGS${NC}"
    else
        echo -e "${RED}  Warning: No .env.local found - service may not start correctly${NC}"
    fi
else
    echo "  .env.local already exists"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete!                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Service Management Commands:${NC}"
echo ""
echo "  Start service:     sudo systemctl start rfchain-hud"
echo "  Stop service:      sudo systemctl stop rfchain-hud"
echo "  Restart service:   sudo systemctl restart rfchain-hud"
echo "  Check status:      sudo systemctl status rfchain-hud"
echo "  View logs:         sudo journalctl -u rfchain-hud -f"
echo "  Enable on boot:    sudo systemctl enable rfchain-hud"
echo "  Disable on boot:   sudo systemctl disable rfchain-hud"
echo ""
echo -e "${YELLOW}Before starting the service:${NC}"
echo "  1. Edit /opt/rfchain-hud/.env.local with your configuration"
echo "  2. Ensure conda environment 'rfchain-hud' is created for user '$SERVICE_USER'"
echo "  3. Run: sudo -u $SERVICE_USER bash -c 'cd $INSTALL_DIR && pnpm install && pnpm build'"
echo ""
echo -e "${CYAN}Access the application at: http://localhost:3007${NC}"
echo ""
