#!/bin/bash
#
# RFChain HUD - Pre-Start Dependency Check Script
# This script verifies all dependencies are available before starting the service
#
# Exit codes:
#   0 - All checks passed
#   1 - Critical dependency missing (service should not start)
#   2 - Non-critical dependency missing (service can start with degraded functionality)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="${RFCHAIN_INSTALL_DIR:-/opt/rfchain-hud}"
CONDA_ENV="${RFCHAIN_CONDA_ENV:-rfchain-hud}"
DB_TIMEOUT="${RFCHAIN_DB_TIMEOUT:-5}"
REQUIRED_NODE_VERSION="18"

# Track check results
CRITICAL_FAILED=0
WARNING_COUNT=0

log_info() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
    ((WARNING_COUNT++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    CRITICAL_FAILED=1
}

echo "═══════════════════════════════════════════════════════════════"
echo "  RFChain HUD - Pre-Start Dependency Check"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 1. Check if installation directory exists
echo "Checking installation directory..."
if [ -d "$INSTALL_DIR" ]; then
    log_info "Installation directory exists: $INSTALL_DIR"
else
    log_error "Installation directory not found: $INSTALL_DIR"
fi

# 2. Check for .env.local configuration file
echo "Checking configuration file..."
if [ -f "$INSTALL_DIR/.env.local" ]; then
    log_info "Configuration file found: $INSTALL_DIR/.env.local"
else
    log_error "Configuration file missing: $INSTALL_DIR/.env.local"
fi

# 3. Check Node.js availability and version
echo "Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -ge "$REQUIRED_NODE_VERSION" ]; then
        log_info "Node.js version: $(node -v)"
    else
        log_error "Node.js version too old: $(node -v) (requires v${REQUIRED_NODE_VERSION}+)"
    fi
else
    log_error "Node.js not found in PATH"
fi

# 4. Check if node_modules exists
echo "Checking dependencies..."
if [ -d "$INSTALL_DIR/node_modules" ]; then
    log_info "Node modules installed"
else
    log_error "Node modules not installed - run 'pnpm install' first"
fi

# 5. Check if build exists (for production)
echo "Checking build artifacts..."
if [ -d "$INSTALL_DIR/dist" ] || [ "$NODE_ENV" = "development" ]; then
    log_info "Build artifacts present (or running in development mode)"
else
    log_warn "Build artifacts missing - run 'pnpm build' for production"
fi

# 6. Check Conda environment (if using conda)
echo "Checking Python environment..."
if command -v conda &> /dev/null; then
    if conda env list | grep -q "$CONDA_ENV"; then
        log_info "Conda environment '$CONDA_ENV' exists"
        
        # Check if CuPy is installed in the environment
        if conda run -n "$CONDA_ENV" python -c "import cupy" 2>/dev/null; then
            log_info "CuPy (GPU acceleration) available"
        else
            log_warn "CuPy not available - GPU acceleration disabled"
        fi
    else
        log_warn "Conda environment '$CONDA_ENV' not found - using system Python"
    fi
else
    log_warn "Conda not found - using system Python"
fi

# 7. Check NVIDIA driver and GPU
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        log_info "GPU detected: $GPU_NAME (Driver: $GPU_DRIVER)"
        
        # Check GPU memory
        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_MEM_PERCENT=$((GPU_MEM_USED * 100 / GPU_MEM_TOTAL))
        
        if [ "$GPU_MEM_PERCENT" -gt 90 ]; then
            log_warn "GPU memory usage high: ${GPU_MEM_PERCENT}% (${GPU_MEM_USED}MB / ${GPU_MEM_TOTAL}MB)"
        else
            log_info "GPU memory available: ${GPU_MEM_PERCENT}% used"
        fi
    else
        log_warn "NVIDIA driver installed but GPU not accessible"
    fi
else
    log_warn "NVIDIA driver not installed - GPU acceleration unavailable"
fi

# 8. Check database connectivity (if DATABASE_URL is set)
echo "Checking database connectivity..."
if [ -f "$INSTALL_DIR/.env.local" ]; then
    # Source the env file to get DATABASE_URL
    export $(grep -v '^#' "$INSTALL_DIR/.env.local" | grep DATABASE_URL | xargs 2>/dev/null) || true
fi

if [ -n "$DATABASE_URL" ]; then
    # Extract host and port from DATABASE_URL
    # Format: mysql://user:pass@host:port/database
    DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:\/]*\).*/\1/p')
    DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    DB_PORT=${DB_PORT:-3306}
    
    if [ -n "$DB_HOST" ]; then
        # Try to connect using nc (netcat) with timeout
        if command -v nc &> /dev/null; then
            if nc -z -w "$DB_TIMEOUT" "$DB_HOST" "$DB_PORT" 2>/dev/null; then
                log_info "Database reachable: $DB_HOST:$DB_PORT"
            else
                log_error "Database unreachable: $DB_HOST:$DB_PORT (timeout: ${DB_TIMEOUT}s)"
            fi
        elif command -v timeout &> /dev/null; then
            # Fallback to bash /dev/tcp
            if timeout "$DB_TIMEOUT" bash -c "echo >/dev/tcp/$DB_HOST/$DB_PORT" 2>/dev/null; then
                log_info "Database reachable: $DB_HOST:$DB_PORT"
            else
                log_error "Database unreachable: $DB_HOST:$DB_PORT"
            fi
        else
            log_warn "Cannot verify database connectivity (nc/timeout not available)"
        fi
    else
        log_warn "Could not parse DATABASE_URL"
    fi
else
    log_warn "DATABASE_URL not set - database connectivity not verified"
fi

# 9. Check required ports are available
echo "Checking port availability..."
PORT="${PORT:-3007}"
if command -v lsof &> /dev/null; then
    if lsof -i ":$PORT" &> /dev/null; then
        PROCESS=$(lsof -i ":$PORT" -t 2>/dev/null | head -1)
        log_warn "Port $PORT is in use by PID $PROCESS"
    else
        log_info "Port $PORT is available"
    fi
elif command -v ss &> /dev/null; then
    if ss -tuln | grep -q ":$PORT "; then
        log_warn "Port $PORT is in use"
    else
        log_info "Port $PORT is available"
    fi
else
    log_warn "Cannot check port availability (lsof/ss not available)"
fi

# 10. Check disk space
echo "Checking disk space..."
DISK_USAGE=$(df "$INSTALL_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 95 ]; then
    log_error "Disk space critical: ${DISK_USAGE}% used"
elif [ "$DISK_USAGE" -gt 90 ]; then
    log_warn "Disk space low: ${DISK_USAGE}% used"
else
    log_info "Disk space OK: ${DISK_USAGE}% used"
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ $CRITICAL_FAILED -eq 1 ]; then
    echo -e "${RED}  CRITICAL CHECKS FAILED - Service cannot start${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
elif [ $WARNING_COUNT -gt 0 ]; then
    echo -e "${YELLOW}  CHECKS PASSED WITH $WARNING_COUNT WARNING(S)${NC}"
    echo "  Service will start with potentially degraded functionality"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    echo -e "${GREEN}  ALL CHECKS PASSED - Service ready to start${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi
