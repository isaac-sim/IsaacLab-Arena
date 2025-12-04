# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Setup documentation environment for isaaclab_arena
#
# USAGE: source ./docs/setup.sh
# This will set up the environment AND activate it in your current shell

# Detect if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ Error: This script must be sourced, not executed!"
    echo ""
    echo "Please run:"
    echo "  source ./docs/setup.sh"
    echo ""
    echo "Or from the docs directory:"
    echo "  source setup.sh"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Isaac Lab Arena - Documentation Setup"
echo "=========================================="
echo ""

# Check if we're in docker (optional warning)
if [ ! -f "/.dockerenv" ]; then
    echo "⚠️  Warning: It appears you're not running inside the Docker container."
    echo "   Consider running: ./docker/run_docker.sh"
    echo ""
fi

# Step 1: Install Python 3.11 and venv
echo "Step 1/4: Checking Python 3.11 and venv..."
if ! command -v python3.11 &> /dev/null; then
    echo "  Installing python3.11 and python3.11-venv..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv
    echo "  ✓ Python 3.11 installed"
else
    echo "  ✓ Python 3.11 already installed"
fi
echo ""

# Step 2: Create virtual environment if it doesn't exist
echo "Step 2/4: Setting up virtual environment..."
if [ ! -d "venv_docs" ]; then
    echo "  Creating new virtual environment..."
    python3.11 -m venv venv_docs
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate the virtual environment
echo "Step 3/4: Activating virtual environment..."
source venv_docs/bin/activate
echo "  ✓ Virtual environment activated"
echo ""

# Step 4: Install/update dependencies
echo "Step 4/4: Installing documentation dependencies..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
echo "  ✓ Dependencies installed"
echo ""

echo "=========================================="
echo "✓ Setup complete and environment active!"
echo "=========================================="
echo ""
echo "Your shell now has the (venv_docs) environment active."
echo ""
echo "Build the docs with:"
echo "  make html"
echo ""
echo "To deactivate the environment later:"
echo "  deactivate"
echo ""
echo "To reactivate in the future:"
echo "  cd docs && source venv_docs/bin/activate"
echo ""
