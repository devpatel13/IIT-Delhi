#!/bin/bash
# env.sh

# 1. Install Dependencies (Requirement)
echo "Installing dependencies..."
pip install networkx numpy gspan-mining scikit-learn pandas

# 2. Detect Python Command (Fix for Windows Git Bash)
# This exports a variable $PYTHON_CMD that other scripts will use.
if command -v python &> /dev/null; then
    export PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    export PYTHON_CMD="python3"
else
    # Fallback for some Windows setups
    export PYTHON_CMD="python"
fi

echo "Environment setup complete. Using: $PYTHON_CMD"