#!/bin/bash

# Setup script for PatchTST project
# Creates virtual environment and installs requirements

# Set paths relative to GIT_REPO_ROOT
GIT_REPO_ROOT=$(git rev-parse --show-toplevel)

echo "Setting up virtual environment for PatchTST..."
echo "Project root: $GIT_REPO_ROOT"

# Setup virtual environment
if [ ! -d "$GIT_REPO_ROOT/.venv" ]; then
    echo "Creating virtual environment at $GIT_REPO_ROOT/.venv"
    python3 -m venv "$GIT_REPO_ROOT/.venv"
else
    echo "Virtual environment already exists at $GIT_REPO_ROOT/.venv"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$GIT_REPO_ROOT/.venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
REQUIREMENTS_FILE="$GIT_REPO_ROOT/PatchTST_supervised/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing requirements from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo "To activate the virtual environment in future sessions, run:"
echo "source $GIT_REPO_ROOT/.venv/bin/activate"