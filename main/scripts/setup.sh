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

# Activate virtual environment (POSIX and Windows venv compatibility)
echo "Activating virtual environment..."
if [ -f "$GIT_REPO_ROOT/.venv/bin/activate" ]; then
    echo "Activating virtualenv (bin/activate)"
    source "$GIT_REPO_ROOT/.venv/bin/activate"
elif [ -f "$GIT_REPO_ROOT/.venv/Scripts/activate" ]; then
    echo "Activating virtualenv (Scripts/activate)"
    source "$GIT_REPO_ROOT/.venv/Scripts/activate"
else
    echo "No activation script found in $GIT_REPO_ROOT/.venv; attempting to create venv with python3"
    python3 -m venv "$GIT_REPO_ROOT/.venv"
    if [ -f "$GIT_REPO_ROOT/.venv/bin/activate" ]; then
        source "$GIT_REPO_ROOT/.venv/bin/activate"
    elif [ -f "$GIT_REPO_ROOT/.venv/Scripts/activate" ]; then
        source "$GIT_REPO_ROOT/.venv/Scripts/activate"
    else
        echo "Failed to create or locate activation script in $GIT_REPO_ROOT/.venv" >&2
        exit 1
    fi
fi

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