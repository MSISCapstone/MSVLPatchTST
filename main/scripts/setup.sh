#!/bin/bash

# Setup script for PatchTST project
# Creates virtual environment and installs requirements

# Set paths relative to GIT_REPO_ROOT
GIT_REPO_ROOT=$(git rev-parse --show-toplevel)

echo "Setting up environment for PatchTST..."
echo "Project root: $GIT_REPO_ROOT"

# Check if running in Google Colab (skip venv if so)
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d "/content" ]; then
    echo "Running in Google Colab, skipping venv setup"
    IN_COLAB=1
else
    IN_COLAB=0
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
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
if [ "$IN_COLAB" -eq 1 ]; then
    echo "Colab detected - checking PyTorch CUDA status..."
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [ "$CUDA_AVAIL" != "True" ]; then
        echo "CUDA not available in Colab - reinstalling PyTorch with CUDA support..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "PyTorch with CUDA already available in Colab"
    fi
else
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    # torchaudio is optional and may not be available for all Python versions
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || echo "Note: torchaudio not available for this Python version (not required for PatchTST)"
fi

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "PyTorch installation verification failed!" >&2
    exit 1
}

# Install requirements (excluding torch to preserve CUDA version)
REQUIREMENTS_FILE="$GIT_REPO_ROOT/PatchTST_supervised/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing requirements from $REQUIREMENTS_FILE (excluding torch)..."
    grep -v "^torch" "$REQUIREMENTS_FILE" | pip install -r /dev/stdin
else
    echo "Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
if [ "$IN_COLAB" -eq 0 ]; then
    echo "To activate the virtual environment in future sessions, run:"
    echo "source $GIT_REPO_ROOT/.venv/bin/activate"
fi
