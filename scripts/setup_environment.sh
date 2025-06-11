#!/bin/bash
# Setup script for BA-hydra project

set -e

echo "Setting up environment..."

# Check directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Run from project root."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install DataTrove
echo "Installing DataTrove..."
if [ -d "external/datatrove" ]; then
    pip install -e external/datatrove
else
    echo "Error: external/datatrove not found"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Test installation
echo "Testing installation..."
python -c "import numpy, torch, lightning; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}')"
python -c "from datatrove.pipeline.stats import DocStats; print('DataTrove: OK')"

# Test additional dependencies for stats modules
echo "Testing DataTrove stats dependencies..."
python -c "import spacy, fasttext, kenlm, tokenizers, tldextract; print('All dependencies: OK')"

echo "Setup complete!"

echo "Setup complete!"
echo "Test with: python src/dataprep/pipelines/run_stats.py stats.limit_documents=10"
