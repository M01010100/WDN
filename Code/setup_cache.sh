#!/bin/bash
# Quick setup script for dataset caching
# Usage: bash setup_cache.sh

set -e
echo ""
echo "WDN Dataset Cache Setup"
echo ""

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo " Python3 not found"
    exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# List available options
echo "Choose what to generate:"
echo ""
echo "1) Minimal  50K SPECK + 50K SIMON (~2 min)"
echo "2) Standard  100K SPECK + 100K SIMON (~3 min)"
echo "3) Balanced  200K SPECK + 200K SIMON (~5 min)"
echo "4) Full  All default datasets (~15 min)"
echo "5) Large  500K samples (~20 min)"
echo "6) Custom: Specify datasets"
echo "7) Just check what's cached"
echo ""

read -p "Enter choice (1-7): " choice

case $choice in
    1)
        echo "Generating minimal datasets (16GB VRAM)..."
        python3 cache_datasets.py --generate speck_7rounds_50k simon_8rounds_50k
        ;;
    2)
        echo "Generating standard datasets (16GB VRAM)..."
        python3 cache_datasets.py --generate speck_7rounds_100k simon_8rounds_100k
        ;;
    3)
        echo "Generating balanced datasets (32GB VRAM)..."
        python3 cache_datasets.py --generate \
            speck_7rounds_200k \
            simon_8rounds_200k \
            speck_6rounds_200k \
            speck_8rounds_200k
        ;;
    4)
        echo "Generating all datasets..."
        python3 cache_datasets.py --generate
        ;;
    5)
        echo "Generating large datasets (36GB+ VRAM)..."
        python3 cache_datasets.py --generate \
            speck_7rounds_500k \
            simon_8rounds_500k \
            speck_6rounds_200k \
            speck_8rounds_200k
        ;;
    6)
        echo "Available datasets:"
        python3 cache_datasets.py --list
        echo ""
        read -p "Enter dataset names (space-separated): " datasets
        python3 cache_datasets.py --generate $datasets
        ;;
    7)
        python3 cache_datasets.py --info
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "=========================================="
echo " Cache setup complete!"

