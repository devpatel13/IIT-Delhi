#!/bin/bash
# identify.sh
source ./env.sh

INPUT_DATASET=$1
OUTPUT_SUBGRAPHS=$2

# 1. Clean Duplicates
echo "Preprocessing..."
$PYTHON_CMD preprocess.py "$INPUT_DATASET" "temp_unique.data"

# 2. Mine Patterns
echo "Mining..."
$PYTHON_CMD miner.py "temp_unique.data" "$OUTPUT_SUBGRAPHS"

# 3. Cleanup
rm temp_unique.data gspan_temp_out.txt 2> /dev/null

