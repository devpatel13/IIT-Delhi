#!/bin/bash
# identify.sh <path_graph_dataset> <path_discriminative_subgraphs>

# 1. Load Environment (Fixes Python path)
source ./env.sh

INPUT_DATASET=$1
OUTPUT_SUBGRAPHS=$2

# 2. Clean Data (preprocess.py)
$PYTHON_CMD preprocess.py "$INPUT_DATASET" "temp_unique.data"

# 3. Mine Patterns (miner.py)
$PYTHON_CMD miner.py "temp_unique.data" "$OUTPUT_SUBGRAPHS"

# 4. Cleanup
rm temp_unique.data gspan_temp_out.txt 2> /dev/null