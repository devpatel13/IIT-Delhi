#!/bin/bash
# generate_candidates.sh <path_db_features> <path_query_features> <path_out_file>

source ./env.sh
$PYTHON_CMD search.py "$1" "$2" "$3"