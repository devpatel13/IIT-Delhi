#!/bin/bash
# convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

source ./env.sh
$PYTHON_CMD vectorizer.py "$1" "$2" "$3"