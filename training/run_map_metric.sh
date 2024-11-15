#!/bin/bash

# Root directory containing the experiments
ROOT_DIR="training/results/experiments"

# Python script and argument
PYTHON_SCRIPT="python3 -m training.map_metric"
ARG="--find"

# Iterate through all subdirectories under ROOT_DIR
for subdir in "$ROOT_DIR"/*/*; do
  # Check if it's a directory
  if [ -d "$subdir" ]; then
    echo "Running command for $subdir..."
    $PYTHON_SCRIPT --model_path "$subdir" $ARG
  fi
done