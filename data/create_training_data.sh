#!/bin/bash

# Default values
GEN_COUNT=1000
NUM_NODES=2

# Override defaults with arguments if provided
if [ ! -z "$1" ]; then
  GEN_COUNT=$1
fi

if [ ! -z "$2" ]; then
  NUM_NODES=$2
fi

# Run the command to generate graph tasks
python3 -m data.create_graph_tasks --generate --gen_count $GEN_COUNT --num_nodes $NUM_NODES

# Simulate latency on graphs
python3 -m data.simulate_latency_on_graphs --sim

# Create test data
python3 -m data.create_test_data