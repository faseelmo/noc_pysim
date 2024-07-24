#!/bin/bash

# Run the command to generate graph tasks
python3 -m data.create_graph_tasks --generate --gen_count 2000 --num_nodes 3

# Simulate latency on graphs
python3 -m data.simulate_latency_on_graphs --sim

# Create test data
python3 -m data.create_test_data