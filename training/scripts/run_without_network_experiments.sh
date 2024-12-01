#!/bin/bash

# File paths
YAML_FILE="training/config/params_without_network.yaml"  # Path to your YAML file
TRAIN_SCRIPT="python3 -m training.train_without_network"  # Command to run training
UPDATE_SCRIPT="./training/scripts/update_yaml.sh"  # Path to your update script

# Parameters to iterate over
CONV_TYPES=("graphconv" "gcn" "gin" "sage" "gat")
NUM_LAYERS=(3)
HIDDEN_CHANNELS=(64)
LOSS_FUNCTIONS=("mse")
MAX_EPOCHS=200

# Set maximum epochs in the YAML file
$UPDATE_SCRIPT "$YAML_FILE" EPOCHS "$MAX_EPOCHS"

# Iterate over all parameter combinations
for CONV in "${CONV_TYPES[@]}"; do
    for LAYER in "${NUM_LAYERS[@]}"; do
        for CHANNEL in "${HIDDEN_CHANNELS[@]}"; do
            for LOSS in "${LOSS_FUNCTIONS[@]}"; do
                # Update parameters in the YAML file using update_yaml.sh
                $UPDATE_SCRIPT "$YAML_FILE" CONV_TYPE "$CONV"
                $UPDATE_SCRIPT "$YAML_FILE" NUM_MPN_LAYERS "$LAYER"
                $UPDATE_SCRIPT "$YAML_FILE" HIDDEN_CHANNELS "$CHANNEL"
                $UPDATE_SCRIPT "$YAML_FILE" LOSS_FN "$LOSS"

                # Generate a unique directory for this run
                RUN_DIR="${CONV}_L${LAYER}_C${CHANNEL}_${LOSS}"
                mkdir -p "$RUN_DIR"

                # Run training
                echo "Running training for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, LOSS=$LOSS"
                $TRAIN_SCRIPT "$RUN_DIR"

                # Check if training succeeded
                if [ $? -ne 0 ]; then
                    echo "Training failed for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, LOSS=$LOSS"
                else
                    echo "Training completed for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, LOSS=$LOSS"
                fi
            done
        done
    done
done

echo "All experiments completed!"
