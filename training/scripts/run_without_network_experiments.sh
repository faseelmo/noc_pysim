#!/bin/bash

# File paths
YAML_FILE="training/config/params_without_network.yaml"  # Path to your YAML file
TRAIN_SCRIPT="python3 -m training.train_without_network"  # Command to run training
UPDATE_SCRIPT="./training/scripts/update_yaml.sh"  # Path to your update script

# Parameters to iterate over
CONV_TYPES=("graphconv") # "graphconv" "gcn" "gin" "sage" "gat"
NUM_LAYERS=(5)
HIDDEN_CHANNELS=(64)
LOSS_FUNCTIONS=("mae") # "mse" "mae" "huber"
AGGR=("add") # "mean" "max" "min" "sum"
MAX_EPOCHS=200

IS_HETERO=True
HAS_TASK_DEPEND=False
HAS_SCHEDULER=False 

$UPDATE_SCRIPT "$YAML_FILE" EPOCHS "$MAX_EPOCHS"
$UPDATE_SCRIPT "$YAML_FILE" IS_HETERO "$IS_HETERO"
$UPDATE_SCRIPT "$YAML_FILE" HAS_TASK_DEPEND "$HAS_TASK_DEPEND"
$UPDATE_SCRIPT "$YAML_FILE" HAS_SCHEDULER "$HAS_SCHEDULER"

# Iterate over all parameter combinations
for CONV in "${CONV_TYPES[@]}"; do
    for LAYER in "${NUM_LAYERS[@]}"; do
        for CHANNEL in "${HIDDEN_CHANNELS[@]}"; do
            for LOSS in "${LOSS_FUNCTIONS[@]}"; do
                for AGGR in "${AGGR[@]}"; do

                    # Update parameters in the YAML file using update_yaml.sh
                    $UPDATE_SCRIPT "$YAML_FILE" CONV_TYPE "$CONV"
                    $UPDATE_SCRIPT "$YAML_FILE" NUM_MPN_LAYERS "$LAYER"
                    $UPDATE_SCRIPT "$YAML_FILE" HIDDEN_CHANNELS "$CHANNEL"
                    $UPDATE_SCRIPT "$YAML_FILE" AGGR "$AGGR"
                    $UPDATE_SCRIPT "$YAML_FILE" LOSS_FN "$LOSS"

                    # Generate a unique directory for this run
                    RUN_DIR="${CONV}_L${LAYER}_C${CHANNEL}_A${AGGR}_${LOSS}"

                    if [ "$IS_HETERO" = True ]; then
                        RUN_DIR="${RUN_DIR}_HETERO"
                    fi

                    if [ "$HAS_TASK_DEPEND" = True ]; then
                        RUN_DIR="${RUN_DIR}_TASKDEPEND"
                    fi

                    if [ "$HAS_SCHEDULER" = True ]; then
                        RUN_DIR="${RUN_DIR}_SCHEDULER"
                    fi

                    # Run training
                    echo "Running training for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, AGGR=$AGGR, LOSS=$LOSS, HETERO=$IS_HETERO, TASK_DEPEND=$HAS_TASK_DEPEND, SCHEDULER=$HAS_SCHEDULER"
                    $TRAIN_SCRIPT "$RUN_DIR"

                    # Check if training succeeded
                    if [ $? -ne 0 ]; then
                        echo "Training failed for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, AGGR=$AGGR, LOSS=$LOSS, HETERO=$IS_HETERO, TASK_DEPEND=$HAS_TASK_DEPEND, SCHEDULER=$HAS_SCHEDULER"
                    else
                        echo "Training completed for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, AGGR=$AGGR, LOSS=$LOSS, HETERO=$IS_HETERO, TASK_DEPEND=$HAS_TASK_DEPEND, SCHEDULER=$HAS_SCHEDULER"
                    fi

                done
            done
        done
    done
done

echo "All experiments completed!"
