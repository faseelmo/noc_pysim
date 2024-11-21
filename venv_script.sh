#!/bin/bash

sudo apt install -y python3.10-venv

python3.10 -m venv venv

TARGET_DIR="data/training_data/simulator"

# Check if the directory already exists
if [ -d "$TARGET_DIR" ]; then
  echo "Training data folder '$TARGET_DIR' already exists. Skipping unzip."
else
  cd data/training_data
  echo "Unzipping training data..."
  unzip -q simulator.zip 
  echo "Training data unzipped in '$TARGET_DIR'"
  cd ../../ 
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8 # for torch.use_deterministic_algorithms(True)

echo source venv/bin/activate > activate_venv.sh

source activate_venv.sh
echo "Activating virtual environment..."

echo "Installing requirements..."
pip install -r requirements.txt 

echo "Requirements installed!"
echo "Setup complete!"