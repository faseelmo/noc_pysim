#!/bin/bash

sudo apt install -y python3.10-venv

python3.10 -m venv venv

echo source venv/bin/activate > activate_venv.sh

source activate_venv.sh
echo "Activating virtual environment..."

echo "Installing requirements..."
pip install -r requirements.txt 

snap install yq # Install yq for yaml parsing. Used in the training scripts for changing the config file.

echo "Requirements installed!"
echo "Setup complete!"
