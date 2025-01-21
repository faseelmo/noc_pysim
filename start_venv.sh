#!/bin/bash

sudo apt install -y python3.10-venv

python3.10 -m venv venv

source venv/bin/activate 
echo "Activating virtual environment..."

echo "Installing requirements..."
pip install -r requirements.txt 

echo "Requirements installed!"
echo "Setup complete!"
