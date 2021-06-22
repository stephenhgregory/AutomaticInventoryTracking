#!/usr/bin/env bash
# Used to install dependencies for this project

# Install ZBar library for reading barcodes
echo "Installing ZBar with the apt package manager..."
sudo apt-get install libzbar0

# Create new conda environment for this project
echo "Creating a new conda environment called \"AutomaticInventoryTracking\""
conda create -n AutomaticInventoryTracking pip python=3.8 -y
conda activate AutomaticInventoryTracking

# Install Tesseract
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

# Install packages into new conda environment
pip install pyzbar # barcode scanning
pip install opencv-python # computer vision
pip install opencv-contrib-python # add-ons to OpenCV (Super-resolution)
pip install pytesseract Pillow # image manipulation/processing
pip install pymongo # Python driver for MongoDB

# Install MongoDB
bash install_mongo_ubuntu.sh

printf "\nNew conda environment: \n\n     \"AutomaticInventoryTracking\"\n\nActivate this environment with \"conda activate AutomaticInventoryTracking\".\n"