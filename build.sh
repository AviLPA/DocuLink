#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies
apt-get update && apt-get install -y poppler-utils

# Create config directory if it doesn't exist
mkdir -p config

# Copy Firebase credentials to the right location
cp doculink-4db99-firebase-adminsdk-fbsvc-6dcd0f868f.json config/

# Upgrade pip and install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt 