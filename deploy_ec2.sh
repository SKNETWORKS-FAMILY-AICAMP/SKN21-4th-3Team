#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting EC2 Deployment Setup..."

# 1. Update System
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libpq-dev nginx git

# 2. Check Python Version
python3 --version

# 3. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Activate Venv and Install Dependencies
echo "📥 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Check .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found! Please create it based on .env.example"
    exit 1
fi

echo "✅ Environment Setup Complete!"
echo "---------------------------------------------------"
echo "To run the server (Test Mode):"
echo "   source venv/bin/activate"
echo "   python run.py"
echo ""
echo "To run with Gunicorn (Production):"
echo "   source venv/bin/activate"
echo "   gunicorn -w 4 -b 0.0.0.0:5000 run:app"
echo "---------------------------------------------------"
