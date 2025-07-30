#!/bin/bash

set -e  # 오류 발생 시 스크립트 종료

echo "🔄 Updating APT package index..."
sudo apt update

echo "🐍 Installing Python 3.11 and development tools..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev

echo "📦 Creating virtual environment at ./venv311..."
python3.11 -m venv venv311

echo "✅ Virtual environment created."

echo
echo "🔔 To activate the virtual environment, run:"
echo "source venv311/bin/activate"

echo "🔔 Activate venv !"
source venv311/bin/activate
