#!/bin/bash

set -e  # 중간에 오류 나면 중단

echo "🚀 Installing build tools for flash_attn..."
pip install packaging wheel POT ninja

echo "⚡ Installing flash_attn==2.7.4.post1..."
pip install flash_attn==2.7.4.post1 --no-build-isolation

echo "🔍 Installing faiss and epiagent..."
pip install faiss-cpu
pip install epiagent==0.0.3 --no-deps

echo "🧹 Uninstalling problematic versions of core libraries..."
pip uninstall -y numpy pandas anndata scanpy

echo "📌 Re-installing fixed versions of numpy, pandas, anndata, scanpy..."
pip install numpy==1.24.4
pip install pandas==2.0.3
pip install anndata==0.11.4
pip install scanpy==1.11.1

echo "🤖 Installing transformers and ipykernel..."
pip install transformers==4.36.1
pip install ipykernel

echo "📊 Installing torchinfo..."
pip install torchinfo==1.8.0

EXCLUDE=("epiagent" "flash_attn" "faiss" \
            "transformers" "torchinfo" "packaging" "wheel" \
            "POT" "ninja" "numpy" "pandas" "anndata" "scanpy" \
            "torch" "torchvision" "torchaudio" \
            "ipykernel" "dbus-python")
echo "📦 Installing packages from requirements.txt, excluding: ${EXCLUDE[*]}"
filtered_reqs=$(mktemp)
grep -v -i -E "$(IFS='|'; echo "${EXCLUDE[*]}")" requirements.txt > "$filtered_reqs"
pip install -r "$filtered_reqs"
rm "$filtered_reqs"

echo "✅ All installations completed successfully!"
