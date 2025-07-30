#!/bin/bash

# 설치할 명령어
PIP_CMD="pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124"

# 무한 재시도 루프
while true; do
  echo "💡 Installing PyTorch stack..."
  
  if $PIP_CMD; then
    echo "✅ Installation successful!"
    break
  else
    echo "❌ Installation failed. Retrying in 3 seconds..."
    sleep 3
  fi
done
