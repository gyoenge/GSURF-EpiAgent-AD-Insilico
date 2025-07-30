#!/bin/bash

check_and_install() {
  local pkg="$1"
  echo "🔍 Checking if $pkg is already installed..."

  if command -v "$pkg" &> /dev/null; then
    echo "✅ $pkg is already installed: $(which $pkg)"
  else
    echo "📦 Installing $pkg..."
    sudo apt update
    sudo apt install -y "$pkg"
    echo "✅ $pkg installation complete."
  fi
}

# wget 설치 확인 및 설치
check_and_install wget

# bedtools 설치 확인 및 설치
check_and_install bedtools


