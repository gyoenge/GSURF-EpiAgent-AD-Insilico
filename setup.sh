#!/bin/bash
set -e  # 에러 발생 시 즉시 종료

echo ">>> Setting up project environment..."

# 프로젝트 루트 기준 실행
cd "$(dirname "$0")"

# 실행 권한 부여 및 실행 함수
run_script() {
  local script_path=$1
  if [ -f "$script_path" ]; then
    chmod +x "$script_path"
    echo ">>> Running $script_path"
    "$script_path"
  else
    echo "!!! Script not found: $script_path"
    exit 1
  fi
}

# 1. (Optional) tmux 스크롤 설정
run_script "./setup/setup_tmux_scroll.sh"

# 2. Python venv 설정
run_script "./setup/setup_venv.sh"

# 3. Torch 설치
run_script "./setup/install_torch.sh"

# 4. Torch 이후 설치
run_script "./setup/install_aftertorch.sh"

# 5. apt 기반 도구 설치
run_script "./setup/install_apttool.sh"

echo ">>> All setup steps completed successfully!"
