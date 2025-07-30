#!/bin/bash

TMUX_CONF="$HOME/.tmux.conf"

echo "🛠 Writing tmux config to $TMUX_CONF"

cat > "$TMUX_CONF" <<EOF
# 마우스 스크롤 및 복사 모드 활성화
set -g mouse on

# vi 키 바인딩 사용 (hjkl로 탐색)
setw -g mode-keys vi

# 복사 모드에서 선택 시작: v, 선택 완료 및 복사: y
bind -T copy-mode-vi v send -X begin-selection
bind -T copy-mode-vi y send -X copy-selection-and-cancel
EOF

echo "🔁 Reloading tmux config..."

# 현재 tmux 세션에서만 적용됨 (스크립트가 tmux 내부에서 실행되는 경우에만 유효)
if tmux info &> /dev/null; then
  tmux source-file "$TMUX_CONF"
  echo "✅ tmux config applied!"
else
  echo "⚠️ tmux 세션이 감지되지 않았습니다. 다음 tmux 실행 시 적용됩니다."
fi
