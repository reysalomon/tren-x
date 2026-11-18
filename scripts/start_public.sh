#!/usr/bin/env bash
# Запуск TREN-X одной командой: сервер + публичная ссылка (Cloudflare tunnel) для телефона.
# Остановить — Ctrl+C или закрыть окно.
cd "$(dirname "$0")/.." || exit 1
CF=$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")
LOG=/tmp/trenx-server.log

if curl -s -o /dev/null http://localhost:8000/api/info; then
  echo "Сервер уже работает."
else
  echo "Запускаю сервер..."
  .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 >"$LOG" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' EXIT
  for _ in $(seq 60); do
    curl -s -o /dev/null http://localhost:8000/api/info && break
    sleep 1
  done
  curl -s -o /dev/null http://localhost:8000/api/info || { echo "Сервер не запустился, смотри $LOG"; exit 1; }
fi
echo "Сервер работает. Открываю туннель (10-20 секунд)..."
# http2, а не QUIC: QUIC-туннель обрывался через ~4 минуты, и живая камера отключалась
"$CF" tunnel --protocol http2 --url http://localhost:8000 2>&1 | while IFS= read -r line; do
  url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' <<<"$line")
  if [ -n "$url" ]; then
    echo
    echo "=================================================="
    echo "  ССЫЛКА ДЛЯ ТЕЛЕФОНА:  $url"
    echo "=================================================="
    echo "  Не закрывай это окно, пока идёт показ."
    echo "  Если ссылка не открывается сразу — подожди 30 секунд."
    echo
  elif grep -qiE 'error|failed' <<<"$line"; then
    echo "$line"
  fi
done
