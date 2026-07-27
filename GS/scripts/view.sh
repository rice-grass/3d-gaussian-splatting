#!/usr/bin/env bash
# 학습된 체크포인트를 viser 실시간 웹 뷰어로 띄움 (브라우저에서 확인)
# 사용법: view.sh <scene_name> [port]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/v3/bin/python"
NAME="${1:?scene 이름을 지정하세요}"
PORT="${2:-8080}"
RESULT="$ROOT/outputs/$NAME"
export TORCH_CUDA_ARCH_LIST="12.0"

CKPT="$(ls -t "$RESULT"/ckpts/*.pt 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "체크포인트 없음: $RESULT/ckpts/"; exit 1; }

echo "viser 뷰어 실행 -> 브라우저에서 http://localhost:$PORT"
cd "$ROOT/gsplat/examples"
"$PY" simple_trainer.py default \
  --data-dir "$ROOT/data/processed/$NAME" --data-factor 1 \
  --result-dir "$RESULT" \
  --ckpt "$CKPT" \
  --port "$PORT"
