#!/usr/bin/env bash
# =============================================================================
# 영상 1개 -> 3D Gaussian Splatting 학습 -> 카메라 궤도 렌더링 영상까지
# End-to-End 원커맨드 파이프라인 (gsplat 백엔드, RTX 5080 / CUDA 12.8)
#
# 사용법:
#   scripts/run_e2e.sh <video.mp4> <scene_name> [fps] [max_frames] [iters]
# 예:
#   scripts/run_e2e.sh data/videos/room.mp4 room 3 250 30000
#
# 단계: (1) 프레임 추출  (2) COLMAP SfM  (3) gsplat 학습  (4) 렌더 영상 출력
# =============================================================================
set -euo pipefail

# --- 경로 설정 ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/v3"                      # v3 전용 uv venv (python 3.11, torch 2.11+cu128)
PY="$VENV/bin/python"
GSPLAT="$ROOT/gsplat"

# --- 인자 ---
VIDEO="${1:?영상 경로를 지정하세요}"
NAME="${2:?scene 이름을 지정하세요}"
FPS="${3:-3}"
MAXF="${4:-250}"
ITERS="${5:-30000}"

SCENE="$ROOT/data/processed/$NAME"
RESULT="$ROOT/outputs/$NAME"
mkdir -p "$SCENE/images" "$RESULT"

# RTX 5080 (Blackwell sm_120) CUDA 컴파일 타깃
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS="${MAX_JOBS:-16}"

echo "======================================================================"
echo " E2E:  $VIDEO  ->  scene='$NAME'  (fps=$FPS max=$MAXF iters=$ITERS)"
echo "======================================================================"

# (1) 프레임 추출 (opencv, ffmpeg 불필요)
echo "[1/4] 프레임 추출"
"$PY" "$ROOT/scripts/extract_frames.py" "$VIDEO" "$SCENE/images" --fps "$FPS" --max "$MAXF"

# (2) COLMAP: 카메라 포즈 + 희소 포인트클라우드
echo "[2/4] COLMAP SfM"
bash "$ROOT/scripts/run_colmap.sh" "$SCENE"

# (3) gsplat 학습 (Default 3DGS 전략).
#     학습 종료 시 자동으로 renders/ 에 궤도 렌더 영상 + ply/ 에 결과를 저장.
echo "[3/4] gsplat 학습 (+ 학습 종료 시 궤도 렌더 자동 생성)"
cd "$GSPLAT/examples"
"$PY" simple_trainer.py default \
  --data-dir "$SCENE" \
  --data-factor 1 \
  --result-dir "$RESULT" \
  --max-steps "$ITERS" \
  --render-traj-path spiral \
  --disable-viewer \
  --save-ply

# (4) (선택) 최종 체크포인트로 궤도 렌더 재생성 — --ckpt 지정 시 학습 없이 렌더만 수행
echo "[4/4] 최종 체크포인트 궤도 렌더 재확인"
CKPT="$(ls -t "$RESULT"/ckpts/*.pt 2>/dev/null | head -1 || true)"
if [ -n "$CKPT" ]; then
  "$PY" simple_trainer.py default \
    --data-dir "$SCENE" --data-factor 1 \
    --result-dir "$RESULT" \
    --render-traj-path spiral \
    --disable-viewer \
    --ckpt "$CKPT"
fi

echo "======================================================================"
echo " 완료!"
echo "   - 학습 결과(.ply):   $RESULT/ply/point_cloud_*.ply"
echo "   - 궤도 렌더 영상:    $RESULT/renders/  (traj_*.mp4)"
echo "   - 평가 지표:         $RESULT/stats/ , 텐서보드: $RESULT/tb/"
echo "   - 실시간 뷰어:       scripts/view.sh $NAME"
echo "======================================================================"
