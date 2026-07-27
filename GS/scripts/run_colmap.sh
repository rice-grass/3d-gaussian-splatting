#!/usr/bin/env bash
# 프레임 이미지 폴더 -> COLMAP SfM -> gsplat이 읽는 포맷(sparse/0)으로 정리
# 사용법: run_colmap.sh <scene_dir>
#   scene_dir/images/*.jpg 가 입력, 결과는 scene_dir/sparse/0 에 생성
set -euo pipefail

SCENE="${1:?scene_dir 를 지정하세요 (예: data/processed/myscene)}"
SCENE="$(realpath "$SCENE")"
IMAGES="$SCENE/images"
DB="$SCENE/database.db"
SPARSE="$SCENE/sparse"

[ -d "$IMAGES" ] || { echo "[colmap] $IMAGES 없음"; exit 1; }
echo "[colmap] 이미지 $(ls "$IMAGES" | wc -l)장 처리: $SCENE"

# GPU 사용 (RTX 5080). WSL에서 SiftGPU 문제 시 --SiftExtraction.use_gpu 0 으로 변경.
GPU=1

rm -f "$DB"; mkdir -p "$SPARSE"

colmap feature_extractor \
  --database_path "$DB" --image_path "$IMAGES" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu $GPU

colmap exhaustive_matcher \
  --database_path "$DB" --SiftMatching.use_gpu $GPU

colmap mapper \
  --database_path "$DB" --image_path "$IMAGES" --output_path "$SPARSE"

# gsplat/nerfstudio 파서는 sparse/0 를 기대. mapper가 0,1,... 로 만들면 0 사용.
echo "[colmap] 완료. 재구성 모델:"; ls "$SPARSE"
echo "[colmap] 등록된 이미지 수 확인:"
colmap model_analyzer --path "$SPARSE/0" 2>/dev/null | grep -iE 'Registered|points' || true
