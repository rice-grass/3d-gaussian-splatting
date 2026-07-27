# v3 — 영상 → 3D Gaussian Splatting End-to-End 파이프라인

영상 **한 개**를 넣으면 그 공간을 3DGS로 재구성하고, 카메라 궤도 렌더링 영상까지
자동으로 뽑아내는 end-to-end 구성입니다. 백엔드는 **gsplat**(nerfstudio-project).

## 환경 (RTX 5080 / Blackwell)

- GPU: RTX 5080 16GB (sm_120) — CUDA arch `12.0`
- CUDA Toolkit 12.8 (`/usr/local/cuda-12.8`, nvcc)
- 실행 파이썬: **v3 전용 uv venv** → `v3/v3` (Python 3.11, torch 2.11+cu128)
- COLMAP: `/usr/local/bin/colmap` (3.9.1)
- 프레임 추출은 **opencv**로 처리 → ffmpeg 불필요

> gsplat CUDA 커널은 `TORCH_CUDA_ARCH_LIST=12.0` 환경변수로 sm_120에 맞춰 빌드되며,
> 스크립트에 이미 설정돼 있습니다. 첫 실행 시 커널 JIT 컴파일로 수 분 소요될 수 있습니다.

## 폴더 구조

```
v3/
├── v3/                    # ★ 전용 uv venv (python 3.11, torch 2.11+cu128, gsplat)
├── gsplat/                # 클론한 오픈소스 (nerfstudio-project/gsplat, 서브모듈 포함)
├── data/
│   ├── videos/            # 입력 영상 (*.mp4 등)을 여기에
│   └── processed/<name>/  # 자동 생성: images/, sparse/0/(COLMAP), database.db
├── outputs/<name>/        # 자동 생성: ckpts/, ply/, renders/(궤도영상), stats/, tb/
├── scripts/
│   ├── run_e2e.sh         # ★ 원커맨드 전체 파이프라인
│   ├── extract_frames.py  # 영상 → 프레임 (opencv)
│   ├── run_colmap.sh      # 프레임 → COLMAP SfM
│   └── view.sh            # viser 실시간 웹 뷰어
└── README.md
```

## 사용법

```bash
# 1) 영상을 data/videos/ 에 넣기 (예: room.mp4)
# 2) 원커맨드 실행:  run_e2e.sh <영상> <이름> [fps] [max_frames] [iters]
cd ~/ssafy/3d-gaussian-splatting/v3
bash scripts/run_e2e.sh data/videos/room.mp4 room 3 250 30000
```

파이프라인 단계:
1. **프레임 추출** — `fps`장/초, 최대 `max_frames`장 (opencv)
2. **COLMAP SfM** — 카메라 포즈 + 희소 포인트클라우드 (`sparse/0`)
3. **gsplat 학습** — `simple_trainer.py default`, 종료 시 `ply/`·`renders/` 자동 생성
4. **궤도 렌더** — 최종 체크포인트로 spiral 궤도 영상 재확인

결과:
- 학습 모델: `outputs/<name>/ply/point_cloud_*.ply`
- 궤도 렌더 영상: `outputs/<name>/renders/traj_*.mp4`
- 실시간 확인: `bash scripts/view.sh <name>` → 브라우저 `http://localhost:8080`

## 파라미터 가이드

| 인자 | 기본 | 설명 |
|---|---|---|
| `fps` | 3 | 초당 추출 프레임. 카메라 이동이 빠르면 4~5로 상향 |
| `max_frames` | 250 | 상한. 많을수록 품질↑·SfM 시간↑ (RTX 5080 16GB는 200~400 적정) |
| `iters` | 30000 | 학습 스텝. 빠른 확인은 7000, 고품질은 30000 |

## 트러블슈팅

- **COLMAP이 이미지를 거의 등록 못 함**: `fps`를 높여 프레임 겹침을 늘리세요. 영상이
  너무 흔들리거나 텍스처 없는 벽면이면 SfM이 실패합니다.
- **WSL에서 SiftGPU 오류**: `scripts/run_colmap.sh`의 `GPU=1`을 `GPU=0`(CPU)으로 변경.
- **VRAM 부족(16GB)**: `max_frames`를 줄이거나 `--data-factor 2`로 이미지 다운스케일.
- **`no kernel image`류 CUDA 오류**: `TORCH_CUDA_ARCH_LIST=12.0`이 설정됐는지 확인
  (스크립트에 포함). 커널 캐시 초기화: `rm -rf ~/.cache/torch_extensions/*/gsplat_cuda`.
- **ppisp 미설치 경고**: 무시해도 됨. 기본 파이프라인(post_processing=None)과 무관.

## 이 구성이 하는 일 (기존 v1/v2 대비)

- v1(INRIA 원조), v2(raw gsplat)는 이미지 폴더 기반의 부분 구성이었음.
- **v3는 "영상 1개 → 렌더 영상"까지 한 번에** 도는 end-to-end 스크립트를 갖춤.
- 프레임 추출(opencv) + COLMAP + gsplat 학습 + 궤도 렌더를 하나로 오케스트레이션.
