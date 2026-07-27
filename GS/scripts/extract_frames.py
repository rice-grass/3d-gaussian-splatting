#!/usr/bin/env python3
"""영상 -> 프레임 이미지 추출 (opencv 사용, ffmpeg 불필요).

사용법:
    python extract_frames.py <video> <out_dir> [--fps 2] [--max 300]

- --fps: 초당 추출 프레임 수 (기본 2). COLMAP 매칭에는 겹침이 충분한
         2~4fps 정도가 적당. 너무 많으면 SfM이 느려지고, 너무 적으면 매칭 실패.
- --max: 최대 프레임 수 상한 (기본 300). 초과 시 균등 간격으로 샘플링.
"""
import argparse
import os
import sys

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=300)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[extract] 영상을 열 수 없습니다: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / args.fps))

    # 예상 추출 개수가 --max 초과 시 step 을 키워 균등 샘플링
    est = total // step if total else args.max
    if est > args.max and total:
        step = max(step, total // args.max)

    os.makedirs(args.out_dir, exist_ok=True)
    idx = saved = 0
    print(f"[extract] src_fps={src_fps:.1f} total={total} step={step} -> 목표 ~{min(est, args.max)}장")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            path = os.path.join(args.out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if saved >= args.max:
                break
        idx += 1
    cap.release()
    print(f"[extract] 완료: {saved}장 -> {args.out_dir}")
    if saved < 20:
        print("[extract][경고] 프레임이 20장 미만이면 COLMAP 재구성이 실패할 수 있습니다. --fps 를 높이세요.")


if __name__ == "__main__":
    main()
