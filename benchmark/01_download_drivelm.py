#!/usr/bin/env python3
"""
DriveLM 이미지 100장 다운로드 스크립트

HuggingFace OpenDriveLab/DriveLM 에서 val split zip을 받아 100장 샘플링.
저장 위치: benchmark/images/  (jpg 파일들)
           benchmark/drivelm_samples.json  (이미지 메타)

Usage:
    HF_TOKEN=hf_xxx python 01_download_drivelm.py
    python 01_download_drivelm.py --local-dir /path/to/images   # 로컬 폴더 사용
"""

import os
import json
import random
import argparse
import shutil
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError
except ImportError:
    print("huggingface-hub 없음. 설치: pip install huggingface-hub")
    raise

REPO_ID = "OpenDriveLab/DriveLM"
REPO_TYPE = "dataset"
NUM_SAMPLES = 100
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_JSON = BASE_DIR / "drivelm_samples.json"
EXTRACT_DIR = BASE_DIR / "_drivelm_extracted"


def download_and_extract_zip(token: str | None) -> Path:
    """drivelm_nus_imgs_val.zip 다운로드 & 압축 해제"""
    zip_filename = "drivelm_nus_imgs_val.zip"

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.jpg")):
        print(f"이미 압축 해제됨: {EXTRACT_DIR}")
        return EXTRACT_DIR

    print(f"{zip_filename} 다운로드 중... (크기에 따라 수분 소요)")
    local_zip = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=zip_filename,
        token=token,
    )
    print(f"다운로드 완료: {local_zip}")

    print(f"압축 해제 중 → {EXTRACT_DIR}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("압축 해제 완료")
    return EXTRACT_DIR


def load_qa_json(token: str | None) -> dict:
    """DriveLM val QA JSON 다운로드"""
    print("QA JSON 다운로드 중...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="v1_1_val_nus_q_only.json",
        token=token,
    )
    with open(local_path, "r") as f:
        return json.load(f)


def build_samples(qa_data: dict, extract_dir: Path) -> list[dict]:
    """QA JSON + 압축 해제된 이미지 매핑"""
    # 압축 해제된 CAM_FRONT 이미지 파일명 → 경로 인덱스
    cam_front_imgs = {p.name: p for p in extract_dir.rglob("*.jpg") if "CAM_FRONT" in str(p) and "LEFT" not in str(p) and "RIGHT" not in str(p)}
    print(f"압축 해제된 CAM_FRONT 이미지: {len(cam_front_imgs)}장")

    samples = []
    for scene_token, scene_data in qa_data.items():
        for frame_token, frame_data in scene_data.get("key_frames", {}).items():
            img_paths = frame_data.get("image_paths", {})
            cam_front_path = img_paths.get("CAM_FRONT", "")
            filename = Path(cam_front_path).name

            if filename not in cam_front_imgs:
                continue

            qa_preview = ""
            for cat in ("planning", "perception"):
                items = frame_data.get("QA", {}).get(cat, [])
                if items:
                    q = items[0].get("Q", "")[:80]
                    qa_preview = f"[{cat}] {q}"
                    break

            samples.append({
                "image_id": frame_token,
                "scene_token": scene_token,
                "frame_token": frame_token,
                "src_image_path": str(cam_front_imgs[filename]),
                "qa_preview": qa_preview,
            })

    return samples


def copy_sampled_images(samples: list[dict]) -> list[dict]:
    """선택된 이미지를 benchmark/images/ 로 복사"""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for i, s in enumerate(samples):
        src = Path(s["src_image_path"])
        dst = IMAGE_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        print(f"  [{i+1}/{len(samples)}] {src.name}")
        result.append({
            "image_id": s["image_id"],
            "scene_token": s["scene_token"],
            "frame_token": s["frame_token"],
            "local_image_path": str(dst.relative_to(BASE_DIR)),
            "qa_preview": s["qa_preview"],
        })
    return result


def from_local_dir(local_dir: Path, num: int, seed: int):
    """로컬 이미지 폴더 직접 사용"""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_imgs = sorted([p for p in local_dir.iterdir() if p.suffix.lower() in exts])
    if not all_imgs:
        print(f"ERROR: {local_dir} 에 이미지 없음")
        return

    print(f"로컬 이미지: {len(all_imgs)}장")
    random.seed(seed)
    selected = random.sample(all_imgs, min(num, len(all_imgs)))

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, src in enumerate(selected):
        dst = IMAGE_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        samples.append({
            "image_id": src.stem,
            "scene_token": src.stem,
            "frame_token": src.stem,
            "local_image_path": str(dst.relative_to(BASE_DIR)),
            "qa_preview": "",
        })
        print(f"  [{i+1}/{len(selected)}] {src.name}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\n완료: {len(samples)}장 → {OUTPUT_JSON}")
    print("다음 단계: python 02_generate_gt.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num", type=int, default=NUM_SAMPLES)
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--extra", type=int, default=0,
                        help="기존 drivelm_samples.json에 추가로 N장 더 받기 (겹치지 않음)")
    args = parser.parse_args()

    if args.local_dir:
        from_local_dir(args.local_dir, args.num, args.seed)
        return

    if not args.token:
        print("ERROR: HF_TOKEN 필요. HF_TOKEN=hf_xxx python 01_download_drivelm.py")
        return

    # 1. QA JSON
    qa_data = load_qa_json(args.token)
    print(f"씬 수: {len(qa_data)}")

    # 2. 이미지 zip 다운로드 & 압축 해제
    extract_dir = download_and_extract_zip(args.token)

    # 3. QA + 이미지 매핑
    all_samples = build_samples(qa_data, extract_dir)
    print(f"매핑된 프레임: {len(all_samples)}장")

    if not all_samples:
        print("ERROR: 매핑 실패.")
        return

    # --extra 모드: 기존 샘플 제외하고 추가로 N장
    if args.extra > 0:
        existing = []
        if OUTPUT_JSON.exists():
            with open(OUTPUT_JSON) as f:
                existing = json.load(f)
        existing_ids = {s["image_id"] for s in existing}

        remaining = [s for s in all_samples if s["image_id"] not in existing_ids]
        random.seed(args.seed + 1)  # 다른 seed로 겹침 방지
        extra_selected = random.sample(remaining, min(args.extra, len(remaining)))
        print(f"추가 샘플링: {len(extra_selected)}장 (기존 {len(existing)}장 제외)")

        print("\nimages/ 복사 중...")
        extra_downloaded = copy_sampled_images(extra_selected)

        merged = existing + extra_downloaded
        with open(OUTPUT_JSON, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"\n완료: 총 {len(merged)}장 ({len(existing)} 기존 + {len(extra_downloaded)} 추가) → {OUTPUT_JSON}")
        print("다음 단계: python 02_generate_gt.py")
        return

    # 일반 모드: N장 샘플링
    random.seed(args.seed)
    selected = random.sample(all_samples, min(args.num, len(all_samples)))
    print(f"샘플링: {len(selected)}장 (seed={args.seed})")

    print(f"\nimages/ 복사 중...")
    downloaded = copy_sampled_images(selected)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(downloaded, f, indent=2, ensure_ascii=False)
    print(f"\n완료: {len(downloaded)}장 → {OUTPUT_JSON}")
    print("다음 단계: python 02_generate_gt.py")


if __name__ == "__main__":
    main()
