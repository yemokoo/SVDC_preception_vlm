#!/usr/bin/env python3
"""
Gemini 3 Flash로 DriveLM 이미지 100장 GT 레이블 생성

step2_gemini_detect_error.py 의 Vertex AI 패턴을 그대로 사용.
vlm_driving_common.py 의 스키마 기준으로 GT 생성:
  - road_type: highway | city | unknown
  - road_surface: dry | wet | unknown
  - hazard_present: true | false
  - hazard_type: none | obstacle | pedestrian_intrusion_risk | unknown

Usage:
    python 02_generate_gt.py
    VERTEX_PROJECT_ID=xxx python 02_generate_gt.py
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Optional, Any

from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-genai 없음. droidrun venv 사용:")
    print("  /home/sem/yemo/droidrun-android-world/.venv/bin/python 02_generate_gt.py")
    raise

# ============================================================================
# CONFIGURATION
# ============================================================================

VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "project-0c4cfc52-d7be-49ff-a36")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
MODEL_NAME = os.environ.get("VERTEX_GEMINI_MODEL", "gemini-3-flash-preview")

MAX_RETRY = 4
TEMPERATURE = 0.0        # GT 생성은 deterministic 하게
MAX_TOKENS = 8192        # thinking 모델은 <think> 블록이 길어서 넉넉하게
SLEEP_BETWEEN_CALLS = 1.0  # API rate limit 방지

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_JSON = BASE_DIR / "drivelm_samples.json"
GT_OUTPUT = BASE_DIR / "gt_labels.json"
IMAGE_DIR = BASE_DIR / "images"

# ============================================================================
# GT 생성 프롬프트 (vlm_driving_common.py 스키마 기반)
# ============================================================================

GT_PROMPT = """You are a ground truth annotator for an autonomous driving perception benchmark.

Carefully analyze this driving scene image and provide accurate ground truth labels.

Output ONLY a valid JSON object with these exact fields:
{
  "road_type": "highway" or "city" or "unknown",
  "road_surface": "dry" or "wet" or "unknown",
  "hazard_present": true or false,
  "hazard_type": "none" or "obstacle" or "pedestrian_intrusion_risk" or "unknown"
}

Field definitions:
- road_type: "highway" for freeway/expressway (multiple lanes, barriers, highway signs), "city" for urban/local roads, "unknown" if genuinely unclear
- road_surface: "wet" ONLY when road visibly appears wet, rainy, or has puddles; "dry" when visibly dry and clear; "unknown" if cannot determine
- hazard_present: true ONLY when there is a clear forward hazard (obstacle blocking path, or person actively entering/crossing lane)
- hazard_type: "none" when hazard_present is false; "obstacle" for stationary/moving objects blocking the path ahead; "pedestrian_intrusion_risk" for pedestrians near or actively entering the lane; "unknown" when hazard is present but type is unclear

Rules:
1. Output ONLY the JSON object — no text before or after
2. Do NOT use markdown code blocks
3. Use lowercase true/false for booleans
4. Be conservative: when in doubt about hazard, set hazard_present=false
5. hazard_type MUST be "none" when hazard_present is false
"""


# ============================================================================
# Gemini Client (step2_gemini_detect_error.py 패턴)
# ============================================================================

class GeminiGTAnnotator:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION,
        )
        self.model_name = MODEL_NAME
        self._gen_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise RuntimeError("No candidates in Gemini response")

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        merged = "\n".join(chunks).strip()
        if merged:
            return merged

        finish = getattr(candidates[0], "finish_reason", "UNKNOWN")
        raise RuntimeError(f"Empty Gemini response. finish_reason={finish}")

    def _parse_json(self, text: str) -> dict:
        # <think> 태그 제거 (Qwen 스타일 reasoning 모델 대응)
        cleaned = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL).strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]

        return json.loads(cleaned)

    def annotate(self, image_path: Path) -> dict:
        """이미지 1장 → GT JSON"""
        image = Image.open(image_path).convert("RGB")

        last_error = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[GT_PROMPT, image],
                    config=self._gen_config,
                )
                raw_text = self._extract_text(response)
                parsed = self._parse_json(raw_text)
                return {"status": "ok", "raw": raw_text, "gt": parsed}

            except json.JSONDecodeError as e:
                last_error = f"JSON 파싱 실패: {e}"
                break  # 파싱 실패는 재시도 불필요
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRY:
                    wait = min(2 ** (attempt - 1), 8)
                    print(f"    attempt {attempt}/{MAX_RETRY} 실패: {e} — {wait}s 후 재시도")
                    time.sleep(wait)

        return {"status": "error", "error": last_error, "gt": None}


# ============================================================================
# GT 정규화 (vlm_driving_common.py normalize_enum 동일 로직)
# ============================================================================

ROAD_TYPES = {"highway", "city", "unknown"}
ROAD_SURFACES = {"dry", "wet", "unknown"}
HAZARD_TYPES = {"none", "obstacle", "pedestrian_intrusion_risk", "unknown"}


def normalize_gt(raw_gt: dict) -> dict:
    def norm_enum(val, allowed):
        if isinstance(val, str):
            v = val.strip().lower().replace(" ", "_")
            if v in allowed:
                return v
        return "unknown"

    def norm_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"true", "1", "yes"}
        return False

    road_type = norm_enum(raw_gt.get("road_type"), ROAD_TYPES)
    road_surface = norm_enum(raw_gt.get("road_surface"), ROAD_SURFACES)
    hazard_present = norm_bool(raw_gt.get("hazard_present"))
    hazard_type = norm_enum(raw_gt.get("hazard_type"), HAZARD_TYPES)

    # 일관성 보정
    if not hazard_present:
        hazard_type = "none"

    return {
        "road_type": road_type,
        "road_surface": road_surface,
        "hazard_present": hazard_present,
        "hazard_type": hazard_type,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    # 이전 결과 로드 (재실행 안전)
    existing_results = {}
    if GT_OUTPUT.exists():
        with open(GT_OUTPUT, "r") as f:
            data = json.load(f)
        existing_results = {item["image_id"]: item for item in data.get("samples", [])}
        print(f"기존 GT 결과 {len(existing_results)}개 로드 (이미 처리된 것은 스킵)")

    # 샘플 목록 로드
    if not SAMPLES_JSON.exists():
        print(f"ERROR: {SAMPLES_JSON} 없음. 먼저 01_download_drivelm.py 실행하세요.")
        return

    with open(SAMPLES_JSON, "r") as f:
        samples = json.load(f)
    print(f"처리할 샘플: {len(samples)}장")

    annotator = GeminiGTAnnotator()
    results = []
    success = 0
    fail = 0

    for i, sample in enumerate(samples):
        image_id = sample.get("frame_token", f"sample_{i:03d}")

        # 재실행 시 기존 결과 재사용
        if image_id in existing_results:
            results.append(existing_results[image_id])
            print(f"[{i+1}/{len(samples)}] 스킵 (기존): {image_id}")
            success += 1
            continue

        # 이미지 경로 확인
        local_path = sample.get("local_image_path", "")
        img_path = BASE_DIR / local_path if local_path else None

        if not img_path or not img_path.exists():
            print(f"[{i+1}/{len(samples)}] ⚠️  이미지 없음: {local_path}")
            fail += 1
            continue

        print(f"[{i+1}/{len(samples)}] 처리 중: {image_id} ({img_path.name})")

        result = annotator.annotate(img_path)

        if result["status"] == "ok":
            gt_normalized = normalize_gt(result["gt"])
            entry = {
                "image_id": image_id,
                "image_path": local_path,
                "scene_token": sample.get("scene_token", ""),
                "qa_preview": sample.get("qa_preview", ""),
                "gt": gt_normalized,
                "gt_raw": result["raw"],
            }
            results.append(entry)
            success += 1
            print(f"  ✅ {gt_normalized}")
        else:
            print(f"  ❌ 실패: {result['error']}")
            fail += 1

        # 중간 저장 (10개마다)
        if (i + 1) % 10 == 0:
            _save(results, success, fail)

        time.sleep(SLEEP_BETWEEN_CALLS)

    _save(results, success, fail)
    print(f"\n완료: 성공 {success}, 실패 {fail}")
    print(f"GT 저장: {GT_OUTPUT}")
    print("\n다음 단계: python 03_evaluate.py")


def _save(results: list, success: int, fail: int):
    output = {
        "meta": {
            "model": MODEL_NAME,
            "total": len(results),
            "success": success,
            "fail": fail,
        },
        "samples": results,
    }
    with open(GT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
