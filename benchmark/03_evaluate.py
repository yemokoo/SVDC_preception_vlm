#!/usr/bin/env python3
"""
VLM (qwen3vl_8b) 벤치마크 평가 스크립트

gt_labels.json (Gemini GT) vs 우리 VLM 출력 비교.
필드별 Accuracy / Precision / Recall / F1 리포트 출력.

Usage:
    python 03_evaluate.py
    VLLM_BASE_URL=http://192.168.0.87:8000 python 03_evaluate.py
    python 03_evaluate.py --gt-only   # GT 분포만 보기 (VLM 호출 없음)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import requests
from PIL import Image

# vlm_driving_common.py 임포트 (상위 디렉터리)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from svdc_preception_vlm.vlm_driving_common import (
    analyze_frame_with_vlm,
    VLLM_BASE_URL as DEFAULT_VLLM_URL,
    MODEL_NAME as DEFAULT_MODEL,
    ROAD_TYPES,
    ROAD_SURFACES,
    HAZARD_TYPES,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", DEFAULT_VLLM_URL)
SLEEP_BETWEEN_CALLS = 0.5

BASE_DIR = Path(__file__).resolve().parent
GT_JSON = BASE_DIR / "gt_labels.json"
RESULTS_JSON = BASE_DIR / "eval_results.json"
REPORT_TXT = BASE_DIR / "eval_report.txt"


# ============================================================================
# VLM 호출
# ============================================================================

def call_vlm(image_path: Path) -> dict | None:
    """qwen3vl_8b에 이미지 보내고 parse된 결과 반환"""
    import cv2
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        return None

    try:
        _, parsed = analyze_frame_with_vlm(frame_bgr)
        return parsed
    except Exception as e:
        print(f"    VLM 호출 실패: {e}")
        return None


def check_vllm_server() -> bool:
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        try:
            resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ============================================================================
# 메트릭 계산
# ============================================================================

def compute_metrics(gt_list: list, pred_list: list, labels: list) -> dict:
    """
    멀티클래스 per-class precision/recall/F1 + macro F1
    gt_list, pred_list: 같은 길이의 라벨 리스트
    """
    # 혼동 행렬
    matrix = defaultdict(Counter)
    for gt, pred in zip(gt_list, pred_list):
        matrix[gt][pred] += 1

    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {"tp": tp, "fp": fp, "fn": fn,
                             "precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(g == p for g, p in zip(gt_list, pred_list)) / len(gt_list) if gt_list else 0.0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels) if labels else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {k: {m: round(v2, 4) for m, v2 in v.items()} for k, v in per_class.items()},
        "confusion_matrix": {gt: dict(pred_counts) for gt, pred_counts in matrix.items()},
    }


def compute_binary_metrics(gt_list: list[bool], pred_list: list[bool]) -> dict:
    tp = sum(g and p for g, p in zip(gt_list, pred_list))
    fp = sum(not g and p for g, p in zip(gt_list, pred_list))
    fn = sum(g and not p for g, p in zip(gt_list, pred_list))
    tn = sum(not g and not p for g, p in zip(gt_list, pred_list))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(gt_list) if gt_list else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ============================================================================
# 리포트 출력
# ============================================================================

def print_report(metrics: dict, total: int, vlm_fail: int) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("  SVDC VLM Driving Perception Benchmark — Evaluation Report")
    lines.append("=" * 65)
    lines.append(f"  총 샘플: {total}  |  VLM 호출 실패: {vlm_fail}  |  평가: {total - vlm_fail}")
    lines.append("")

    # road_type
    rt = metrics.get("road_type", {})
    lines.append("[ road_type ]  Accuracy: {:.1f}%  Macro F1: {:.3f}".format(
        rt.get("accuracy", 0) * 100, rt.get("macro_f1", 0)))
    for label, v in rt.get("per_class", {}).items():
        lines.append(f"  {label:30s}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")

    lines.append("")

    # road_surface
    rs = metrics.get("road_surface", {})
    lines.append("[ road_surface ]  Accuracy: {:.1f}%  Macro F1: {:.3f}".format(
        rs.get("accuracy", 0) * 100, rs.get("macro_f1", 0)))
    for label, v in rs.get("per_class", {}).items():
        lines.append(f"  {label:30s}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")

    lines.append("")

    # hazard_present
    hp = metrics.get("hazard_present", {})
    lines.append("[ hazard_present ]  Accuracy: {:.1f}%  F1: {:.3f}".format(
        hp.get("accuracy", 0) * 100, hp.get("f1", 0)))
    lines.append(f"  TP={hp.get('tp',0)}  FP={hp.get('fp',0)}  FN={hp.get('fn',0)}  TN={hp.get('tn',0)}")
    lines.append(f"  Precision={hp.get('precision',0):.3f}  Recall={hp.get('recall',0):.3f}")

    lines.append("")

    # hazard_type (hazard_present=True 서브셋)
    ht = metrics.get("hazard_type", {})
    n_hazard = ht.pop("n_samples", 0)
    lines.append("[ hazard_type ]  (hazard 있는 {}개 서브셋)  Accuracy: {:.1f}%  Macro F1: {:.3f}".format(
        n_hazard, ht.get("accuracy", 0) * 100, ht.get("macro_f1", 0)))
    for label, v in ht.get("per_class", {}).items():
        lines.append(f"  {label:30s}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")

    lines.append("")

    # JSON format compliance
    fmt = metrics.get("format", {})
    lines.append("[ JSON Format Compliance ]")
    lines.append(f"  파싱 성공률:    {fmt.get('parse_rate', 0)*100:.1f}%")
    lines.append(f"  enum 이탈률:    {fmt.get('enum_violation_rate', 0)*100:.1f}%")

    lines.append("=" * 65)

    report = "\n".join(lines)
    print(report)
    return report


# ============================================================================
# GT 분포 요약
# ============================================================================

def print_gt_distribution(samples: list):
    rt_cnt = Counter(s["gt"]["road_type"] for s in samples)
    rs_cnt = Counter(s["gt"]["road_surface"] for s in samples)
    hp_cnt = Counter(s["gt"]["hazard_present"] for s in samples)
    ht_cnt = Counter(s["gt"]["hazard_type"] for s in samples)

    print("\n=== GT 분포 ===")
    print(f"road_type:    {dict(rt_cnt)}")
    print(f"road_surface: {dict(rs_cnt)}")
    print(f"hazard_present: {dict(hp_cnt)}")
    print(f"hazard_type:  {dict(ht_cnt)}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-only", action="store_true", help="GT 분포만 출력, VLM 호출 없음")
    parser.add_argument("--resume", action="store_true", help="기존 eval_results.json 이어서 실행")
    parser.add_argument("--top", type=int, default=0, help="GT 파싱 성공한 것 중 상위 N개만 평가 (0=전체)")
    args = parser.parse_args()

    # GT 로드
    if not GT_JSON.exists():
        print(f"ERROR: {GT_JSON} 없음. 먼저 02_generate_gt.py 실행하세요.")
        return

    with open(GT_JSON, "r") as f:
        gt_data = json.load(f)
    samples = gt_data.get("samples", [])
    # GT 파싱 성공한 것만 (gt 필드가 있는 것)
    samples = [s for s in samples if s.get("gt") is not None]
    if args.top > 0:
        samples = samples[:args.top]
    print(f"GT 로드: {len(samples)}개 샘플")
    print_gt_distribution(samples)

    if args.gt_only:
        return

    # VLM 서버 확인
    print(f"VLM 서버 확인: {VLLM_BASE_URL} ...", end=" ")
    if not check_vllm_server():
        print("❌ 연결 실패")
        print("VLLM_BASE_URL 환경변수를 확인하거나 서버를 시작하세요.")
        return
    print("✅")

    # 기존 결과 로드 (resume)
    existing = {}
    if args.resume and RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r") as f:
            prev = json.load(f)
        existing = {r["image_id"]: r for r in prev.get("results", [])}
        print(f"기존 평가 결과 {len(existing)}개 재사용")

    # VLM 실행
    all_results = []
    vlm_fail = 0

    for i, sample in enumerate(samples):
        image_id = sample["image_id"]

        if image_id in existing:
            all_results.append(existing[image_id])
            print(f"[{i+1}/{len(samples)}] 스킵: {image_id}")
            continue

        img_path = BASE_DIR / sample["image_path"]
        print(f"[{i+1}/{len(samples)}] {image_id} ({img_path.name}) ...", end=" ")

        start = time.time()
        vlm_output = call_vlm(img_path)
        elapsed = time.time() - start

        if vlm_output is None:
            print(f"❌ 실패 ({elapsed:.1f}s)")
            vlm_fail += 1
            all_results.append({
                "image_id": image_id,
                "image_path": sample["image_path"],
                "gt": sample["gt"],
                "pred": None,
                "elapsed": elapsed,
                "status": "error",
            })
        else:
            print(f"✅ {elapsed:.1f}s")
            all_results.append({
                "image_id": image_id,
                "image_path": sample["image_path"],
                "gt": sample["gt"],
                "pred": vlm_output,
                "elapsed": elapsed,
                "status": "ok",
            })

        # 중간 저장
        if (i + 1) % 10 == 0:
            _save_results(all_results)

        time.sleep(SLEEP_BETWEEN_CALLS)

    _save_results(all_results)

    # 메트릭 계산
    metrics = compute_all_metrics(all_results)

    # 리포트
    report = print_report(metrics, len(all_results), vlm_fail)
    with open(REPORT_TXT, "w") as f:
        f.write(report)
    print(f"\n리포트 저장: {REPORT_TXT}")
    print(f"상세 결과: {RESULTS_JSON}")


def compute_all_metrics(all_results: list) -> dict:
    valid = [r for r in all_results if r["status"] == "ok" and r["pred"] is not None]

    # --- road_type ---
    rt_gt = [r["gt"]["road_type"] for r in valid]
    rt_pred = [r["pred"].get("road_type", "unknown") for r in valid]
    rt_metrics = compute_metrics(rt_gt, rt_pred, sorted(ROAD_TYPES))

    # --- road_surface ---
    rs_gt = [r["gt"]["road_surface"] for r in valid]
    rs_pred = [r["pred"].get("road_surface", "unknown") for r in valid]
    rs_metrics = compute_metrics(rs_gt, rs_pred, sorted(ROAD_SURFACES))

    # --- hazard_present ---
    hp_gt = [r["gt"]["hazard_present"] for r in valid]
    hp_pred = [r["pred"].get("hazard_present", False) for r in valid]
    hp_metrics = compute_binary_metrics(hp_gt, hp_pred)

    # --- hazard_type (hazard_present=True 서브셋) ---
    hazard_subset = [r for r in valid if r["gt"]["hazard_present"]]
    if hazard_subset:
        ht_gt = [r["gt"]["hazard_type"] for r in hazard_subset]
        ht_pred = [r["pred"].get("hazard_type", "unknown") for r in hazard_subset]
        ht_metrics = compute_metrics(ht_gt, ht_pred, sorted(HAZARD_TYPES - {"none"}))
        ht_metrics["n_samples"] = len(hazard_subset)
    else:
        ht_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "n_samples": 0}

    # --- JSON format compliance ---
    total = len(all_results)
    parse_ok = sum(1 for r in all_results if r["status"] == "ok")
    enum_violations = sum(
        1 for r in all_results
        if r["status"] == "ok" and r["pred"] is not None
        and (
            r["pred"].get("road_type") not in ROAD_TYPES
            or r["pred"].get("road_surface") not in ROAD_SURFACES
            or r["pred"].get("hazard_type") not in HAZARD_TYPES
        )
    )
    fmt_metrics = {
        "parse_rate": parse_ok / total if total else 0.0,
        "enum_violation_rate": enum_violations / parse_ok if parse_ok else 0.0,
    }

    return {
        "road_type": rt_metrics,
        "road_surface": rs_metrics,
        "hazard_present": hp_metrics,
        "hazard_type": ht_metrics,
        "format": fmt_metrics,
    }


def _save_results(results: list):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
