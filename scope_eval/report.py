from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _json_if_exists(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def generate_report(config: dict) -> tuple[Path, Path]:
    artifacts = config["artifacts_dir"]
    audit = _json_if_exists(artifacts / "audit" / "dataset_audit.json")
    formal_history = _json_if_exists(artifacts / "runs" / "formal" / "history.json")
    formal_metadata = _json_if_exists(artifacts / "runs" / "formal" / "run_metadata.json")
    smoke_history = _json_if_exists(artifacts / "runs" / "smoke" / "history.json")
    rating_summary_raw = _json_if_exists(artifacts / "ratings" / "summary.json")
    rating_summary = rating_summary_raw[0] if isinstance(rating_summary_raw, list) and rating_summary_raw else rating_summary_raw
    metrics_path = artifacts / "metrics" / "camera_metrics.csv"
    metrics = pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else []
    segment_evaluations = {
        path.stem.removeprefix("segment_").removesuffix("_metrics"): _json_if_exists(path)
        for path in sorted((artifacts / "evaluation").glob("segment_*_metrics.json"))
    }
    formal_evaluations = {
        split: _json_if_exists(artifacts / "evaluation" / f"{split}_metrics.json")
        for split in ("train", "val", "test")
        if (artifacts / "evaluation" / f"{split}_metrics.json").exists()
    }
    error_analysis = _json_if_exists(artifacts / "error_analysis" / "summary.json")
    error_images_path = artifacts / "error_analysis" / "per_image.csv"
    ranking_failures = None
    if error_images_path.exists():
        error_images = pd.read_csv(error_images_path)
        test_errors = error_images[error_images.split == "test"]
        ranking_failures = int(((test_errors.top_iou < 0.5) & (test_errors.best_iou_any >= 0.5)).sum())
    inference_metadata = {
        path.parent.name: _json_if_exists(path)
        for path in sorted((artifacts / "inference").glob("*/metadata.json"))
    }
    test_evaluation = formal_evaluations.get("test", {})
    model_usable = test_evaluation.get("map_50", 0.0) >= 0.5 and test_evaluation.get("map", 0.0) >= 0.25
    validity = {
        "model_quality_gate_passed": model_usable,
        "gate": "held-out test mAP50 >= 0.50 and mAP50-95 >= 0.25 (engineering minimum, not a clinical threshold)",
        "interpretation": (
            "模型达到最低工程门槛，可以继续做轨迹误差分析。"
            if model_usable else
            "模型未达到最低工程门槛；轨迹指标仅验证计算流程，不能用于比较真实扶镜质量。"
        ),
        "independent_case_generalization": False,
        "reason": "B/C/D 均来自同一病例 VID001。",
    }
    train_result, val_result, test_result = (formal_evaluations.get(key, {}) for key in ("train", "val", "test"))
    debug_lines = [
        "1. 单类别分类头默认先验 0.5 会造成大量高置信度查询；已改为 0.01，修复后初始损失从约 204 降至约 22.5。",
        f"2. 过拟合明显：train/val/test mAP50-95 分别为 {train_result.get('map', float('nan')):.3f} / {val_result.get('map', float('nan')):.3f} / {test_result.get('map', float('nan')):.3f}。",
        f"3. D 上候选排序是主要错误之一：{ranking_failures if ranking_failures is not None else 'NA'} 帧存在 IoU≥0.5 的候选，但最高分候选 IoU<0.5。",
        "4. 定性抽检发现高光组织假阳性、运动模糊/画面边缘漏检，以及不同段对器械工作端和杆部的标注范围不一致。",
        "5. IoU tracker 轨迹碎片严重；轨迹覆盖率不足 80% 时，运动学扶镜指标自动标记为不可靠。",
        "6. 下一步优先统一标注规范并增加独立病例，其次做跨段均衡采样、尺度/模糊/高光增强、置信度-IoU排序校准和 ByteTrack/Kalman 追踪。",
    ]
    lines = [
        "# 胰腺手术扶镜质量评价复现报告", "",
        "## 结果性质", "",
        (
            "正式跨病例研究结果。" if formal_history and formal_metadata and formal_metadata.get("independent_case_generalization")
            else "完整单病例分段留出实验，不代表跨病例泛化结果。" if formal_history
            else "当前为工程流程/单病例 smoke 验证，不代表跨病例泛化结果。"
        ), "",
        "## 数据审计", "", "```json", json.dumps(audit or {"status": "not run"}, ensure_ascii=False, indent=2), "```", "",
        "## RT-DETRv2 训练", "", "```json", json.dumps(formal_history or smoke_history or {"status": "not run"}, ensure_ascii=False, indent=2), "```", "",
        "## 正式 train/val/test 指标", "", "```json", json.dumps(formal_evaluations or {"status": "not run"}, ensure_ascii=False, indent=2), "```", "",
        "## 三段完整视频推理", "", "```json", json.dumps(inference_metadata or {"status": "not run"}, ensure_ascii=False, indent=2), "```", "",
        "## 扶镜轨迹指标", "", "```json", json.dumps(metrics or {"status": "not run"}, ensure_ascii=False, indent=2), "```", "",
        "## 结果有效性判定", "", "```json", json.dumps(validity, ensure_ascii=False, indent=2), "```", "",
        "## 检测误差分析", "", "```json", json.dumps(error_analysis or {"status": "not run"}, ensure_ascii=False, indent=2), "```", "",
        "## 调试结论与下一步", "", *debug_lines, "",
        "## 专家评分验证", "", "```json", json.dumps(rating_summary or {"status": "ratings not supplied"}, ensure_ascii=False, indent=2), "```", "",
        "## 可复现性", "", f"配置：`{config['config_path']}`", "", f"随机种子：`{config['seed']}`", "",
    ]
    report_dir = artifacts / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    html_path = report_dir / "report.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>扶镜质量评价复现报告</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:15px/1.5 sans-serif}pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style>"
        f"<pre>{html.escape(markdown_path.read_text(encoding='utf-8'))}</pre>", encoding="utf-8",
    )
    return markdown_path, html_path
