from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scope-eval", description="Transformer surgical instrument detection and camera evaluation pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    anomaly_samples = commands.add_parser("sample-anomalies")
    anomaly_samples.add_argument("--samples-per-reason", type=int, default=4)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--jpeg-quality", type=int, default=95)
    split = commands.add_parser("split")
    split.add_argument("--smoke", action="store_true")
    split.add_argument("--segment-holdout", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--smoke", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", default="test")
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--video", type=Path, required=True)
    infer.add_argument("--no-render", action="store_true")
    infer.add_argument("--max-frames", type=int)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--tracks", type=Path, nargs="+", required=True)
    metrics.add_argument("--output", type=Path)
    metrics.add_argument("--confidence-threshold", type=float)
    ratings = commands.add_parser("ratings")
    ratings.add_argument("--ratings", type=Path, required=True)
    ratings.add_argument("--metrics", type=Path, required=True)
    errors = commands.add_parser("analyze-errors")
    errors.add_argument("--checkpoint", type=Path, required=True)
    errors.add_argument("--validation-split", default="val")
    errors.add_argument("--test-split", default="test")
    register = commands.add_parser("register-experiment")
    register.add_argument("--name", required=True)
    register.add_argument("--architecture", required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--experiment-config", type=Path)
    register.add_argument("--metrics-dir", type=Path)
    register.add_argument("--notes", default="")
    dfine_candidates = commands.add_parser("dfine-candidates")
    dfine_candidates.add_argument("--dfine-config", type=Path, required=True)
    dfine_candidates.add_argument("--checkpoint", type=Path, required=True)
    dfine_candidates.add_argument("--split", choices=("train", "val", "test"), required=True)
    dfine_candidates.add_argument("--output", type=Path, required=True)
    dfine_candidates.add_argument("--image-size", type=int, default=640)
    temporal = commands.add_parser("temporal-evaluate")
    temporal.add_argument("--validation-candidates", type=Path, required=True)
    temporal.add_argument("--test-candidates", type=Path, required=True)
    temporal.add_argument("--output-dir", type=Path, required=True)
    dfine_infer = commands.add_parser("dfine-infer")
    dfine_infer.add_argument("--dfine-config", type=Path, required=True)
    dfine_infer.add_argument("--checkpoint", type=Path, required=True)
    dfine_infer.add_argument("--video", type=Path, required=True)
    dfine_infer.add_argument("--output-dir", type=Path, required=True)
    dfine_infer.add_argument("--temporal-summary", type=Path, required=True)
    dfine_infer.add_argument("--image-size", type=int, default=640)
    learned = commands.add_parser("learned-rerank")
    learned.add_argument("--validation-candidates", type=Path, required=True)
    learned.add_argument("--test-candidates", type=Path, required=True)
    learned.add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("rating-template")
    commands.add_parser("report")
    all_command = commands.add_parser("run-all")
    all_command.add_argument("--smoke", action="store_true")
    all_command.add_argument("--smoke-frames", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "audit":
        from .data import audit_dataset
        print(json.dumps(audit_dataset(config), ensure_ascii=False, indent=2))
    elif args.command == "sample-anomalies":
        from .data_quality import sample_annotation_anomalies
        path, samples = sample_annotation_anomalies(config, args.samples_per_reason)
        print(f"Saved {len(samples)} review candidates: {path}")
    elif args.command == "prepare":
        from .data import prepare_dataset
        manifest, coco = prepare_dataset(config, args.jpeg_quality)
        print(f"Prepared {manifest.image_id.nunique()} images and {len(manifest)} annotations: {coco}")
    elif args.command == "split":
        from .data import assign_splits
        manifest = assign_splits(config, smoke=args.smoke, segment_holdout=args.segment_holdout)
        print(manifest.groupby("split").image_id.nunique().to_string())
    elif args.command == "train":
        from .model import train_model
        print(train_model(config, smoke=args.smoke))
    elif args.command == "evaluate":
        from .model import evaluate_checkpoint
        print(json.dumps(evaluate_checkpoint(config, args.checkpoint, args.split), indent=2))
    elif args.command == "infer":
        from .inference import infer_video
        path, metadata = infer_video(config, args.checkpoint, args.video, render=not args.no_render, max_frames=args.max_frames)
        print(path)
        print(json.dumps(metadata, indent=2))
    elif args.command == "metrics":
        from .metrics import compute_metrics_files
        if args.confidence_threshold is not None:
            config["metrics"]["confidence_threshold"] = args.confidence_threshold
        print(compute_metrics_files(config, args.tracks, args.output))
    elif args.command == "ratings":
        from .ratings import analyze_ratings
        path, summary = analyze_ratings(config, args.ratings, args.metrics)
        print(path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "rating-template":
        from .ratings import write_rating_template
        print(write_rating_template(config))
    elif args.command == "analyze-errors":
        from .error_analysis import analyze_detection_errors
        path, summary = analyze_detection_errors(config, args.checkpoint, args.validation_split, args.test_split)
        print(path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "register-experiment":
        from .experiments import register_experiment
        path, record = register_experiment(
            config, args.name, args.architecture, args.checkpoint,
            args.experiment_config, args.metrics_dir, args.notes,
        )
        print(path)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    elif args.command == "dfine-candidates":
        from .dfine_adapter import extract_dfine_candidates
        path, metadata = extract_dfine_candidates(
            config, args.dfine_config, args.checkpoint, args.split,
            args.output, args.image_size,
        )
        print(path)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    elif args.command == "temporal-evaluate":
        from .dfine_adapter import evaluate_temporal_candidates
        path, summary = evaluate_temporal_candidates(
            args.validation_candidates, args.test_candidates, args.output_dir,
        )
        print(path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "dfine-infer":
        from .dfine_adapter import infer_dfine_video
        from .temporal import TemporalParameters
        summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
        parameters = TemporalParameters(**summary["parameters"])
        path, metadata = infer_dfine_video(
            config, args.dfine_config, args.checkpoint, args.video, args.output_dir,
            parameters, float(summary["threshold"]), args.image_size,
        )
        print(path)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    elif args.command == "learned-rerank":
        from .reranking import evaluate_learned_reranker
        path, summary = evaluate_learned_reranker(
            args.validation_candidates, args.test_candidates, args.output_dir,
        )
        print(path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "report":
        from .report import generate_report
        print("\n".join(map(str, generate_report(config))))
    elif args.command == "run-all":
        _run_all(config, smoke=args.smoke, smoke_frames=args.smoke_frames)


def _run_all(config: dict, smoke: bool, smoke_frames: int) -> None:
    from .data import assign_splits, audit_dataset, prepare_dataset
    from .inference import infer_video
    from .metrics import compute_metrics_file
    from .model import evaluate_checkpoint, train_model
    from .ratings import write_rating_template
    from .report import generate_report

    audit_dataset(config)
    prepare_dataset(config)
    assign_splits(config, smoke=smoke)
    checkpoint = train_model(config, smoke=smoke)
    split = "smoke_val" if smoke else "test"
    evaluate_checkpoint(config, checkpoint, split)
    ready_video = next(row for row in audit_dataset(config)["videos"] if row["readable"] and row["segment_id"] != row["case_id"])
    tracks, _ = infer_video(config, checkpoint, Path(ready_video["path"]), render=True, max_frames=smoke_frames if smoke else None)
    compute_metrics_file(config, tracks)
    write_rating_template(config)
    markdown, html = generate_report(config)
    print(f"Completed: {markdown}\n{html}")


if __name__ == "__main__":
    main()
