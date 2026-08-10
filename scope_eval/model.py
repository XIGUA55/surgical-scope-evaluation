from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import seed_everything, write_json


class CocoInstrumentDataset(Dataset):
    def __init__(self, annotation_path: Path, dataset_root: Path, processor, augmentation: dict | None = None) -> None:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.images = payload["images"]
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            self.annotations[int(annotation["image_id"])].append(annotation)
        self.dataset_root = dataset_root
        self.processor = processor
        self.augmentation = augmentation or {}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_info = self.images[index]
        image = Image.open(self.dataset_root / image_info["file_name"]).convert("RGB")
        annotations = [
            {
                "id": int(row["id"]), "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]), "bbox": list(row["bbox"]),
                "area": float(row["area"]), "iscrowd": int(row.get("iscrowd", 0)),
            }
            for row in self.annotations[int(image_info["id"])]
        ]
        flip_probability = float(self.augmentation.get("horizontal_flip_probability", 0.0))
        if flip_probability and random.random() < flip_probability:
            from PIL import ImageOps
            image = ImageOps.mirror(image)
            width = float(image_info["width"])
            for row in annotations:
                row["bbox"][0] = width - row["bbox"][0] - row["bbox"][2]
        jitter = float(self.augmentation.get("color_jitter", 0.0))
        if jitter:
            from torchvision.transforms import ColorJitter
            image = ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.1, jitter / 2))(image)
        target = {
            "image_id": int(image_info["id"]),
            "annotations": annotations,
        }
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"].squeeze(0), "labels": encoded["labels"][0]}


def make_collator(processor):
    def collate(batch: list[dict]) -> dict:
        encoded = processor.pad([row["pixel_values"] for row in batch], return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"], "pixel_mask": encoded["pixel_mask"], "labels": [row["labels"] for row in batch]}
    return collate


def _target_xyxy(labels: dict) -> torch.Tensor:
    boxes = labels["boxes"]
    size = labels["orig_size"].to(boxes.device)
    height, width = size[0], size[1]
    cx, cy, box_width, box_height = boxes.unbind(-1)
    result = torch.stack((cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2), dim=-1)
    result = result * torch.stack((width, height, width, height))
    return result


@torch.no_grad()
def evaluate_model(model, loader, processor, accelerator) -> dict[str, float]:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    model.eval()
    total_loss, batches = 0.0, 0
    for batch in loader:
        outputs = model(**batch)
        total_loss += float(outputs.loss.detach().item())
        batches += 1
        target_sizes = torch.stack([labels["orig_size"] for labels in batch["labels"]])
        predictions = processor.post_process_object_detection(outputs, threshold=0.001, target_sizes=target_sizes)
        targets = [
            {"boxes": _target_xyxy(labels).cpu(), "labels": labels["class_labels"].cpu()}
            for labels in batch["labels"]
        ]
        predictions = [{key: value.cpu() for key, value in row.items()} for row in predictions]
        metric.update(predictions, targets)
    computed = metric.compute()
    result = {key: float(value.item()) for key, value in computed.items() if value.numel() == 1}
    result["loss"] = total_loss / max(1, batches)
    model.train()
    return result


def _annotation_path(config: dict, split: str) -> Path:
    path = config["artifacts_dir"] / "dataset" / f"annotations_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run prepare and split first")
    return path


def train_model(config: dict, smoke: bool = False) -> Path:
    from accelerate import Accelerator
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    seed_everything(int(config["seed"]))
    settings = config["model"]
    train_split, val_split = ("smoke_train", "smoke_val") if smoke else ("train", "val")
    processor = AutoImageProcessor.from_pretrained(settings["checkpoint"], size={"height": settings["image_size"], "width": settings["image_size"]})
    dataset_root = config["artifacts_dir"] / "dataset"
    train_data = CocoInstrumentDataset(_annotation_path(config, train_split), dataset_root, processor, augmentation=settings.get("augmentation"))
    val_data = CocoInstrumentDataset(_annotation_path(config, val_split), dataset_root, processor)
    if not train_data or not val_data:
        raise ValueError("Training and validation datasets must both be non-empty")
    collator = make_collator(processor)
    workers = 0 if smoke else int(settings["num_workers"])
    train_loader = DataLoader(train_data, batch_size=min(int(settings["batch_size"]), len(train_data)), shuffle=True, num_workers=workers, collate_fn=collator)
    val_loader = DataLoader(val_data, batch_size=min(int(settings["batch_size"]), len(val_data)), shuffle=False, num_workers=workers, collate_fn=collator)
    id2label = config["class_names"]
    label2id = {value: key for key, value in id2label.items()}
    model = RTDetrV2ForObjectDetection.from_pretrained(
        settings["checkpoint"], num_labels=len(id2label), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True, initializer_bias_prior_prob=float(settings["classifier_prior_probability"]),
    )
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        (backbone if "backbone" in name else other).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": float(settings["backbone_learning_rate"])},
        {"params": other, "lr": float(settings["learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]))
    epochs = 1 if smoke else int(settings["epochs"])
    accumulation = 1 if smoke else int(settings["gradient_accumulation"])
    total_steps = max(1, math.ceil(len(train_loader) / accumulation) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no", gradient_accumulation_steps=accumulation)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    run_dir = config["artifacts_dir"] / "runs" / ("smoke" if smoke else "formal")
    run_dir.mkdir(parents=True, exist_ok=True)
    history, best_map, stale = [], -1.0, 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        train_loss, batches = 0.0, 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(outputs.loss.detach().item())
            batches += 1
        validation = evaluate_model(model, val_loader, processor, accelerator)
        row = {"epoch": epoch + 1, "train_loss": train_loss / max(1, batches), **{f"val_{key}": value for key, value in validation.items()}}
        history.append(row)
        if accelerator.is_main_process:
            write_json(run_dir / "history.json", history)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        current_map = validation.get("map", -1.0)
        if accelerator.is_main_process and current_map > best_map:
            best_map, stale = current_map, 0
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(run_dir / "best", save_function=accelerator.save)
            processor.save_pretrained(run_dir / "best")
        else:
            stale += 1
        if stale >= int(settings["early_stopping_patience"]):
            break
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(run_dir / "last", save_function=accelerator.save)
        processor.save_pretrained(run_dir / "last")
        write_json(run_dir / "history.json", history)
        manifest = pd.read_csv(dataset_root / "manifest.csv")
        split_cases = {split: sorted(manifest.loc[manifest.split == split, "case_id"].unique().tolist()) for split in (train_split, val_split)}
        independent_cases = set(split_cases[train_split]).isdisjoint(split_cases[val_split])
        write_json(run_dir / "run_metadata.json", {
            "smoke": smoke, "formal_research_result": not smoke and independent_cases,
            "experiment_type": "smoke" if smoke else "case_holdout" if independent_cases else "single_case_segment_holdout",
            "independent_case_generalization": independent_cases, "split_cases": split_cases,
            "classifier_prior_probability": float(settings["classifier_prior_probability"]), "best_map": best_map,
        })
    accelerator.wait_for_everyone()
    return run_dir / "best"


def evaluate_checkpoint(config: dict, checkpoint: Path, split: str = "test") -> dict[str, float]:
    from accelerate import Accelerator
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
    dataset = CocoInstrumentDataset(_annotation_path(config, split), config["artifacts_dir"] / "dataset", processor)
    evaluation_batch_size = int(config.get("evaluation", {}).get("batch_size", config["model"]["batch_size"]))
    loader = DataLoader(dataset, batch_size=evaluation_batch_size, shuffle=False, collate_fn=make_collator(processor))
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
    model, loader = accelerator.prepare(model, loader)
    metrics = evaluate_model(model, loader, processor, accelerator)
    output = config["artifacts_dir"] / "evaluation" / f"{split}_metrics.json"
    write_json(output, metrics)
    return metrics
