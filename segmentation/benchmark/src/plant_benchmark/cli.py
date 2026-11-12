"""Command-line entry point for the target-plant segmentation benchmark.

Run from ``segmentation/benchmark`` with ``PYTHONPATH=src``:
``python -m plant_benchmark.cli <command> --config configs/fixed_split.yaml ...``
"""

from __future__ import annotations

import argparse
import platform

import torch

from .config import load_config
from .data import (
    PlantDataset,
    load_mask,
    load_upright_image,
    prepare_yolo26_dataset,
    read_manifest,
)
from .engine import (
    combined_loss,
    model_input_tensors,
    predict,
    sam_inputs,
    sam_logits,
    sam_postprocess_logits,
    sam_target,
    train,
)
from .models import create_dense_model, create_sam21, freeze_batch_norm, set_train_scope


MODELS = ("sam21", "birefnet_hr", "segformer_b2", "deeplabv3plus_r50", "yolo26")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="plant_benchmark.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = root.add_subparsers(dest="command", required=True)
    item = commands.add_parser(
        "doctor", help="Print environment information and manifest split counts."
    )
    item.add_argument("--config", required=True)
    item = commands.add_parser(
        "prepare-yolo",
        help="Export upright JPEGs and class-map PNGs for YOLO26 from the manifest.",
    )
    item.add_argument("--config", required=True)
    item = commands.add_parser(
        "smoke", help="One-image forward (or forward+backward) check of a model."
    )
    item.add_argument("--config", required=True)
    item.add_argument("--model", required=True, choices=MODELS)
    item.add_argument("--backward", action="store_true")
    for command, text in (
        ("train", "Train a model; best.pt is the epoch with the lowest validation loss."),
        (
            "predict",
            "Select the threshold on the validation split, then write binary "
            "val/test masks.",
        ),
    ):
        item = commands.add_parser(command, help=text)
        item.add_argument("--config", required=True)
        item.add_argument("--model", required=True, choices=MODELS)
        item.add_argument("--protocol", choices=("pretrained", "finetuned"), required=True)
        item.add_argument("--fold", type=int, required=True)
    return root


def doctor(cfg: dict) -> None:
    frame = read_manifest(cfg)
    print("platform:", platform.platform())
    print("python project:", cfg["_project_dir"])
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    print("gpu count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"gpu {index}:", torch.cuda.get_device_name(index))
    print("dataset:", cfg["paths"]["dataset"])
    print("records:", len(frame))
    if "split" in frame.columns:
        print("split mode: fixed image-level train/val")
        print("split counts:", frame["split"].value_counts().sort_index().to_dict())
    else:
        print("fold counts:", frame["fold"].value_counts().sort_index().to_dict())
    print("manual masks:", int(frame["manually_edited"].sum()))
    if len(frame) == 0:
        raise SystemExit("Dataset is empty")


def smoke(cfg: dict, model_name: str, backward: bool) -> None:
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    frame = read_manifest(cfg).head(1)
    if model_name == "yolo26":
        from .yolo26 import smoke_yolo26

        shape, _ = smoke_yolo26(cfg, backward)
    elif model_name == "sam21":
        model, processor = create_sam21(cfg)
        row = frame.iloc[0]
        image = load_upright_image(row.image_path)
        if backward:
            mask = load_mask(row.mask_path)
            inputs = sam_inputs(processor, image, row.seed_x, row.seed_y, mask)
            set_train_scope(model, cfg["models"]["sam21"]["train_scope"])
            model.to(target_device).train()
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.float16,
                enabled=target_device.type == "cuda",
            ):
                logits = sam_logits(model, inputs, target_device)
                target = sam_target(inputs, logits, target_device)
                loss = combined_loss(logits, target, cfg)
            loss.backward()
            shape = tuple(logits.shape)
        else:
            inputs = sam_inputs(processor, image, row.seed_x, row.seed_y)
            model.to(target_device).eval()
            with torch.inference_mode(), torch.autocast(
                device_type=target_device.type,
                dtype=torch.float16,
                enabled=target_device.type == "cuda",
            ):
                outputs = model(
                    **model_input_tensors(inputs, target_device), multimask_output=False
                )
            masks = sam_postprocess_logits(processor, outputs.pred_masks, inputs)
            shape = tuple(masks.shape)
    else:
        model = create_dense_model(model_name, cfg).to(target_device)
        if backward and model_name == "birefnet_hr":
            model.float()
        sample = PlantDataset(
            frame, int(cfg["data"]["height"]), int(cfg["data"]["width"])
        )[0]
        batch_size = int(cfg["models"][model_name]["batch_size"]) if backward else 1
        image_tensor = sample["image"].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        target = sample["mask"].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        image_tensor, target = image_tensor.to(target_device), target.to(target_device)
        if backward:
            if model_name == "birefnet_hr":
                set_train_scope(model, cfg["models"][model_name]["train_scope"])
            model.train()
            if model_name == "birefnet_hr" and batch_size == 1:
                freeze_batch_norm(model)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(cfg["models"][model_name]["learning_rate"]),
            )
            scaler = torch.amp.GradScaler(
                "cuda", enabled=target_device.type == "cuda"
            )
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.float16,
                enabled=target_device.type == "cuda",
            ):
                logits = model(image_tensor)
                loss = combined_loss(logits, target, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            model.eval()
            with torch.inference_mode(), torch.autocast(
                device_type=target_device.type,
                dtype=torch.float16,
                enabled=target_device.type == "cuda",
            ):
                logits = model(image_tensor)
        shape = tuple(logits.shape)
    peak = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if target_device.type == "cuda"
        else 0.0
    )
    mode = "forward+backward" if backward else "forward"
    print(f"{model_name}: {mode} smoke OK, output={shape}, peak_cuda_gib={peak:.2f}")


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare-yolo":
        frame = read_manifest(cfg)
        yolo_path = prepare_yolo26_dataset(cfg, frame)
        print(f"Prepared YOLO26 data for {len(frame)} manifest records: {yolo_path}")
    elif args.command == "doctor":
        doctor(cfg)
    elif args.command == "smoke":
        smoke(cfg, args.model, args.backward)
    elif args.command == "train":
        train(cfg, args.model, args.protocol, args.fold)
    elif args.command == "predict":
        predict(cfg, args.model, args.protocol, args.fold)


if __name__ == "__main__":
    main()
