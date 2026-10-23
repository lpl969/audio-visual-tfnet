import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import AudioDataset
from models import TFNet


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(save_dir: str, file_name: str, logger_name: str):
    ensure_dir(save_dir)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(save_dir, file_name), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def resolve_manifest(root: str, version: str, split: str, fold: int):
    annotation_dir = os.path.join(root, "annotation", version)
    candidates = []
    if fold >= 0:
        candidates.extend([
            os.path.join(annotation_dir, f"fold{fold}_{split}.txt"),
            os.path.join(annotation_dir, f"fold{fold}_{split}_shuffled.txt"),
        ])
    candidates.extend([
        os.path.join(annotation_dir, f"{split}_shuffled.txt"),
        os.path.join(annotation_dir, f"{split}.txt"),
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find {split} manifest under {annotation_dir}")


def build_output_dir(base_dir: str, fold: int):
    out_dir = base_dir
    if fold >= 0:
        out_dir = os.path.join(out_dir, f"fold_{fold}")
    ensure_dir(out_dir)
    return out_dir


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def compute_batch_metrics(cls_logits, pos_pred, batch, cls_loss_fn, pos_loss_fn, pos_weight: float):
    cls_loss = cls_loss_fn(cls_logits, batch["label"])

    valid = batch["position_mask"] > 0.5
    if valid.any():
        pos_loss = pos_loss_fn(pos_pred[valid], batch["position"][valid])
    else:
        pos_loss = cls_loss.new_tensor(0.0)

    total_loss = cls_loss + pos_weight * pos_loss

    preds = torch.argmax(cls_logits, dim=1)
    correct = (preds == batch["label"]).sum().item()
    total = batch["label"].size(0)

    valid_mask = batch["position_mask"] > 0.5
    if valid_mask.any():
        pos_diff = torch.abs(pos_pred[valid_mask] - batch["position"][valid_mask])
        ape = torch.norm(pos_pred[valid_mask] - batch["position"][valid_mask], dim=1)
        pos_count = int(valid_mask.sum().item())
        dx = pos_diff[:, 0].sum().item()
        dy = pos_diff[:, 1].sum().item()
        dz = pos_diff[:, 2].sum().item()
        ape_sum = ape.sum().item()
    else:
        pos_count = 0
        dx = dy = dz = ape_sum = 0.0

    return {
        "loss": total_loss,
        "cls_loss": cls_loss.detach().item(),
        "pos_loss": pos_loss.detach().item(),
        "correct": correct,
        "total": total,
        "pos_count": pos_count,
        "dx_sum": dx,
        "dy_sum": dy,
        "dz_sum": dz,
        "ape_sum": ape_sum,
    }


def run_epoch(model, loader, device, cls_loss_fn, pos_loss_fn, pos_weight, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pos_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_pos_samples = 0
    dx_sum = dy_sum = dz_sum = ape_sum = 0.0

    progress = tqdm(loader, total=len(loader), desc="Train" if is_train else "Val", unit="batch")
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            cls_logits, pos_pred = model(batch["audio"], batch["shc"], batch["image_224"])
            metrics = compute_batch_metrics(cls_logits, pos_pred, batch, cls_loss_fn, pos_loss_fn, pos_weight)

            if is_train:
                metrics["loss"].backward()
                optimizer.step()

            total_loss += metrics["loss"].detach().item()
            total_cls_loss += metrics["cls_loss"]
            total_pos_loss += metrics["pos_loss"]
            total_correct += metrics["correct"]
            total_samples += metrics["total"]
            total_pos_samples += metrics["pos_count"]
            dx_sum += metrics["dx_sum"]
            dy_sum += metrics["dy_sum"]
            dz_sum += metrics["dz_sum"]
            ape_sum += metrics["ape_sum"]

    denom = max(len(loader), 1)
    pos_denom = max(total_pos_samples, 1)
    return {
        "loss": total_loss / denom,
        "cls_loss": total_cls_loss / denom,
        "pos_loss": total_pos_loss / denom,
        "acc": total_correct / max(total_samples, 1),
        "dx_mae": dx_sum / pos_denom,
        "dy_mae": dy_sum / pos_denom,
        "dz_mae": dz_sum / pos_denom,
        "ape": ape_sum / pos_denom,
        "valid_pos_samples": total_pos_samples,
    }


def save_checkpoint(path: str, model, optimizer, epoch: int, best_score: float, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_ape": best_score,
            "args": public_config(args),
        },
        path,
    )


def build_dataloaders(args):
    train_manifest = resolve_manifest(args.root, args.annotation_version, "train", args.fold)
    val_manifest = resolve_manifest(args.root, args.annotation_version, "val", args.fold)

    audio_dir = os.path.join(args.root, args.audio_dir)
    image_dir = os.path.join(args.root, "visual_data", args.image_dir)
    cls_dir = os.path.join(args.root, args.label_dir)
    pos_dir = os.path.join(args.root, args.gt_dir)

    train_ds = AudioDataset(train_manifest, cls_dir, pos_dir, audio_dir, image_dir, return_dict=True)
    val_ds = AudioDataset(val_manifest, cls_dir, pos_dir, audio_dir, image_dir, return_dict=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_manifest, val_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Train the full audio-visual TFNet")
    parser.add_argument("--root", type=str, required=True, help="Local dataset root (never stored in checkpoints)")
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--annotation_version", type=str, default="V3")
    parser.add_argument("--audio_dir", type=str, default="audio_npy")
    parser.add_argument("--image_dir", type=str, default="image_npy")
    parser.add_argument("--label_dir", type=str, default="label")
    parser.add_argument("--gt_dir", type=str, default="gt")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epoch", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pos_weight", type=float, default=1.5)
    parser.add_argument("--num_cls", type=int, default=5)
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def public_config(args):
    """Only save numeric training settings, never local data/output paths."""
    keys = ("fold", "batch_size", "train_epoch", "num_workers", "lr",
            "weight_decay", "pos_weight", "num_cls", "seed")
    return {key: getattr(args, key) for key in keys}


def main():
    args = parse_args()

    seed_everything(args.seed)
    out_dir = build_output_dir(args.save_path, args.fold)
    logger = setup_logger(out_dir, "train.log", f"train.{args.fold}")

    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(public_config(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, train_manifest, val_manifest = build_dataloaders(args)
    logger.info("Samples: train=%d, val=%d", len(train_loader.dataset), len(val_loader.dataset))

    model = TFNet(num_cls=args.num_cls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    pos_loss_fn = torch.nn.L1Loss()

    best_val_ape = float("inf")
    best_path = os.path.join(out_dir, "best.pth")

    for epoch in range(1, args.train_epoch + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            cls_loss_fn,
            pos_loss_fn,
            args.pos_weight,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            cls_loss_fn,
            pos_loss_fn,
            args.pos_weight,
            optimizer=None,
        )

        logger.info(
            "Epoch [%03d/%03d] | "
            "Train Loss %.4f | Train ACC %.4f | Train APE %.4f | "
            "Val Loss %.4f | Val ACC %.4f | Val APE %.4f",
            epoch,
            args.train_epoch,
            train_metrics["loss"],
            train_metrics["acc"],
            train_metrics["ape"],
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["ape"],
        )

        if val_metrics["valid_pos_samples"] == 0:
            raise ValueError("Validation requires at least one valid position for APE selection")
        if val_metrics["ape"] < best_val_ape:
            best_val_ape = val_metrics["ape"]
            save_checkpoint(best_path, model, optimizer, epoch, best_val_ape, args)
            logger.info(
                "Best updated at epoch %03d | Val Loss %.4f | ACC %.4f | APE %.4f | Saved: %s",
                epoch,
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["ape"],
                "best.pth",
            )

    logger.info("Training finished. Best validation APE: %.4f", best_val_ape)


if __name__ == "__main__":
    main()
