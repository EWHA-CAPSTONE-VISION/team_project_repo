from pyexpat import model
import warnings
warnings.filterwarnings('ignore')

import os
import json
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset.loader import CustomSample, create_wsi_dataloader
from models.model_ver1 import MultiModalMILModel

CONFIG_PATH = r"YOUR_PATH/configs/train.yaml"

def load_config(path=CONFIG_PATH):
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG = {
        # data
        "root_dir": cfg["data"]["root_dir"],
        "max_spots": cfg["data"]["max_spots"],
        "metadata_dir": cfg["data"]["metadata_dir"],
        "output_dir": "results_ver1",
        "hvg_path": cfg["data"]["hvg_path"],

        # model
        "num_genes": cfg["model"]["num_genes"],
        "num_classes": cfg["model"]["num_classes"],
        "embed_dim": cfg["model"]["embed_dim"],
        "fusion_option": cfg["model"]["fusion_option"],
        "top_k_genes": cfg["model"].get("top_k_genes"),

        # spatial attn
        "use_spatial_attn": cfg["model"].get("use_spatial_attn", False),
        "spatial_attn_k": cfg["model"].get("spatial_attn_k", 8),
        "spatial_attn_heads": cfg["model"].get("spatial_attn_heads", 4),
        "spatial_attn_dropout": cfg["model"].get("spatial_attn_dropout", 0.1),

        # training
        "epochs": cfg["training"]["epochs"],
        "lr": cfg["training"]["lr"],
        "weight_decay": cfg["training"]["weight_decay"],
        "batch_size": cfg["training"]["batch_size"],

        # memory
        "batch_spots": cfg["memory"]["batch_spots"],
        "accum_steps": cfg["memory"]["accum_steps"],
        "freeze_image_encoder": cfg["memory"]["freeze_image_encoder"],

        # misc
        "device": cfg["misc"]["device"],
        "seed": cfg["misc"]["seed"],
    }

    return CONFIG

def summarize_config(CONFIG):
    print("\n===== Training Configuration =====")

    print("\n[Data]")
    print(f"root_dir: {CONFIG['root_dir']}")
    print(f"max_spots: {CONFIG['max_spots']}")

    print("\n[Model]")
    print(f"embed_dim: {CONFIG['embed_dim']}")
    print(f"fusion_option: {CONFIG['fusion_option']}")
    print(f"top_k_genes: {CONFIG['top_k_genes']}")

    print("\n[Spatial Attention]")
    print(f"use_spatial_attn: {CONFIG['use_spatial_attn']}")
    if CONFIG["use_spatial_attn"]:
        print(f"k: {CONFIG['spatial_attn_k']}")
        print(f"heads: {CONFIG['spatial_attn_heads']}")
        print(f"dropout: {CONFIG['spatial_attn_dropout']}")

    print("\n[Training]")
    print(f"epochs: {CONFIG['epochs']}")
    print(f"lr: {CONFIG['lr']}")
    print(f"weight_decay: {CONFIG['weight_decay']}")

    print("\n[Memory]")
    print(f"batch_spots: {CONFIG['batch_spots']}")
    print(f"accum_steps: {CONFIG['accum_steps']}")
    print(f"freeze_image_encoder: {CONFIG['freeze_image_encoder']}")

    print("\n[Misc]")
    print(f"device: {CONFIG['device']}")
    print(f"seed: {CONFIG['seed']}")

    print("=================================\n")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_samples(root_dir, metadata_dir="./hest_data/metadata"):
    """
    Discover samples directly from root_dir instead of split txt files.
    Requires both:
      - st_preprocessed_global_hvg/{sid}.h5ad
      - patches/{sid}.h5
    """
    st_dir = Path(root_dir) / "st_preprocessed_global_hvg"
    patch_dir = Path(root_dir) / "patches"

    if not st_dir.exists():
        raise FileNotFoundError(f"ST dir not found: {st_dir}")
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch dir not found: {patch_dir}")

    sample_ids = []
    for fp in sorted(st_dir.glob("*.h5ad")):
        sid = fp.stem
        if (patch_dir / f"{sid}.h5").exists():
            sample_ids.append(sid)

    samples = []
    for sid in sample_ids:
        try:
            sample = CustomSample(root_dir, sid)
            if sample.label in [0, 1]:
                samples.append(sample)
        except Exception as e:
            print(f"Failed to load {sid}: {e}")

    if not samples:
        raise RuntimeError("No valid samples were discovered.")

    return samples


def split_samples(samples, val_ratio=0.2, seed=42):
    labels = [s.label for s in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_samples, val_samples


def load_global_hvg(hvg_path):
    hvg_path = Path(hvg_path)
    if not hvg_path.exists():
        raise FileNotFoundError(f"Global HVG file not found: {hvg_path}")

    with open(hvg_path) as f:
        genes = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(genes)} HVGs from {hvg_path}")
    return genes


def compute_class_weights(samples, device):
    counts = Counter([s.label for s in samples])
    n = len(samples)
    weights = torch.tensor(
        [
            n / (2 * counts[0]),
            n / (2 * counts[1]),
        ],
        dtype=torch.float,
        device=device,
    )
    return weights, counts


def forward_one_sample(model, batch, device, batch_spots, save_embeddings=False):
    """
    Forward one WSI sample through model.forward().
    """
    images = batch["images"]
    expr = batch["expr"]
    coords = batch["coords"]
    label = batch["label"].to(device)
    sample_id = batch.get("sample_id", "unknown")

    if images.dim() == 5 and images.size(0) == 1:
        images = images.squeeze(0)
    if expr.dim() == 3 and expr.size(0) == 1:
        expr = expr.squeeze(0)
    if coords.dim() == 3 and coords.size(0) == 1:
        coords = coords.squeeze(0)

    images = images.to(device, non_blocking=True)
    expr = expr.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)

    with autocast(enabled=(device.type == "cuda")):
        outputs = model(
            images,
            expr,
            coords,
            batch_spots=batch_spots,
            save_embeddings=save_embeddings,
        )

    out = {
        "logits": outputs["logits"],
        "label": label,
        "sample_id": sample_id,
        "wsi_embed": outputs["wsi_embed"],
        "mil_attn": outputs["attn_weights"],
        "spatial_attn_map": outputs["spatial_attn_map"],
    }

    if save_embeddings:
        out["coords"] = outputs["coords"].detach().cpu()

        if "img_embed" in outputs:
            out["img_embed"] = outputs["img_embed"].detach().cpu()
        if "sc_embed" in outputs:
            out["sc_embed"] = outputs["sc_embed"].detach().cpu()
        if "st_embed" in outputs:
            out["st_embed"] = outputs["st_embed"].detach().cpu()
        if "spot_fusion_embed" in outputs:
            out["spot_fusion_embed"] = outputs["spot_fusion_embed"].detach().cpu()
        if "spatial_embed" in outputs:
            out["spatial_embed"] = outputs["spatial_embed"].detach().cpu()

    return out

def train_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
    model.img_encoder.eval()

    epoch_loss = 0.0
    correct = 0
    n_samples = 0
    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(loader, desc="Training")

    for step, batch in enumerate(loop):
        outputs = forward_one_sample(
            model,
            batch,
            device=device,
            batch_spots=config["batch_spots"],
            save_embeddings=False,
        )

        logits = outputs["logits"]
        label = outputs["label"]

        with autocast(enabled=(device.type == "cuda")):
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss = loss / config["accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % config["accum_steps"] == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        pred = logits.argmax().item()
        epoch_loss += loss.item() * config["accum_steps"]
        correct += int(pred == label.item())
        n_samples += 1

        loop.set_postfix(
            loss=f"{epoch_loss / max(n_samples, 1):.4f}",
            acc=f"{100 * correct / max(n_samples, 1):.1f}%",
        )

    return epoch_loss / max(n_samples, 1), 100 * correct / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, config, device, save_embeddings=False):
    model.eval()

    total_loss = 0.0
    correct = 0
    y_true, y_pred, y_score = [], [], []

    collected = []

    for batch in tqdm(loader, desc="Validation"):
        outputs = forward_one_sample(
            model,
            batch,
            device=device,
            batch_spots=config["batch_spots"],
            save_embeddings=save_embeddings,
        )

        logits = outputs["logits"]
        label = outputs["label"]

        with autocast(enabled=(device.type == "cuda")):
          loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
        total_loss += loss.item()

        pred = logits.argmax().item()
        prob_pos = torch.softmax(logits, dim=0)[1].item()

        correct += int(pred == label.item())
        y_true.append(label.item())
        y_pred.append(pred)
        y_score.append(prob_pos)

        if save_embeddings:
            item = {
                "sample_id": outputs["sample_id"],
                "label": int(label.item()),
                "pred": int(pred),
                "score": float(prob_pos),
                "coords": outputs["coords"].cpu().numpy(),
                "img_embed": outputs["img_embed"].detach().cpu().float().numpy(),
                "sc_embed": outputs["sc_embed"].detach().cpu().float().numpy(),
                "st_embed": outputs["st_embed"].detach().cpu().float().numpy(),
                "spot_fusion_embed": outputs["spot_fusion_embed"].detach().cpu().float().numpy(),
                "mil_embed": outputs["wsi_embed"].detach().cpu().float().numpy(),
                "mil_attn": outputs["mil_attn"].detach().cpu().float().numpy(),
            }
            if outputs["spatial_embed"] is not None:
                item["spatial_embed"] = outputs["spatial_embed"].detach().cpu().float().numpy()
            if outputs["spatial_attn_map"] is not None:
                item["spatial_attn_map"] = outputs["spatial_attn_map"].detach().cpu().float().numpy()
            collected.append(item)

    val_loss = total_loss / max(len(loader), 1)
    val_acc = 100 * correct / max(len(loader), 1)

    try:
        val_auc = roc_auc_score(y_true, y_score)
    except Exception:
        val_auc = float("nan")

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_auc": float(val_auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }

    return metrics, collected


def is_better_epoch(curr_metrics, best_metrics):
    if best_metrics is None:
        return True

    if curr_metrics["val_acc"] == 100:
        return False

    if curr_metrics["val_acc"] > best_metrics["val_acc"]:
        return True
    if curr_metrics["val_acc"] < best_metrics["val_acc"]:
        return False

    curr_auc = curr_metrics["val_auc"]
    best_auc = best_metrics["val_auc"]

    if math.isnan(best_auc) and not math.isnan(curr_auc):
        return True
    if math.isnan(curr_auc):
        return False

    return curr_auc > best_auc


def save_confusion_matrix_png(cm, save_path, class_names=("Healthy", "Cancer")):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_best_embeddings(model, loader, device, config, save_dir):
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Saving embeddings"):
        outputs = forward_one_sample(
            model,
            batch,
            device=device,
            batch_spots=config["batch_spots"],
            save_embeddings=True,
        )

        sid = outputs["sample_id"]
        label = outputs["label"].item()

        logits = outputs["logits"]
        pred = logits.argmax().item()
        score = torch.softmax(logits, dim=0)[1].item()

        np.savez_compressed(
            save_dir / f"{sid}.npz",
            img_embed=outputs["img_embed"].cpu().numpy() if "img_embed" in outputs else None,
            sc_embed=outputs["sc_embed"].cpu().numpy() if "sc_embed" in outputs else None,
            st_embed=outputs["st_embed"].cpu().numpy() if "st_embed" in outputs else None,
            spot_fusion_embed=outputs["spot_fusion_embed"].cpu().numpy(),
            wsi_embed=outputs["wsi_embed"].cpu().numpy(),
            mil_attn=outputs["mil_attn"].cpu().numpy(),
            spatial_attn_map=outputs["spatial_attn_map"].cpu().numpy() if outputs["spatial_attn_map"] is not None else None,
            label=label,
            pred=pred,
            score=score,
            sample_id=sid,
        )


def main():
    CONFIG = load_config(CONFIG_PATH)

    summarize_config(CONFIG)

    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    
    output_dir = Path("./results_ver1") / f"fusion_{CONFIG['fusion_option']}_spatialattn_{CONFIG['use_spatial_attn']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_dir = output_dir / "embeddings" / "best_epoch"
    embed_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = discover_samples(CONFIG["root_dir"], CONFIG["metadata_dir"])
    train_samples, val_samples = split_samples(
        all_samples,
        val_ratio=0.2,
        seed=CONFIG["seed"],
    )
    print(f"Total: {len(all_samples)}")
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")

    print("\n[2] Load HVGs")
    hvg_genes = load_global_hvg(CONFIG["hvg_path"])

    print("\n[3] Build dataloaders")
    train_loader = create_wsi_dataloader(
        train_samples,
        batch_size=1,
        shuffle=True,
        max_spots=CONFIG["max_spots"],
    )
    val_loader = create_wsi_dataloader(
        val_samples,
        batch_size=1,
        shuffle=False,
        max_spots=CONFIG["max_spots"],
    )

    class_weights, label_counts = compute_class_weights(train_samples, device)
    print(f"\nLabel distribution (train): {label_counts}")
    print(f"Class weights: {class_weights.tolist()}")

    print("\n[4] Build model")
    model = MultiModalMILModel(
        num_genes=CONFIG["num_genes"],
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        fusion_option=CONFIG["fusion_option"],
        top_k_genes=CONFIG["top_k_genes"],
        use_spatial_attn=CONFIG["use_spatial_attn"],
        spatial_attn_k=CONFIG["spatial_attn_k"],
    ).to(device)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    best_metrics = None
    best_epoch = -1
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            CONFIG,
            device,
        )

        val_metrics, _ = validate(
            model,
            val_loader,
            criterion,
            CONFIG,
            device,
            save_embeddings=False,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["val_auc"].append(val_metrics["val_auc"])
        history["precision"].append(val_metrics["precision"])
        history["recall"].append(val_metrics["recall"])
        history["f1"].append(val_metrics["f1"])

        print(
            f"Train  Loss={train_loss:.4f}, Acc={train_acc:.2f}%\n"
            f"Val    Loss={val_metrics['val_loss']:.4f}, Acc={val_metrics['val_acc']:.2f}%, AUC={val_metrics['val_auc']:.4f}\n"
            f"P/R/F1 {val_metrics['precision']:.4f}/{val_metrics['recall']:.4f}/{val_metrics['f1']:.4f}"
        )

        if is_better_epoch(val_metrics, best_metrics):
            best_epoch = epoch + 1
            best_metrics = val_metrics

            torch.save(model.state_dict(), output_dir / "best_model.pt")

            best_metrics_with_embeds, _ = validate(
                model,
                val_loader,
                criterion,
                CONFIG,
                device,
                save_embeddings=False,
            )
            best_metrics = best_metrics_with_embeds

            save_best_embeddings(
                model=model,
                loader=val_loader,
                device=device,
                config=CONFIG,
                save_dir=embed_dir,
            )

            np.save(output_dir / "confusion_matrix.npy",
                    np.array(best_metrics["confusion_matrix"], dtype=np.int64))

            print(
                f"Updated best epoch -> {best_epoch} "
                f"(val_acc={best_metrics['val_acc']:.2f}, val_auc={best_metrics['val_auc']:.4f})"
            )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Training complete. Best epoch: {best_epoch}")
    print(f"Output saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()