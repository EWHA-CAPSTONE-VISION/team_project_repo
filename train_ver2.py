import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from dataset.loader import CustomSample, create_wsi_dataloader
from models.model_ver2 import MultiModalMILModel

CONFIG_PATH = r"C:\Users\rdh08\Desktop\Capstone\configs\train.yaml"

# ======================================================
# Utils
# ======================================================
def load_config(path=CONFIG_PATH):
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG = {
        # data
        "root_dir": cfg["data"]["root_dir"],
        "max_spots": cfg["data"]["max_spots"],
        "metadata_dir": cfg["data"]["metadata_dir"],
        "output_dir": cfg["data"]["output_dir"],

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def discover_samples(root_dir):
    samples = []
    st_dir = os.path.join(root_dir, "st_preprocessed_global_hvg")

    for fn in os.listdir(st_dir):
        if fn.endswith(".h5ad"):
            sid = fn[:-5]
            try:
                s = CustomSample(root_dir, sid)
                if s.label in [0, 1]:
                    samples.append(s)
            except:
                continue
    return samples


def split_dataset(samples, val_ratio=0.2):
    np.random.shuffle(samples)
    n = int(len(samples) * val_ratio)
    return samples[n:], samples[:n]

def compress_spatial_feature(outputs_cpu, use_after=False, attn_reduce="mean"):
    """
    spatial_attn과 spot embedding을 이용해
    (N, N) -> (N, D) 형태의 compressed spatial feature를 생성.

    Args:
        outputs_cpu: CPU로 옮겨진 model output dict
        use_after: True면 spot_embeds_after_spatial 사용, False면 before 사용
        attn_reduce: multi-head attention일 경우 "mean" 또는 "first"

    Returns:
        spatial_feature: torch.Tensor of shape (N, D)
    """
    attn = outputs_cpu.get("spatial_attn", None)
    if attn is None:
        return None

    if use_after:
        spot = outputs_cpu["spot_embeds_after_spatial"]   # (N, D)
    else:
        spot = outputs_cpu["spot_embeds_before_spatial"]  # (N, D)

    # ---- attention shape 정리 ----
    # case 1) (N, N)
    if attn.dim() == 2:
        attn_mat = attn

    # case 2) (H, N, N)
    elif attn.dim() == 3:
        if attn_reduce == "mean":
            attn_mat = attn.mean(dim=0)   # (N, N)
        elif attn_reduce == "first":
            attn_mat = attn[0]            # (N, N)
        else:
            raise ValueError(f"Unsupported attn_reduce: {attn_reduce}")

    # case 3) (1, H, N, N) 또는 (B, H, N, N)
    elif attn.dim() == 4:
        # batch_size=1 전제
        attn = attn[0]  # (H, N, N)
        if attn_reduce == "mean":
            attn_mat = attn.mean(dim=0)
        elif attn_reduce == "first":
            attn_mat = attn[0]
        else:
            raise ValueError(f"Unsupported attn_reduce: {attn_reduce}")
    else:
        raise ValueError(f"Unexpected spatial_attn shape: {attn.shape}")

    # (N, N) @ (N, D) -> (N, D)
    spatial_feature = attn_mat @ spot
    return spatial_feature

# ======================================================
# Embedding 저장
# ======================================================
def save_embeddings_per_sample(outputs, sample_id, label, pred, score, save_dir,
                               spatial_feature=None, save_full_spatial_attn=True):
    save_path = save_dir / f"{sample_id}.npz"

    save_dict = {
        "img_embed": outputs["img_embed"].numpy() if outputs["img_embed"] is not None else None,
        "st_embed": outputs["st_embed"].numpy() if outputs["st_embed"] is not None else None,
        "spot_before": outputs["spot_embeds_before_spatial"].numpy(),
        "spot_after": outputs["spot_embeds_after_spatial"].numpy(),
        "wsi_embed": outputs["wsi_embed"].numpy(),
        "mil_attn": outputs["mil_attn"].numpy(),
        "label": label,
        "pred": pred,
        "score": score,
        "sample_id": sample_id,
    }

    if spatial_feature is not None:
        save_dict["spatial_embed"] = spatial_feature.numpy()

    if save_full_spatial_attn:
        save_dict["spatial_attn_map"] = (
            outputs["spatial_attn_map"].numpy()
            if outputs["spatial_attn_map"] is not None else None
        )
    
    np.savez(save_path, **save_dict)

# ======================================================
# Train / Validate
# ======================================================
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0
    correct = 0

    y_true, y_score, y_pred = [], [], []

    for batch in tqdm(loader, desc="Train" if train else "Val"):
        images = batch["images"]
        expr   = batch["expr"]
        coords = batch["coords"]

        # batch dim 제거
        if images.dim() == 5:
            images = images.squeeze(0)
        if expr.dim() == 3:
            expr = expr.squeeze(0)
        if coords.dim() == 3:
            coords = coords.squeeze(0)

        images = images.to(device)
        expr   = expr.to(device)
        coords = coords.to(device)

        label  = batch["label"].to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images, expr, coords)

        logits = outputs["logits"]
        loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax().item()
        correct += int(pred == label.item())

        prob = torch.softmax(logits, dim=0)[1].item()

        y_true.append(label.item())
        y_score.append(prob)
        y_pred.append(pred)

    acc = 100 * correct / len(loader)

    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = float("nan")

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return (
        total_loss / len(loader),
        acc,
        auc,
        float(p),
        float(r),
        float(f1),
        cm
    )


# ======================================================
# Best epoch embedding 저장
# ======================================================
@torch.no_grad()
def save_best_embeddings(model, loader, device, save_dir,
    use_after_for_compress=False,
    attn_reduce="mean",
    save_full_spatial_attn=True
    ):
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Saving embeddings"):
        images = batch["images"]
        expr   = batch["expr"]
        coords = batch["coords"]

        # batch dim 제거
        if images.dim() == 5:
            images = images.squeeze(0)
        if expr.dim() == 3:
            expr = expr.squeeze(0)
        if coords.dim() == 3:
            coords = coords.squeeze(0)

        images = images.to(device)
        expr   = expr.to(device)
        coords = coords.to(device)

        label  = batch["label"].item()
        sid    = batch["sample_id"]

        outputs = model(images, expr, coords)
        logits = outputs["logits"]

        pred = logits.argmax().item()
        score = torch.softmax(logits, dim=0)[1].item()

        outputs_cpu = {}
        for k, v in outputs.items():
            if torch.is_tensor(v):
                outputs_cpu[k] = v.detach().cpu()
            else:
                outputs_cpu[k] = v

        del outputs, logits, images, expr, coords
        torch.cuda.empty_cache()

        spatial_feature = compress_spatial_feature(
            outputs_cpu,
            use_after=use_after_for_compress,
            attn_reduce=attn_reduce
        )

        save_embeddings_per_sample(
            outputs=outputs_cpu,
            sample_id=sid,
            label=label,
            pred=pred,
            score=score,
            save_dir=save_dir,
            spatial_feature=spatial_feature,
            save_full_spatial_attn=save_full_spatial_attn
        )

        del outputs_cpu, spatial_feature


# ======================================================
# Main
# ======================================================
def main():

    CONFIG = load_config(CONFIG_PATH)

    summarize_config(CONFIG)

    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    # dataset
    samples = discover_samples(CONFIG["root_dir"])
    train_samples, val_samples = split_dataset(samples)

    train_loader = create_wsi_dataloader(
        train_samples, batch_size=1, shuffle=True, max_spots=CONFIG["max_spots"]
    )
    val_loader = create_wsi_dataloader(
        val_samples, batch_size=1, shuffle=False, max_spots=CONFIG["max_spots"]
    )

    # model
    model = MultiModalMILModel(
        num_genes=CONFIG["num_genes"],
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        fusion_option=CONFIG["fusion_option"],
        top_k_genes=CONFIG["top_k_genes"],

        use_image=True,
        use_st=True,

        # spatial attn
        use_spatial_attn=CONFIG["use_spatial_attn"],
        spatial_attn_k=CONFIG["spatial_attn_k"],
        spatial_attn_heads=CONFIG["spatial_attn_heads"],
        spatial_attn_dropout=CONFIG["spatial_attn_dropout"],

        freeze_image_encoder=CONFIG["freeze_image_encoder"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # dirs
    output_dir = Path("./results_ver2") / f"fusion_{CONFIG['fusion_option']}_spatialattn_{CONFIG['use_spatial_attn']}"
    emb_dir = output_dir / "embeddings" / "best_epoch"

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0
    best_auc = 0

    history = []

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}")

        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        
        torch.cuda.empty_cache()

        val_metrics   = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        _, val_acc, val_auc, val_p, val_r, val_f1, cm = val_metrics

        history.append({
            "epoch": epoch,
            "val_acc": val_acc,
            "val_auc": val_auc
        })

        print(f"Val Acc: {val_acc:.2f} | AUC: {val_auc:.4f}")

        # best 기준 (acc > auc)
        is_best = False
        if val_acc == 100:
            print("Perfect accuracy achieved ... ignoring the current epoch")
            is_best = False
        else:
            if val_acc > best_acc:
                is_best = True
            elif val_acc == best_acc and val_auc > best_auc:
                is_best = True

        if is_best:
            torch.cuda.empty_cache()

            best_acc = val_acc
            best_auc = val_auc

            torch.save(model.state_dict(), output_dir / "best_model.pt")

            save_best_embeddings(
                model=model,
                loader=val_loader,
                device=device,
                save_dir=emb_dir,
                use_after_for_compress=False,
                attn_reduce="mean",
                save_full_spatial_attn=True
            )
            np.save(output_dir / "confusion_matrix.npy", cm)

    # save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()