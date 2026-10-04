"""plots.py — Hoàn thiện vẽ biểu đồ thí nghiệm và so sánh.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Đánh dấu best_epoch bằng đường thẳng đứng.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    cfg = result.get("cfg", {})
    history = result.get("history", {})
    summary = result.get("summary", {})

    exp_id = cfg.get("exp_id", "experiment")
    opt_name = cfg.get("optimizer", "opt")
    lr = cfg.get("lr", "auto")
    batch = cfg.get("batch", "batch")
    epochs_cfg = cfg.get("epochs", len(history.get("epoch", [])))
    best_epoch = summary.get("best_epoch", None)

    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Exp: {exp_id} | Opt: {opt_name}, lr: {lr}, Batch: {batch}, Epochs: {epochs_cfg}",
                 fontsize=13, fontweight="bold")

    # (1) Loss
    ax = axes[0]
    if "train_loss" in history and history["train_loss"]:
        ax.plot(epochs, history["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_title("Train & Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # (2) Metrics (Val Acc & Val Macro-F1)
    ax = axes[1]
    if "val_acc" in history and history["val_acc"]:
        ax.plot(epochs, history["val_acc"], label="Val Accuracy", color="#2ca02c", lw=2)
    if "val_macro_f1" in history and history["val_macro_f1"]:
        ax.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="#9467bd", lw=2)
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_title("Validation Accuracy & Macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # (3) Grad Norm
    ax = axes[2]
    if "grad_norm" in history and history["grad_norm"]:
        ax.plot(epochs, history["grad_norm"], label="Grad Norm (pre-clip)", color="#d62728", lw=2)
        if cfg.get("clip_norm") is not None:
            ax.axhline(cfg["clip_norm"], color="black", linestyle=":", label=f"Clip norm ({cfg['clip_norm']})")
    ax.set_title("Global Gradient Norm (Pre-clip)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm L2")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        cfg = res.get("cfg", {})
        exp_id = cfg.get("exp_id", "run")
        history = res.get("history", {})
        if metric in history and history[metric]:
            epochs = history.get("epoch", list(range(1, len(history[metric]) + 1)))
            ax.plot(epochs, history[metric], label=exp_id, lw=2)

    ax.set_title(title if title else f"Comparison of {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
