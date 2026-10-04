"""train.py — Hoàn thiện cho Part 2 và toàn bộ lab.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` chọn hợp lý theo val (0.05).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base, seed 1",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn quanh 0.01 - 0.1
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Đúng định nghĩa trong scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str, reduction: str = "mean") -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.size(-1)).float()
        return F.mse_loss(logits, y_onehot, reduction=reduction)
    else:
        raise ValueError(f"Hàm mất mát không hỗ trợ: {loss_name}")


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    was_training = model.training
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    if was_training:
        model.train()
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    all_preds = []
    n = len(X)

    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        batch_loss = compute_loss(logits, yb, loss_name, reduction="sum")
        total_loss += float(batch_loss.item())
        all_preds.append(torch.argmax(logits, dim=1))

    preds = torch.cat(all_preds, dim=0)
    acc = float((preds == y).float().mean().item())

    # Ma trận nhầm lẫn 7x7
    cm = np.zeros((7, 7), dtype=np.int64)
    y_np = y.cpu().numpy()
    pred_np = preds.cpu().numpy()
    np.add.at(cm, (y_np, pred_np), 1)

    macro_f1 = macro_f1_from_confusion(cm)
    avg_loss = total_loss / n

    if was_training:
        model.train()

    return dict(loss=float(avg_loss), acc=float(acc), macro_f1=float(macro_f1))


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất}
    """
    # 0. Thiết lập seed và khởi tạo model
    seed = cfg.get("seed", 1)
    set_seed(seed)

    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(
        hidden=hidden,
        dropout=float(cfg.get("dropout", 0.0)),
        init=str(cfg.get("init", "he"))
    )
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], (
            f"Số tham số không khớp: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"
        )

    device = data["X_tr"].device
    model = model.to(device)

    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = cfg.get("precision", "fp32")
    use_amp = (precision in ("fp16", "bf16")) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if (precision == "fp16" and device.type == "cuda") else None

    # Tập con cố định của train để đánh giá train_loss ở eval mode (50 000 mẫu như gợi ý GUIDE)
    subset_n = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:subset_n]
    y_tr_eval = data["y_tr"][:subset_n]

    # 1. Đo loss bước 0 trên tập val TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7)
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = float(step0_res["loss"])

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)
    if clip_norm is not None:
        clip_norm = float(clip_norm)

    # 2. Vòng lặp huấn luyện theo epoch
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_gn: list[float] = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"[CẢNH BÁO] Loss bị NaN/inf tại epoch {epoch}! Dừng sớm.")
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), clip_norm)
                else:
                    gn = clip_gradients(model.parameters(), None)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            epoch_gn.append(gn)

        if diverged:
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0

        # Đánh giá cuối epoch ở chế độ eval
        tr_res = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
        mean_gn = float(np.mean(epoch_gn)) if epoch_gn else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(float(tr_res["loss"]))
        history["val_loss"].append(float(val_res["loss"]))
        history["val_acc"].append(float(val_res["acc"]))
        history["val_macro_f1"].append(float(val_res["macro_f1"]))
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(float(ep_time))

        # Lưu best state theo val_loss
        if val_res["loss"] < best_val_loss:
            best_val_loss = float(val_res["loss"])
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 3. Tổng hợp summary
    peak_mem = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if device.type == "cuda" else 0.0

    if diverged or len(history["epoch"]) == 0:
        summary = {
            "step0_loss": step0_loss,
            "best_val_loss": None,
            "best_epoch": None,
            "final_train_loss": None,
            "final_val_loss": None,
            "val_acc": None,
            "val_macro_f1": None,
            "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
            "peak_mem_MB": peak_mem,
            "diverged": True,
        }
    else:
        best_idx = best_epoch - 1
        summary = {
            "step0_loss": step0_loss,
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "val_acc": float(history["val_acc"][best_idx]),
            "val_macro_f1": float(history["val_macro_f1"][best_idx]),
            "time_per_epoch_s": float(np.mean(history["epoch_time_s"])),
            "peak_mem_MB": peak_mem,
            "diverged": False,
        }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "row_id": np.asarray(row_id, dtype=np.int64),
        "pred": np.asarray(preds, dtype=np.int64),
    })
    df.to_csv(path, index=False)
    print(f"[train] Đã ghi file dự đoán: {path} ({len(df)} dòng)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               out_json: str | None = None,
               data_path: str = "data/covtype.csv.gz",
               meta_path: str = "data/split_metadata.csv") -> dict:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và trả về kết quả dict
    """
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(
        hidden=hidden,
        dropout=float(cfg.get("dropout", 0.0)),
        init=str(cfg.get("init", "he"))
    )
    if result.get("best_state") is not None:
        model.load_state_dict(result["best_state"])
    model = model.to(device)

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()

    write_predictions(data["eval_row_id"], preds_np, pred_path)

    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--pred", str(pred_path),
        "--data", str(data_path),
        "--meta", str(meta_path)
    ]
    if out_json:
        cmd.extend(["--out", str(out_json)])

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(res.stdout)

    # Đọc lại JSON kết quả nếu đã ghi
    if out_json and Path(out_json).exists():
        import json
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f)

    # Nếu không thì tính trực tiếp
    eval_res = evaluate(model, data["X_eval"], data["y_eval"], loss_name=cfg.get("loss", "ce"))
    return {"accuracy": eval_res["acc"], "macro_f1": eval_res["macro_f1"]}
