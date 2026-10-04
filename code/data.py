"""data.py — Hoàn thiện cho Part 0.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz."""
    from pathlib import Path
    p = Path(processed_dir)
    if not (p / "train.npz").exists():
        candidates = [
            Path("../data/processed"),
            Path("../../data/processed"),
            Path("data/processed"),
            Path("/content/K4-Track4-Day1-Neuralnetwork/data/processed")
        ]
        for c in candidates:
            if (c / "train.npz").exists():
                p = c
                break
    train_data = np.load(p / "train.npz")
    eval_data = np.load(p / "eval.npz")

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"]

    # Kiểm tra kích thước và kiểu dữ liệu
    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train shape sai: {X_train_full.shape}"
    assert y_train_full.ndim == 1, f"y_train shape sai: {y_train_full.shape}"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval shape sai: {X_eval.shape}"
    assert y_eval.ndim == 1, f"y_eval shape sai: {y_eval.shape}"
    assert X_train_full.dtype == np.float32, "X_train phải là float32"
    assert y_train_full.dtype == np.int64, "y_train phải là int64"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    → Tránh rò rỉ thông tin (data leakage).
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    # Xử lý trường hợp std == 0 để tránh chia cho 0
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    # 1. Nạp và tách val
    X_full, y_full, X_eval_np, y_eval_np, eval_row_id = load_split(processed_dir)
    X_tr_np, y_tr_np, X_val_np, y_val_np = make_val_split(
        X_full, y_full, val_fraction=val_fraction, seed=seed
    )

    # 2. Chuẩn hoá: fit CHỈ trên X_tr
    mean, std = fit_standardizer(X_tr_np)
    X_tr_np = apply_standardizer(X_tr_np, mean, std)
    X_val_np = apply_standardizer(X_val_np, mean, std)
    X_eval_np = apply_standardizer(X_eval_np, mean, std)

    # 3. Đưa lên device dưới dạng tensor
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr_np, dtype=torch.int64, device=device)
    X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val_np, dtype=torch.int64, device=device)
    X_eval = torch.tensor(X_eval_np, dtype=torch.float32, device=device)
    y_eval = torch.tensor(y_eval_np, dtype=torch.int64, device=device)

    # 4. In thông tin kiểm tra
    majority_class = int(np.bincount(y_tr_np).argmax())
    majority_acc_val = float((y_val_np == majority_class).mean())

    print(f"[data] X_tr:   {tuple(X_tr.shape)}   y_tr:   {tuple(y_tr.shape)}")
    print(f"[data] X_val:  {tuple(X_val.shape)}  y_val:  {tuple(y_val.shape)}")
    print(f"[data] X_eval: {tuple(X_eval.shape)} y_eval: {tuple(y_eval.shape)}")
    print(f"[data] Lop da so tren train: {majority_class} -> accuracy doan da so tren val = {majority_acc_val:.4f}")

    return dict(
        X_tr=X_tr, y_tr=y_tr,
        X_val=X_val, y_val=y_val,
        X_eval=X_eval, y_eval=y_eval,
        eval_row_id=eval_row_id,
        mean=mean, std=std,
    )


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; giữ nguyên không bỏ qua.
    """
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]
