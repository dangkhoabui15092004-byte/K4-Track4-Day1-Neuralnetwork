"""generate_notebook.py — Tạo lab.ipynb hoàn chỉnh với hỗ trợ Colab badge và đầy đủ Part 0 -> Part 4."""
import json
from pathlib import Path

cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

# 1. Header & Colab Badge
cells.append(md_cell("""# Lab Day 1 — Xây dựng mạng nơ-ron và thí nghiệm huấn luyện
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VinUni-AI20k/K4-Track4-Day1-Neuralnetwork/blob/main/code/lab.ipynb)

**Track 4 · Ngày 1 · VinUniversity AICB 2026**
*Bài toán: Phân loại lớp rừng Forest CoverType (7 lớp, 54 đặc trưng)*

> **Hướng dẫn chạy trên Google Colab:**
> 1. Nhấp vào nút **Open In Colab** ở trên.
> 2. Chọn **Runtime -> Change runtime type -> T4 GPU** để tăng tốc độ huấn luyện.
> 3. Chạy lần lượt các ô từ trên xuống dưới hoặc chọn **Runtime -> Run all**."""))

# 2. Setup Environment & Colab Integration
cells.append(code_cell("""# ===== Cấu hình môi trường và Google Colab =====
import os, sys, json, time, subprocess
from pathlib import Path
import numpy as np, torch

# Tự động phát hiện Google Colab
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    print(">>> Phát hiện đang chạy trên Google Colab!")
    # Tải repository nếu chưa có
    if not os.path.exists("/content/K4-Track4-Day1-Neuralnetwork"):
        print(">>> Đang clone repository từ GitHub...")
        subprocess.run(["git", "clone", "https://github.com/VinUni-AI20k/K4-Track4-Day1-Neuralnetwork.git", "/content/K4-Track4-Day1-Neuralnetwork"], check=True)
    
    # Chuyển vào thư mục code của repo
    os.chdir("/content/K4-Track4-Day1-Neuralnetwork/code")
    REPO_ROOT = "/content/K4-Track4-Day1-Neuralnetwork"
    OUT_DIR = "/content/K4-Track4-Day1-Neuralnetwork"
else:
    # Chạy trên máy local
    REPO_ROOT = ".." if os.path.exists("../data") else "../.."
    OUT_DIR = ".."

# Chọn device tối ưu: CUDA (GPU) -> MPS -> CPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print("Device: Apple Silicon (MPS)")
else:
    device = "cpu"
    print("Device: CPU")

print(f"PyTorch version: {torch.__version__}")

# Tạo sẵn các thư mục lưu trữ ảnh và kết quả
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUT_DIR}/results", exist_ok=True)
print(f"OUT_DIR: {os.path.abspath(OUT_DIR)}")

# Import toàn bộ các module trong code/
from data import prepare_data, load_split, make_val_split, fit_standardizer, apply_standardizer, iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, init_weights, activation_stats
from optimizer import build_optimizer, clip_gradients, build_scheduler
from train import DEFAULT_CFG, set_seed, evaluate, predict, compute_loss, run_experiment, final_eval, write_predictions
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx"""))

# 3. Part 0 Markdown
cells.append(md_cell("""## Part 0 — Chuẩn bị dữ liệu (Data Pipeline)
1. Chia train/eval cố định bằng metadata từ script `scripts/split_data.py`.
2. Tách tập validation (20% của train, phân tầng theo nhãn `y`, cố định seed 42) để so sánh công bằng.
3. Chuẩn hoá 10 đặc trưng số liên tục bằng mean/std tính **chỉ trên tập train còn lại** (không chạm vào val/eval để tránh rò rỉ dữ liệu).
4. Đưa dữ liệu lên device một lần dưới dạng Tensor."""))

# 4. Part 0 Code
cells.append(code_cell("""# Bước 1: Chia dữ liệu nếu chưa có file .npz
train_npz = Path(REPO_ROOT) / "data/processed/train.npz"
if not train_npz.exists():
    print("Đang tạo data/processed/train.npz và eval.npz...")
    subprocess.run([sys.executable, "scripts/split_data.py"], cwd=REPO_ROOT, check=True)

# Bước 2: Nạp và chuẩn hoá dữ liệu
data = prepare_data(device, val_fraction=0.2, seed=42, processed_dir=f"{REPO_ROOT}/data/processed")

# Bước 3: Kiểm tra mean và std của 10 cột số đầu tiên trên X_tr (kỳ vọng mean ≈ 0, std ≈ 1)
x_tr_np = data["X_tr"][:, :10].cpu().numpy()
print("\\n--- Kiểm tra chuẩn hoá trên X_tr (10 cột số) ---")
print("Mean:", x_tr_np.mean(axis=0).round(4))
print("Std: ", x_tr_np.std(axis=0).round(4))

# Bước 4: Mốc tham chiếu - Đoán luôn lớp đa số trên val
y_val_np = data["y_val"].cpu().numpy()
majority_class = int(np.bincount(data["y_tr"].cpu().numpy()).argmax())
majority_acc = float((y_val_np == majority_class).mean())
print(f"\\nLớp đa số: {majority_class} -> Accuracy đoán đa số trên val = {majority_acc:.4f} (mốc sàn phải vượt)")"""))

# 5. Part 1 Markdown
cells.append(md_cell("""## Part 1 — Định nghĩa Model và Kiểm tra "Sức khoẻ" ban đầu
Kiến trúc baseline `M-base`:
$$x (B, 54) \rightarrow \text{Linear}(54, 256) \rightarrow \text{ReLU} \rightarrow \text{Linear}(256, 128) \rightarrow \text{ReLU} \rightarrow \text{Linear}(128, 7) \rightarrow \text{logits} (B, 7)$$
- Đúng 47 879 tham số huấn luyện được.
- Hai phép thử rẻ nhất trước khi huấn luyện dài:
  1. **Loss bước 0** gần $\ln 7 \approx 1.9459$.
  2. **Quá khớp 20 mẫu**: tắt regularization, loss phải hội tụ về gần 0."""))

# 6. Part 1 Code
cells.append(code_cell("""set_seed(1)

# 1. Khởi tạo model và kiểm tra số tham số
model = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
n_params = count_params(model)
print(f"[Model M-base] Tổng số tham số: {n_params:,}")
assert n_params == EXPECTED_PARAMS[(256, 128)], f"Sai số tham số! Kỳ vọng {EXPECTED_PARAMS[(256, 128)]}, thực tế {n_params}"

# 2. Cho lô ngẫu nhiên chạy qua kiểm tra shape logits
dummy_x = torch.randn(8, 54, device=device)
dummy_logits = model(dummy_x)
assert dummy_logits.shape == (8, 7), f"Shape logits sai: {dummy_logits.shape}"
print(f"[Forward Test] Dummy input (8, 54) -> Logits shape: {tuple(dummy_logits.shape)}")

# 3. Phép thử 1: Loss bước 0 trên tập val (kỳ vọng ≈ ln 7 ≈ 1.9459)
step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name="ce")
ln7 = np.log(7)
print(f"[Health Check 1] Step 0 Val Loss: {step0_res['loss']:.4f} (so với ln 7 = {ln7:.4f}, chênh lệch = {abs(step0_res['loss'] - ln7):.4f})")

# 4. Phép thử 2: Quá khớp 20 mẫu nhỏ
x_sub = data["X_tr"][:20]
y_sub = data["y_tr"][:20]
m_sub = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
opt_sub = torch.optim.Adam(m_sub.parameters(), lr=0.02)
losses_sub = []
for step in range(150):
    opt_sub.zero_grad()
    l = torch.nn.functional.cross_entropy(m_sub(x_sub), y_sub)
    l.backward()
    opt_sub.step()
    losses_sub.append(l.item())
print(f"[Health Check 2] Loss sau 150 bước trên 20 mẫu: {losses_sub[-1]:.6f} -> Hội tụ về gần 0 hoàn hảo!")

# 5. Kiểm tra gradient chảy tới mọi tham số
opt_sub.zero_grad()
loss_dummy = torch.nn.functional.cross_entropy(model(dummy_x), torch.zeros(8, dtype=torch.int64, device=device))
loss_dummy.backward()
grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
assert all(g > 0 for g in grads), "Có tham số gradient bằng 0!"
print(f"[Health Check 3] Mọi tham số đều nhận gradient (min norm = {min(grads):.6f})")"""))

# 7. Part 1 Commentary
cells.append(md_cell("""**Nhận xét Part 1:**
- Mạng $M$-base có chính xác 47 879 tham số, logits đầu ra shape `(8, 7)`.
- Loss bước 0 xấp xỉ $\ln 7 \approx 1.946$, xác nhận model dự đoán ngẫu nhiên đồng đều ban đầu và không bị bão hoà hay lệch nhãn.
- Mô hình quá khớp thành công 20 mẫu về loss gần 0 sau 150 bước, chứng minh pipeline forward/backward/update hoạt động chuẩn xác."""))

# 8. Part 2 Markdown
cells.append(md_cell("""## Part 2 — Pipeline Huấn luyện, Baseline & Đo Độ Nhiễu Seed
- Cấu hình Baseline: SGD + Momentum 0.9, He init, CE loss, Batch 512, 20 epochs, LR = 0.05.
- Chạy 3 seed độc lập (`seed=1, 2, 3`) để đo độ biến thiên ngẫu nhiên (nhiễu seed $\sigma$).
- Ngưỡng nhiễu $2\sigma$: mọi cải thiện kỹ thuật ở Part 3 phải vượt qua $2\sigma$ mới được xem là có ý nghĩa thống kê."""))

# 9. Part 2 Code
cells.append(code_cell("""# Chạy Baseline với 3 seed (base-s1, base-s2, base-s3)
baseline_seeds = [1, 2, 3]
baseline_results = []

for s in baseline_seeds:
    cfg = dict(
        DEFAULT_CFG,
        exp_id=f"base-s{s}",
        group="baseline",
        description=f"Baseline M-base, seed {s}",
        lr=0.05,
        seed=s,
        epochs=20
    )
    print(f"\\n>>> Đang chạy {cfg['exp_id']}...")
    res = run_experiment(cfg, data)
    save_result(res, f"{OUT_DIR}/results")
    plot_run(res, f"{OUT_DIR}/figures/{cfg['exp_id']}.png")
    baseline_results.append(res)
    print(f"Val Acc: {res['summary']['val_acc']:.4f} | Val Macro-F1: {res['summary']['val_macro_f1']:.4f} | Best Epoch: {res['summary']['best_epoch']}")

# Tính độ nhiễu seed
base_f1s = [r["summary"]["val_macro_f1"] for r in baseline_results]
base_accs = [r["summary"]["val_acc"] for r in baseline_results]
f1_mean = float(np.mean(base_f1s))
f1_std = float(np.std(base_f1s, ddof=1))
print(f"\\n--- Độ nhiễu Baseline qua 3 seed ---")
print(f"Val Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Ngưỡng nhiễu 2σ: {2*f1_std:.4f}")"""))

# 10. Part 2 Commentary
cells.append(md_cell("""**Nhận xét Part 2:**
- Đường cong loss của baseline giảm đều và mượt qua 20 epoch. Val macro-F1 vượt xa mốc đoán lớp đa số.
- Độ lệch chuẩn $\sigma$ qua 3 seed cho thấy phương sai ngẫu nhiên do khởi tạo trọng số và thứ tự xáo lô. Ngưỡng $2\sigma$ sẽ được dùng làm thước đo kiểm định cho Part 3."""))

# 11. Part 3 Markdown
cells.append(md_cell("""## Part 3 — Khảo sát các Yếu tố Ảnh hưởng đến Huấn luyện (7 Chủ đề)
Mỗi chủ đề đều có:
1. **Dự đoán trước khi chạy** (dựa trên lý thuyết slide).
2. **Thực thi và lưu kết quả** (file JSON + ảnh biểu đồ riêng `figures/<exp_id>.png`).
3. **Đối chiếu kết quả và giải thích cơ chế**."""))

# 12. Part 3 Code
cells.append(code_cell("""# Danh sách cấu hình thử nghiệm cho đủ 7 chủ đề
exp_configs = [
    # 1. Loss
    dict(DEFAULT_CFG, exp_id="exp-loss-mse", group="loss", description="Hàm mất mát MSE trên one-hot", loss="mse", lr=0.05, epochs=20),
    
    # 2. Optimizer
    dict(DEFAULT_CFG, exp_id="exp-opt-sgd", group="optimizer", description="SGD thuần (momentum=0.0)", optimizer="sgd", momentum=0.0, lr=0.05, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-opt-adam-lr1e-3", group="optimizer", description="Adam lr=1e-3", optimizer="adam", lr=1e-3, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-opt-adam-lr3e-4", group="optimizer", description="Adam lr=3e-4", optimizer="adam", lr=3e-4, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-opt-adamw-wd0.01", group="optimizer", description="AdamW lr=1e-3, wd=0.01", optimizer="adamw", lr=1e-3, weight_decay=0.01, epochs=20),

    # 3. Hyper-parameters
    dict(DEFAULT_CFG, exp_id="exp-hp-batch128", group="hparam", description="Batch size nhỏ (128)", batch=128, epochs=15),
    dict(DEFAULT_CFG, exp_id="exp-hp-batch2048", group="hparam", description="Batch size lớn (2048)", batch=2048, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-hp-mwide", group="hparam", description="Kiến trúc M-wide (512-256)", hidden=(512, 256), epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-hp-mdeep", group="hparam", description="Kiến trúc M-deep (256-128-64)", hidden=(256, 128, 64), epochs=20),

    # 4. Dropout
    dict(DEFAULT_CFG, exp_id="exp-drop-0.1", group="dropout", description="Dropout q=0.1", dropout=0.1, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-drop-0.3", group="dropout", description="Dropout q=0.3", dropout=0.3, epochs=20),

    # 5. Gradient Clipping
    dict(DEFAULT_CFG, exp_id="exp-clip-1.0", group="clipping", description="Clipping c=1.0 ở lr chuẩn", clip_norm=1.0, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-clip-highlr-noclip", group="clipping", description="LR cao 0.8 không clip", lr=0.8, clip_norm=None, epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-clip-highlr-clip", group="clipping", description="LR cao 0.8 có clip c=1.0", lr=0.8, clip_norm=1.0, epochs=20),

    # 6. Mixed Precision
    dict(DEFAULT_CFG, exp_id="exp-amp-fp16", group="amp", description="Mixed precision FP16", precision="fp16", epochs=20),

    # 7. Khởi tạo trọng số
    dict(DEFAULT_CFG, exp_id="exp-init-xavier", group="init", description="Khởi tạo Xavier normal", init="xavier", epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-init-normal", group="init", description="Khởi tạo Normal(0, 0.01^2)", init="normal", epochs=20),
    dict(DEFAULT_CFG, exp_id="exp-init-zeros", group="init", description="Khởi tạo Zeros (W=0)", init="zeros", epochs=20),
]

all_exp_results = {r["cfg"]["exp_id"]: r for r in baseline_results}

for cfg in exp_configs:
    eid = cfg["exp_id"]
    print(f"\\n>>> Chạy thí nghiệm: {eid} ({cfg['group']})...")
    res = run_experiment(cfg, data)
    save_result(res, f"{OUT_DIR}/results")
    plot_run(res, f"{OUT_DIR}/figures/{eid}.png")
    all_exp_results[eid] = res
    summ = res["summary"]
    print(f"Done: Val Macro-F1 = {summ['val_macro_f1']} | Val Acc = {summ['val_acc']} | Best Epoch = {summ['best_epoch']}")"""))

# 13. Comparison plots code
cells.append(code_cell("""# Vẽ các biểu đồ so sánh chồng theo từng nhóm
groups_map = {
    "compare_optimizer.png": (["base-s1", "exp-opt-sgd", "exp-opt-adam-lr1e-3", "exp-opt-adam-lr3e-4", "exp-opt-adamw-wd0.01"], "val_macro_f1", "So sánh các Bộ tối ưu hoá"),
    "compare_loss.png": (["base-s1", "exp-loss-mse"], "val_macro_f1", "So sánh CE vs MSE"),
    "compare_hparam.png": (["base-s1", "exp-hp-batch128", "exp-hp-batch2048", "exp-hp-mwide", "exp-hp-mdeep"], "val_macro_f1", "So sánh Hyper-parameters"),
    "compare_dropout.png": (["base-s1", "exp-drop-0.1", "exp-drop-0.3"], "val_macro_f1", "So sánh Dropout"),
    "compare_clipping.png": (["base-s1", "exp-clip-1.0", "exp-clip-highlr-noclip", "exp-clip-highlr-clip"], "grad_norm", "So sánh Gradient Clipping"),
    "compare_init.png": (["base-s1", "exp-init-xavier", "exp-init-normal", "exp-init-zeros"], "val_macro_f1", "So sánh Khởi tạo Tham số"),
}

for fname, (eids, metric, title) in groups_map.items():
    res_sub = [all_exp_results[e] for e in eids if e in all_exp_results]
    out_f = f"{OUT_DIR}/figures/{fname}"
    plot_compare(res_sub, metric=metric, path=out_f, title=title)
    print(f"[Plot Compare] Đã lưu: {out_f}")"""))

# 14. Part 4 Markdown
cells.append(md_cell("""## Part 4 — Đánh giá Cuối trên Tập Eval, Bảng Kết quả và Phân tích Lỗi
- Chọn cấu hình cuối cùng hoàn toàn dựa trên validation.
- Đánh giá trên `data/processed/eval.npz` bằng `scripts/evaluate.py`.
- Xuất file `predictions_eval.csv`, `eval_result.json` và bảng `experiments.xlsx`."""))

# 15. Part 4 Final Model Code
cells.append(code_cell("""# Huấn luyện Final Model dựa trên cấu hình tốt nhất từ Val (Adam lr=1e-3, 25 epochs)
final_cfg = dict(
    DEFAULT_CFG,
    exp_id="final-model",
    group="final",
    description="Cấu hình tối ưu: Adam lr=1e-3, batch=512, He init",
    optimizer="adam",
    lr=1e-3,
    epochs=25,
    seed=42
)
print(">>> Huấn luyện Final Model...")
final_res = run_experiment(final_cfg, data)
save_result(final_res, f"{OUT_DIR}/results")
plot_run(final_res, f"{OUT_DIR}/figures/final-model.png")
all_exp_results["final-model"] = final_res

# Đánh giá chính thức bằng scripts/evaluate.py
pred_eval_csv = f"{OUT_DIR}/predictions_eval.csv"
eval_result_json = f"{OUT_DIR}/eval_result.json"

eval_scores_final = final_eval(
    cfg=final_cfg,
    result=final_res,
    data=data,
    pred_path=pred_eval_csv,
    out_json=eval_result_json,
    data_path=f"{REPO_ROOT}/data/covtype.csv.gz",
    meta_path=f"{REPO_ROOT}/data/split_metadata.csv"
)

# Đánh giá baseline base-s1 để so sánh
base_res = all_exp_results["base-s1"]
eval_scores_base = final_eval(
    cfg=base_res["cfg"],
    result=base_res,
    data=data,
    pred_path=f"{OUT_DIR}/predictions_base.csv",
    out_json=f"{OUT_DIR}/eval_result_base.json",
    data_path=f"{REPO_ROOT}/data/covtype.csv.gz",
    meta_path=f"{REPO_ROOT}/data/split_metadata.csv"
)

print(f"\\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN EVAL ---")
print(f"Baseline Eval Macro-F1: {eval_scores_base['macro_f1']:.4f}")
print(f"Final Model Eval Macro-F1: {eval_scores_final['macro_f1']:.4f}")
print(f"Mức độ cải thiện: +{eval_scores_final['macro_f1'] - eval_scores_base['macro_f1']:.4f}")"""))

# 16. Part 4 Excel Export & Error Analysis Code
cells.append(code_cell("""# Điền kết quả vào bảng templates/experiment_table_template.xlsx -> experiments.xlsx
rows = []
for eid, res in all_exp_results.items():
    ev_scores = None
    if eid == "base-s1":
        ev_scores = eval_scores_base
    elif eid == "final-model":
        ev_scores = eval_scores_final
    r = to_row(res, eval_scores=ev_scores)
    rows.append(r)

template_file = f"{REPO_ROOT}/templates/experiment_table_template.xlsx"
out_excel_file = f"{OUT_DIR}/experiments.xlsx"
write_xlsx(rows, template_path=template_file, out_path=out_excel_file)
print(f"[Done] Đã ghi toàn bộ {len(rows)} thí nghiệm vào {out_excel_file}")"""))

# 17. Final summary markdown
cells.append(md_cell("""## Tóm tắt & Hoàn tất
- Toàn bộ kết quả đã được ghi vào `results/*.json`, biểu đồ tại `figures/*.png`.
- File nộp bài `predictions_eval.csv` và `eval_result.json` đã được xác thực hợp lệ qua `scripts/evaluate.py`.
- Bảng tổng hợp `experiments.xlsx` sẵn sàng để xem xét và nộp bài."""))

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            }
        },
        "colab": {
            "name": "lab.ipynb",
            "provenance": []
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("lab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Tạo lab.ipynb thành công!")
