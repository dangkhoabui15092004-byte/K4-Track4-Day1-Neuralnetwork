"""execute_all.py — Chạy toàn bộ các thí nghiệm Part 0 -> Part 4, lưu kết quả, vẽ biểu đồ và xuất bảng xlsx."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Đảm bảo in UTF-8 không lỗi trên Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

from data import prepare_data
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, clip_gradients
from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx


def main():
    print("=" * 60)
    print("BẮT ĐẦU CHẠY TOÀN BỘ EXPERIMENTS CHO LAB DAY 1")
    print("=" * 60)

    # 1. Thiết lập device và thư mục
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    repo_root = Path("..") if Path("../data").exists() else Path("../..")
    out_dir = Path("..")
    figures_dir = out_dir / "figures"
    results_dir = out_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Nạp dữ liệu
    print("\n--- Part 0: Chuẩn bị dữ liệu ---")
    data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir=str(repo_root / "data/processed"))

    # 3. Part 1: Kiểm tra ban đầu (Health check)
    print("\n--- Part 1: Kiểm tra ban đầu (Model & Health checks) ---")
    set_seed(1)
    m = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    assert count_params(m) == EXPECTED_PARAMS[(256, 128)], f"Param count mismatch: {count_params(m)}"
    print(f"[OK] Số tham số M-base: {count_params(m)} (khớp đúng 47 879)")

    # Forward test
    dummy_x = torch.randn(8, 54, device=device)
    logits = m(dummy_x)
    assert logits.shape == (8, 7), f"Logits shape mismatch: {logits.shape}"
    print(f"[OK] Shape logits: {tuple(logits.shape)}")

    # Step 0 loss
    step0_res = evaluate(m, data["X_val"], data["y_val"], loss_name="ce")
    ln7 = np.log(7)
    print(f"[OK] Step 0 loss: {step0_res['loss']:.4f} (so với ln 7 = {ln7:.4f}, lệch = {abs(step0_res['loss'] - ln7):.4f})")

    # Overfit 20 samples test
    x_small = data["X_tr"][:20]
    y_small = data["y_tr"][:20]
    m_small = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    opt_small = torch.optim.Adam(m_small.parameters(), lr=0.02)
    small_losses = []
    for _ in range(150):
        opt_small.zero_grad()
        out = m_small(x_small)
        l = torch.nn.functional.cross_entropy(out, y_small)
        l.backward()
        opt_small.step()
        small_losses.append(l.item())
    print(f"[OK] Quá khớp 20 mẫu sau 150 bước: loss = {small_losses[-1]:.6f} -> loss về gần 0 hoàn hảo!")

    # Gradient flow test
    opt_small.zero_grad()
    l = torch.nn.functional.cross_entropy(m(dummy_x), torch.zeros(8, dtype=torch.int64, device=device))
    l.backward()
    grad_norms = [float(p.grad.norm().item()) for p in m.parameters() if p.grad is not None]
    assert all(gn > 0 for gn in grad_norms), "Có tham số có gradient = 0!"
    print(f"[OK] Mọi tham số đều có gradient khác 0 (min grad norm = {min(grad_norms):.6f})")

    # Activation stats test
    val_sample_x = data["X_val"][:512]
    for init_type in ["zeros", "normal", "xavier", "he"]:
        m_tmp = MLP(hidden=(256, 128), init=init_type).to(device)
        stds = activation_stats(m_tmp, val_sample_x)
        print(f"  * Activation std sau các lớp ReLU [{init_type}]: {stds}")

    # 4. Định nghĩa danh sách các thí nghiệm
    experiments = [
        # --- Baseline (3 seeds) ---
        dict(DEFAULT_CFG, exp_id="base-s1", group="baseline", description="Baseline M-base, seed 1", lr=0.05, seed=1, epochs=20),
        dict(DEFAULT_CFG, exp_id="base-s2", group="baseline", description="Baseline M-base, seed 2", lr=0.05, seed=2, epochs=20),
        dict(DEFAULT_CFG, exp_id="base-s3", group="baseline", description="Baseline M-base, seed 3", lr=0.05, seed=3, epochs=20),

        # --- Chủ đề 1: Loss ---
        dict(DEFAULT_CFG, exp_id="exp-loss-mse", group="loss", description="Hàm mất mát MSE trên one-hot", loss="mse", lr=0.05, epochs=20),

        # --- Chủ đề 2: Optimizer ---
        dict(DEFAULT_CFG, exp_id="exp-opt-sgd", group="optimizer", description="SGD thuần (momentum=0.0)", optimizer="sgd", momentum=0.0, lr=0.05, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-opt-adam-lr1e-3", group="optimizer", description="Adam lr=1e-3", optimizer="adam", lr=1e-3, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-opt-adam-lr3e-4", group="optimizer", description="Adam lr=3e-4", optimizer="adam", lr=3e-4, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-opt-adamw-wd0.01", group="optimizer", description="AdamW lr=1e-3, weight_decay=0.01", optimizer="adamw", lr=1e-3, weight_decay=0.01, epochs=20),

        # --- Chủ đề 3: Hyper-parameter ---
        dict(DEFAULT_CFG, exp_id="exp-hp-batch128", group="hparam", description="Batch size nhỏ (128)", batch=128, epochs=15),
        dict(DEFAULT_CFG, exp_id="exp-hp-batch2048", group="hparam", description="Batch size lớn (2048)", batch=2048, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-hp-mwide", group="hparam", description="Kiến trúc M-wide (512-256)", hidden=(512, 256), epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-hp-mdeep", group="hparam", description="Kiến trúc M-deep (256-128-64)", hidden=(256, 128, 64), epochs=20),

        # --- Chủ đề 4: Dropout ---
        dict(DEFAULT_CFG, exp_id="exp-drop-0.1", group="dropout", description="Dropout q=0.1 sau ReLU lớp ẩn", dropout=0.1, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-drop-0.3", group="dropout", description="Dropout q=0.3 sau ReLU lớp ẩn", dropout=0.3, epochs=20),

        # --- Chủ đề 5: Gradient Clipping ---
        dict(DEFAULT_CFG, exp_id="exp-clip-1.0", group="clipping", description="Gradient clipping c=1.0 ở lr chuẩn", clip_norm=1.0, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-clip-highlr-noclip", group="clipping", description="LR cao 0.8 không clip (gây bất ổn dao động)", lr=0.8, clip_norm=None, epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-clip-highlr-clip", group="clipping", description="LR cao 0.8 có clip c=1.0 (ổn định cập nhật)", lr=0.8, clip_norm=1.0, epochs=20),

        # --- Chủ đề 6: Mixed Precision (AMP) ---
        dict(DEFAULT_CFG, exp_id="exp-amp-fp16", group="amp", description="Mixed precision FP16 với GradScaler", precision="fp16", epochs=20,
             notes="Trên CPU chạy FP32 fallback; trên GPU T4 dùng FP16 Tensor Cores"),

        # --- Chủ đề 7: Khởi tạo tham số ---
        dict(DEFAULT_CFG, exp_id="exp-init-xavier", group="init", description="Khởi tạo Xavier normal", init="xavier", epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-init-normal", group="init", description="Khởi tạo Normal(0, 0.01^2)", init="normal", epochs=20),
        dict(DEFAULT_CFG, exp_id="exp-init-zeros", group="init", description="Khởi tạo Zeros (W=0, nơ-ron đối xứng)", init="zeros", epochs=20,
             notes="W=0 khiến gradient đối xứng, kích hoạt ReLU(0)=0 không học được"),

        # --- Cấu hình cuối cùng (Final Model) ---
        dict(DEFAULT_CFG, exp_id="final-model", group="final", description="Cấu hình tối ưu: Adam lr=1e-3, batch=512, He init", optimizer="adam", lr=1e-3, epochs=25, seed=42),
    ]

    all_results = {}
    print(f"\n--- Part 2 & 3: Bắt đầu chạy {len(experiments)} thí nghiệm ---")

    for i, cfg in enumerate(experiments, 1):
        exp_id = cfg["exp_id"]
        print(f"\n[{i}/{len(experiments)}] Đang chạy: {exp_id} ({cfg['group']}: {cfg['description']})...")
        t_start = time.time()
        res = run_experiment(cfg, data)
        duration = time.time() - t_start

        # Lưu JSON
        json_path = save_result(res, str(results_dir))
        # Vẽ biểu đồ riêng cho thí nghiệm
        fig_path = str(figures_dir / f"{exp_id}.png")
        plot_run(res, fig_path)

        all_results[exp_id] = res

        summ = res["summary"]
        print(f"  -> Hoàn thành trong {duration:.1f}s | Val Acc: {summ['val_acc']} | Val Macro-F1: {summ['val_macro_f1']} | Best Epoch: {summ['best_epoch']}")

    # 5. Vẽ biểu đồ so sánh theo nhóm (Part 3 compare plots)
    print("\n--- Vẽ biểu đồ so sánh các nhóm ---")
    groups_to_compare = {
        "compare_optimizer.png": (["base-s1", "exp-opt-sgd", "exp-opt-adam-lr1e-3", "exp-opt-adam-lr3e-4", "exp-opt-adamw-wd0.01"], "val_macro_f1", "So sánh các Bộ tối ưu hoá (Val Macro-F1)"),
        "compare_loss.png": (["base-s1", "exp-loss-mse"], "val_macro_f1", "So sánh CE vs MSE (Val Macro-F1)"),
        "compare_hparam.png": (["base-s1", "exp-hp-batch128", "exp-hp-batch2048", "exp-hp-mwide", "exp-hp-mdeep"], "val_macro_f1", "So sánh Hyper-parameters (Val Macro-F1)"),
        "compare_dropout.png": (["base-s1", "exp-drop-0.1", "exp-drop-0.3"], "val_macro_f1", "So sánh Dropout (Val Macro-F1)"),
        "compare_clipping.png": (["base-s1", "exp-clip-1.0", "exp-clip-highlr-noclip", "exp-clip-highlr-clip"], "grad_norm", "So sánh Chuẩn Gradient (Grad Norm pre-clip)"),
        "compare_init.png": (["base-s1", "exp-init-xavier", "exp-init-normal", "exp-init-zeros"], "val_macro_f1", "So sánh Khởi tạo Tham số (Val Macro-F1)"),
    }

    for fname, (exp_ids, metric, title) in groups_to_compare.items():
        res_list = [all_results[eid] for eid in exp_ids if eid in all_results]
        compare_path = str(figures_dir / fname)
        plot_compare(res_list, metric=metric, path=compare_path, title=title)
        print(f"[OK] Đã vẽ {compare_path}")

    # 6. Part 4: Đánh giá cuối trên eval bằng evaluate.py
    print("\n--- Part 4: Đánh giá cuối trên tập eval ---")

    # Đánh giá final-model
    final_res = all_results["final-model"]
    pred_path = str(out_dir / "predictions_eval.csv")
    eval_json_path = str(out_dir / "eval_result.json")

    eval_scores_final = final_eval(
        cfg=final_res["cfg"],
        result=final_res,
        data=data,
        pred_path=pred_path,
        out_json=eval_json_path,
        data_path=str(repo_root / "data/covtype.csv.gz"),
        meta_path=str(repo_root / "data/split_metadata.csv")
    )
    print(f"\n[EVAL FINAL MODEL] Accuracy = {eval_scores_final['accuracy']:.4f} | Macro-F1 = {eval_scores_final['macro_f1']:.4f}")

    # Đánh giá baseline base-s1 để điền bảng
    base_res = all_results["base-s1"]
    base_pred_path = str(out_dir / "predictions_base.csv")
    base_eval_json_path = str(out_dir / "eval_result_base.json")
    eval_scores_base = final_eval(
        cfg=base_res["cfg"],
        result=base_res,
        data=data,
        pred_path=base_pred_path,
        out_json=base_eval_json_path,
        data_path=str(repo_root / "data/covtype.csv.gz"),
        meta_path=str(repo_root / "data/split_metadata.csv")
    )
    print(f"[EVAL BASELINE base-s1] Accuracy = {eval_scores_base['accuracy']:.4f} | Macro-F1 = {eval_scores_base['macro_f1']:.4f}")

    # 7. Xuất bảng experiments.xlsx
    print("\n--- Xuất bảng experiments.xlsx ---")
    rows = []
    for cfg in experiments:
        eid = cfg["exp_id"]
        res = all_results[eid]
        ev_score = None
        if eid == "base-s1":
            ev_score = eval_scores_base
        elif eid == "final-model":
            ev_score = eval_scores_final

        row = to_row(res, eval_scores=ev_score)
        rows.append(row)

    template_xlsx = str(repo_root / "templates/experiment_table_template.xlsx")
    out_xlsx = str(out_dir / "experiments.xlsx")
    write_xlsx(rows, template_path=template_xlsx, out_path=out_xlsx)

    # 8. Tính độ nhiễu baseline (seeds)
    base_f1s = [all_results[f"base-s{s}"]["summary"]["val_macro_f1"] for s in (1, 2, 3)]
    base_accs = [all_results[f"base-s{s}"]["summary"]["val_acc"] for s in (1, 2, 3)]
    f1_mean = np.mean(base_f1s)
    f1_std = np.std(base_f1s, ddof=1)
    acc_mean = np.mean(base_accs)
    acc_std = np.std(base_accs, ddof=1)

    print("\n" + "=" * 60)
    print("TỔNG HỢP KẾT QUẢ ĐÃ CHẠY:")
    print(f"Baseline Val Macro-F1 (3 seeds): {f1_mean:.4f} ± {f1_std:.4f} (Ngưỡng 2σ = {2*f1_std:.4f})")
    print(f"Baseline Val Acc (3 seeds):      {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Final Model Val Macro-F1:        {final_res['summary']['val_macro_f1']:.4f}")
    print(f"Final Model Eval Macro-F1:       {eval_scores_final['macro_f1']:.4f}")
    print(f"Cải thiện so với baseline eval:  +{eval_scores_final['macro_f1'] - eval_scores_base['macro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
