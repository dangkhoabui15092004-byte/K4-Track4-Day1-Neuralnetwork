"""finish_post_process.py — Đánh giá baseline trên eval, xuất experiments.xlsx và in tổng hợp kết quả."""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import subprocess
from pathlib import Path
import numpy as np
import torch

from model import MLP
from train import predict, write_predictions
from data import prepare_data
from results_table import to_row, write_xlsx, load_results

# Nạp dữ liệu
data = prepare_data("cpu", val_fraction=0.2, seed=42, processed_dir="../data/processed")

# Đánh giá baseline base-s1 trên eval
with open("../results/base-s1.json", "r", encoding="utf-8") as f:
    base_res = json.load(f)

# Tạo lại model và dự đoán
# Do best_state không được lưu trong JSON (để nhẹ file), ta train nhanh base-s1 hoặc nạp lại
print("Dự đoán eval cho base-s1...")
m_base = MLP(hidden=(256, 128), init="he")
# train base-s1 20 epochs để lấy best state
from train import run_experiment
res_base_trained = run_experiment(base_res["cfg"], data)
m_base.load_state_dict(res_base_trained["best_state"])
preds_base = predict(m_base, data["X_eval"])

write_predictions(data["eval_row_id"], preds_base.cpu().numpy(), "../predictions_base.csv")

# Chạy evaluate.py cho base-s1
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
cmd = [sys.executable, "scripts/evaluate.py", "--pred", "predictions_base.csv", "--out", "eval_result_base.json"]
subprocess.run(cmd, cwd="..", env=env, check=True)

with open("../eval_result_base.json", "r", encoding="utf-8") as f:
    eval_base = json.load(f)

print(f"Baseline Eval Acc: {eval_base['accuracy']:.4f} | Eval Macro-F1: {eval_base['macro_f1']:.4f}")

with open("../eval_result.json", "r", encoding="utf-8") as f:
    eval_final = json.load(f)

print(f"Final Model Eval Acc: {eval_final['accuracy']:.4f} | Eval Macro-F1: {eval_final['macro_f1']:.4f}")

# Nạp toàn bộ kết quả để ghi vào experiments.xlsx
results = load_results("../results")
rows = []
for res in results:
    eid = res["cfg"]["exp_id"]
    ev_score = None
    if eid == "base-s1":
        ev_score = eval_base
    elif eid == "final-model":
        ev_score = eval_final
    r = to_row(res, eval_scores=ev_score)
    rows.append(r)

write_xlsx(rows, template_path="../templates/experiment_table_template.xlsx", out_path="../experiments.xlsx")
print("Đã ghi experiments.xlsx thành công!")
