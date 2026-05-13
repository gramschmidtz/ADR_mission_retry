import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch, yaml
from src.ann_dataset import load_csv_as_arrays, Scaler
from src.ann_model import build_model_from_config

with open('configs/network.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

ckpt_dir = 'results/checkpoints/20260513_222801'  # 본인 경로
state = torch.load(f'{ckpt_dir}/best.pt', map_location='cpu')
model = build_model_from_config(cfg['model'])
model.load_state_dict(state['model_state'])
model.eval()
scaler = Scaler.load_npz(f'{ckpt_dir}/scaler.npz')

X, Y = load_csv_as_arrays('data/training_data.csv')
Xn = scaler.transform_x(X)
with torch.no_grad():
    Yn_pred = model(torch.from_numpy(Xn).float()).numpy()
Y_pred = scaler.inverse_y(Yn_pred)

err_mp = 100 * (Y_pred[:,0] - Y[:,0]) / np.maximum(np.abs(Y[:,0]), 1e-3)
err_tof = 100 * (Y_pred[:,1] - Y[:,1]) / np.maximum(np.abs(Y[:,1]), 1e-3)

# 오차 누적 분포
for th in [1, 2, 5, 10, 20, 50, 100]:
    n_mp  = int((np.abs(err_mp)  > th).sum())
    n_tof = int((np.abs(err_tof) > th).sum())
    print(f"  |err| > {th:>4d}% :  m_prop {n_mp:>6d} ({100*n_mp/len(Y):.3f}%)   "
          f"TOF {n_tof:>6d} ({100*n_tof/len(Y):.3f}%)")