# 학습 끝난 후 한 번 돌려보면 가장 정보량 큰 진단
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, numpy as np
from src.ann_dataset import prepare_datasets, load_csv_as_arrays
from src.ann_model import build_model_from_config
from src.ann_dataset import Scaler
import yaml

with open('configs/network.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

ckpt_dir = 'results/checkpoints/20260513_222801'
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

# 오차가 큰 상위 100 개 찾기
err_pct_mp = 100 * (Y_pred[:,0] - Y[:,0]) / np.maximum(Y[:,0], 1e-3)
err_pct_tof = 100 * (Y_pred[:,1] - Y[:,1]) / np.maximum(Y[:,1], 1e-3)

# m_prop 오차 큰 top 20 출력
idx = np.argsort(np.abs(err_pct_mp))[-20:][::-1]
print("m_prop 큰 오차 상위 20:")
print(" h_D1   Ω_D1    m_D1  h_D2   Ω_D2    m_SC | m_prop_true m_prop_pred err%")
for i in idx:
    print(f"{X[i,0]:6.1f} {X[i,1]:6.1f} {X[i,2]:6.1f} {X[i,3]:6.1f} "
          f"{X[i,4]:6.1f} {X[i,5]:6.1f} | {Y[i,0]:10.3f} {Y_pred[i,0]:10.3f} "
          f"{err_pct_mp[i]:+.2f}%")