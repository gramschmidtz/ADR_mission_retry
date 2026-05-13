"""
ann_dataset.py
==============
training_data.csv → PyTorch Dataset + normalization.

논문 §3.1 매핑:
  입력 x = [h_D1, RAAN_D1, m_D1, h_D2, RAAN_D2, m_SC]        (6)
  출력 y = [m_prop, TOF]                                      (2)

정규화
------
입력 : minmax (학습 split 의 min/max 사용)
출력 :
  m_prop : minmax
  TOF    : log1p → minmax   (long-tail 압축)

split
-----
무작위 셔플 후 train/val/test 비율로 절단. seed 로 재현성 확보.

저장 / 로드
-----------
fit_scaler 가 만든 dict 를 `Scaler.save_npz(path)` 로 저장하고,
inference 시 `Scaler.load_npz(path)` 로 복원 후 inverse_transform 사용.
"""

from __future__ import annotations

import os
import csv
import math
import numpy as np
import torch
from torch.utils.data import Dataset


# 컬럼 이름 — generate_training_data.py 출력과 일치해야 함
INPUT_COLS  = ['h_D1_km', 'RAAN_D1_deg', 'm_D1_kg',
               'h_D2_km', 'RAAN_D2_deg', 'm_SC_kg']
OUTPUT_COLS = ['m_prop_kg', 'TOF_days']


# ──────────────────────────────────────────────────────────────
# CSV 로드 (success=1 행만, numpy array 반환)
# ──────────────────────────────────────────────────────────────
def load_csv_as_arrays(csv_path, require_success=True):
    """
    csv → (X, Y) numpy arrays.
    X.shape = (N, 6), Y.shape = (N, 2). dtype float32.

    require_success=True 면 success=1 행만 반환.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv not found: {csv_path}")

    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if require_success and r.get('success', '0') != '1':
                continue
            try:
                x = [float(r[c]) for c in INPUT_COLS]
                y = [float(r[c]) for c in OUTPUT_COLS]
            except (ValueError, KeyError):
                continue
            # 유효성 검사 — NaN 행 skip
            if any(math.isnan(v) for v in x) or any(math.isnan(v) for v in y):
                continue
            rows.append(x + y)

    if not rows:
        raise RuntimeError(f"csv 에 유효한 sample 이 없음: {csv_path}")

    arr = np.asarray(rows, dtype=np.float32)
    X = arr[:, :len(INPUT_COLS)]
    Y = arr[:, len(INPUT_COLS):]
    return X, Y


# ──────────────────────────────────────────────────────────────
# Scaler — fit / transform / inverse / 저장
# ──────────────────────────────────────────────────────────────
class Scaler:
    """
    입력 6 차원 + 출력 2 차원의 normalization 파라미터를 보관.

    입력 : minmax 만 지원
    출력 :
      m_prop : minmax
      TOF    : log1p_minmax  (log1p(y) → minmax)

    내부 표현 :
      x_min, x_max         : (6,)
      y_method[i]          : 'minmax' | 'log1p_minmax'  (i=0 m_prop, i=1 TOF)
      y_min, y_max         : (2,)     transform 후 도메인에서의 min/max
    """

    EPS = 1e-12

    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.y_method = ['minmax', 'minmax']   # default
        self.y_min = None
        self.y_max = None

    # ── fit ───────────────────────────────────────────────
    def fit(self, X_train, Y_train, cfg_norm):
        # 입력
        in_method = cfg_norm.get('inputs', {}).get('method', 'minmax')
        if in_method != 'minmax':
            raise ValueError(
                f"현재는 입력 minmax 만 지원 (요청: {in_method}). "
                "확장하려면 Scaler 수정."
            )
        self.x_min = X_train.min(axis=0).astype(np.float32)
        self.x_max = X_train.max(axis=0).astype(np.float32)

        # 출력 m_prop
        m_method = cfg_norm.get('outputs', {}).get('m_prop', {}).get(
            'transform', 'minmax')
        # 출력 TOF
        t_method = cfg_norm.get('outputs', {}).get('TOF', {}).get(
            'transform', 'log1p_minmax')
        self.y_method = [m_method, t_method]

        Y_t = self._apply_y_forward(Y_train)
        self.y_min = Y_t.min(axis=0).astype(np.float32)
        self.y_max = Y_t.max(axis=0).astype(np.float32)
        return self

    # ── forward (X, Y → normalized) ───────────────────────
    def transform_x(self, X):
        return (X - self.x_min) / (self.x_max - self.x_min + self.EPS)

    def transform_y(self, Y):
        Y_t = self._apply_y_forward(Y)
        return (Y_t - self.y_min) / (self.y_max - self.y_min + self.EPS)

    # ── inverse (normalized → 원 단위) ───────────────────
    def inverse_x(self, X_norm):
        return X_norm * (self.x_max - self.x_min) + self.x_min

    def inverse_y(self, Y_norm):
        Y_t = Y_norm * (self.y_max - self.y_min) + self.y_min
        return self._apply_y_inverse(Y_t)

    # ── 출력 변환 도우미 ────────────────────────────────
    def _apply_y_forward(self, Y):
        """원 단위 → transform 도메인 (still float, not minmax-normalized)."""
        out = np.empty_like(Y, dtype=np.float32)
        for j, method in enumerate(self.y_method):
            if method == 'minmax':
                out[:, j] = Y[:, j]
            elif method == 'log1p_minmax':
                # 음수 안전 처리 — clipping
                out[:, j] = np.log1p(np.maximum(Y[:, j], 0.0))
            else:
                raise ValueError(f"unsupported y transform: {method}")
        return out

    def _apply_y_inverse(self, Y_t):
        """transform 도메인 → 원 단위."""
        out = np.empty_like(Y_t, dtype=np.float32)
        for j, method in enumerate(self.y_method):
            if method == 'minmax':
                out[:, j] = Y_t[:, j]
            elif method == 'log1p_minmax':
                out[:, j] = np.expm1(Y_t[:, j])
            else:
                raise ValueError(f"unsupported y transform: {method}")
        return out

    # ── 직렬화 ───────────────────────────────────────────
    def save_npz(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            x_min   = self.x_min,
            x_max   = self.x_max,
            y_min   = self.y_min,
            y_max   = self.y_max,
            y_method = np.asarray(self.y_method),
        )

    @classmethod
    def load_npz(cls, path):
        data = np.load(path, allow_pickle=False)
        s = cls()
        s.x_min = data['x_min'].astype(np.float32)
        s.x_max = data['x_max'].astype(np.float32)
        s.y_min = data['y_min'].astype(np.float32)
        s.y_max = data['y_max'].astype(np.float32)
        s.y_method = [str(m) for m in data['y_method']]
        return s


# ──────────────────────────────────────────────────────────────
# Tensor Dataset (정규화 적용 후 보관)
# ──────────────────────────────────────────────────────────────
class TransferDataset(Dataset):
    """이미 정규화된 (X, Y) 를 보관하는 단순 Dataset."""

    def __init__(self, X_norm, Y_norm):
        self.X = torch.from_numpy(np.ascontiguousarray(X_norm)).float()
        self.Y = torch.from_numpy(np.ascontiguousarray(Y_norm)).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ──────────────────────────────────────────────────────────────
# 80/10/10 split + Scaler fit + Dataset 생성
# ──────────────────────────────────────────────────────────────
def prepare_datasets(csv_path, split_cfg, norm_cfg, seed=42):
    """
    csv → train/val/test Dataset 3 개 + scaler 반환.

    Parameters
    ----------
    split_cfg : dict   {'train': 0.8, 'val': 0.1, 'test': 0.1}
    norm_cfg  : dict   network.yaml 의 normalization 섹션
    seed      : int    셔플용 seed

    Returns
    -------
    datasets : dict    {'train': TransferDataset, 'val': ..., 'test': ...}
    scaler   : Scaler  fit 끝난 객체 (저장 / inverse 용)
    raw_info : dict    원본 split 의 index / 통계 등 (디버깅용)
    """
    X, Y = load_csv_as_arrays(csv_path, require_success=True)
    N = X.shape[0]

    p_train = float(split_cfg.get('train', 0.8))
    p_val   = float(split_cfg.get('val',   0.1))
    p_test  = float(split_cfg.get('test',  0.1))
    s = p_train + p_val + p_test
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"split 비율 합이 1 이 아님: {s}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(round(N * p_train))
    n_val   = int(round(N * p_val))
    n_test  = N - n_train - n_val

    idx_tr  = idx[:n_train]
    idx_va  = idx[n_train:n_train + n_val]
    idx_te  = idx[n_train + n_val:]

    X_tr, Y_tr = X[idx_tr], Y[idx_tr]
    X_va, Y_va = X[idx_va], Y[idx_va]
    X_te, Y_te = X[idx_te], Y[idx_te]

    # 정규화는 train 통계로만 fit
    scaler = Scaler().fit(X_tr, Y_tr, norm_cfg)

    datasets = {
        'train': TransferDataset(
            scaler.transform_x(X_tr), scaler.transform_y(Y_tr)),
        'val'  : TransferDataset(
            scaler.transform_x(X_va), scaler.transform_y(Y_va)),
        'test' : TransferDataset(
            scaler.transform_x(X_te), scaler.transform_y(Y_te)),
    }
    raw_info = {
        'n_total'     : int(N),
        'n_train'     : int(n_train),
        'n_val'       : int(n_val),
        'n_test'      : int(n_test),
        'input_cols'  : list(INPUT_COLS),
        'output_cols' : list(OUTPUT_COLS),
        'seed'        : int(seed),
    }
    return datasets, scaler, raw_info