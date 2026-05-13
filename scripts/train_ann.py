"""
train_ann.py
============
MLP 학습 실행 entry point.

흐름 :
  1. configs/network.yaml 로드
  2. data/training_data.csv → 80/10/10 split + normalization (Scaler fit)
  3. MLP 구성 + Trainer 로 학습
  4. best checkpoint, history.json, scaler.npz 저장
  5. TensorBoard 로그는 results/runs/<timestamp>/ 에 기록

사용 :
  python scripts/train_ann.py
  python scripts/train_ann.py --config configs/network.yaml
  python scripts/train_ann.py --csv data/training_data.csv   (cfg override)

학습 후 :
  tensorboard --logdir results/runs
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ann_dataset import prepare_datasets
from src.ann_model   import build_model_from_config
from src.ann_trainer import Trainer


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 완전한 결정성보다는 속도를 우선
    torch.backends.cudnn.benchmark = True


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Train MLP for transfer prediction")
    ap.add_argument('--config', default='configs/network.yaml',
                    help='network 설정 yaml 경로')
    ap.add_argument('--csv',    default=None,
                    help='입력 csv (지정 시 yaml 의 data.csv_path 를 override)')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.csv:
        cfg['data']['csv_path'] = args.csv

    seed = int(cfg.get('data', {}).get('seed', 42))
    set_global_seed(seed)

    print("=" * 70)
    print("  Train MLP (논문 §3 ANN 재현)")
    print("=" * 70)
    print(f"  config : {args.config}")
    print(f"  csv    : {cfg['data']['csv_path']}")
    print(f"  seed   : {seed}")

    # ── Dataset 준비 (split + normalization) ────────────────
    datasets, scaler, info = prepare_datasets(
        csv_path  = cfg['data']['csv_path'],
        split_cfg = cfg['data']['split'],
        norm_cfg  = cfg['normalization'],
        seed      = seed,
    )
    print("-" * 70)
    print(f"  N_total : {info['n_total']}")
    print(f"    train : {info['n_train']}")
    print(f"    val   : {info['n_val']}")
    print(f"    test  : {info['n_test']}")
    print(f"  inputs  : {info['input_cols']}")
    print(f"  outputs : {info['output_cols']}")
    print(f"  scaler.x_min = {scaler.x_min}")
    print(f"  scaler.x_max = {scaler.x_max}")
    print(f"  scaler.y_min = {scaler.y_min}  (transform domain)")
    print(f"  scaler.y_max = {scaler.y_max}  (transform domain)")
    print(f"  y_method     = {scaler.y_method}")
    print("-" * 70)

    # ── 모델 구축 ─────────────────────────────────────────
    model = build_model_from_config(cfg['model'])
    print(f"  model :")
    print(model)
    print(f"  parameter count : {model.count_parameters()}")
    print("-" * 70)

    # ── 학습 ──────────────────────────────────────────────
    trainer = Trainer(model=model, scaler=scaler, cfg=cfg, datasets=datasets)
    result  = trainer.fit()

    # ── 요약 ──────────────────────────────────────────────
    print("=" * 70)
    print("  학습 완료")
    print("=" * 70)
    print(f"  best epoch    : {result['best_epoch']}")
    print(f"  best val loss : {result['best_val_loss']:.6f}")
    print(f"  ckpt dir      : {result['ckpt_dir']}")
    print(f"  log  dir      : {result['log_dir']}")
    test = result['test']
    print("-" * 70)
    print(f"  TEST loss_total       : {test['loss_total']:.5f}")
    print(f"  TEST R²(m_prop)       : {test['m_prop_R2']:.4f}")
    print(f"  TEST R²(TOF)          : {test['TOF_R2']:.4f}")
    print(f"  TEST MAE m_prop  [kg] : {test['m_prop_MAE_kg']:.4f}")
    print(f"  TEST MAE TOF     [d]  : {test['TOF_MAE_days']:.4f}")
    if test['m_prop_err_pct_mean'] is not None:
        print(f"  TEST err m_prop  %    : "
              f"mean {test['m_prop_err_pct_mean']:+.3f}  "
              f"|max| {test['m_prop_err_pct_abs_max']:.2f}")
    if test['TOF_err_pct_mean'] is not None:
        print(f"  TEST err TOF     %    : "
              f"mean {test['TOF_err_pct_mean']:+.3f}  "
              f"|max| {test['TOF_err_pct_abs_max']:.2f}")
    print("=" * 70)
    print()
    print("TensorBoard 시작:")
    print(f"  tensorboard --logdir {os.path.dirname(result['log_dir'])}")
    print()


if __name__ == '__main__':
    main()