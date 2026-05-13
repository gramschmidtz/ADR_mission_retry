"""
ann_trainer.py
==============
MLP 학습 loop. TensorBoard 로 진행상황 / 회귀 품질을 시각화한다.

기능
----
  - train / val epoch loop, AdamW/Adam/SGD, ReduceLROnPlateau
  - per-component loss : m_prop / TOF 따로 + 합산
  - 매 epoch 마다 정규화된 도메인 + 원 단위 도메인 모두에서 metric 계산
  - early stopping
  - best checkpoint 저장 (val_loss 기준)
  - TensorBoard scalar / scatter / histogram

TensorBoard 에 기록되는 것
--------------------------
  scalar :
    loss/train_total      loss/val_total
    loss/train_m_prop     loss/val_m_prop
    loss/train_TOF        loss/val_TOF
    metric/val_m_prop_R2  metric/val_TOF_R2
    metric/val_m_prop_MAE_kg     (원 단위)
    metric/val_TOF_MAE_days      (원 단위)
    lr
    epoch_time_sec
  figure (scatter_every_n_epochs 마다) :
    val/scatter_m_prop    (pred vs true, 원 단위)
    val/scatter_TOF       (pred vs true, 원 단위, log-log 옵션)
  histogram :
    val/err_m_prop_pct    (mean percentage error)
    val/err_TOF_pct
"""

from __future__ import annotations

import os
import io
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ──────────────────────────────────────────────────────────────
# Optimizer / scheduler builders
# ──────────────────────────────────────────────────────────────
def _build_optimizer(model, opt_cfg):
    name = opt_cfg.get('name', 'adam').lower()
    lr   = float(opt_cfg.get('lr', 1e-3))
    wd   = float(opt_cfg.get('weight_decay', 0.0))
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'sgd':
        momentum = float(opt_cfg.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr,
                               weight_decay=wd, momentum=momentum)
    raise ValueError(f"unsupported optimizer: {name}")


def _build_scheduler(optimizer, sch_cfg):
    if not sch_cfg:
        return None
    name = sch_cfg.get('name', '').lower()
    if name in ('', 'none', 'null'):
        return None
    if name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor   = float(sch_cfg.get('factor', 0.5)),
            patience = int(sch_cfg.get('patience', 10)),
            min_lr   = float(sch_cfg.get('min_lr', 1e-6)),
        )
    raise ValueError(f"unsupported scheduler: {name}")


# ──────────────────────────────────────────────────────────────
# Metric 도우미
# ──────────────────────────────────────────────────────────────
def _r2(y_true, y_pred):
    """R^2 = 1 - SS_res / SS_tot. shape : (N,) 또는 (N, 1)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-20:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _mean_pct_err(y_true, y_pred):
    """100 * (pred - true) / true. true ≈ 0 인 행 제외."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return np.zeros(0, dtype=np.float32)
    return 100.0 * (y_pred[mask] - y_true[mask]) / y_true[mask]


# ──────────────────────────────────────────────────────────────
# Scatter plot → matplotlib figure (TensorBoard add_figure 용)
# ──────────────────────────────────────────────────────────────
def _scatter_fig(true_vals, pred_vals, label, max_points=5000,
                 log_axes=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    true_vals = np.asarray(true_vals).reshape(-1)
    pred_vals = np.asarray(pred_vals).reshape(-1)
    n = len(true_vals)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        true_vals = true_vals[idx]
        pred_vals = pred_vals[idx]

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(true_vals, pred_vals, s=3, alpha=0.3, color='tab:blue')
    lo = float(min(np.min(true_vals), np.min(pred_vals)))
    hi = float(max(np.max(true_vals), np.max(pred_vals)))
    ax.plot([lo, hi], [lo, hi], 'r-', lw=1, alpha=0.7, label='y = x')
    if log_axes:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(f'true {label}')
    ax.set_ylabel(f'pred {label}')
    ax.set_title(f'{label} : pred vs true  (R² shown in scalars, n={n})')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, scaler, cfg, datasets, device=None):
        """
        Parameters
        ----------
        model    : torch.nn.Module
        scaler   : src.ann_dataset.Scaler  (inverse 용 — 원 단위 metric 계산)
        cfg      : dict  (network.yaml 전체)
        datasets : dict  {'train': Dataset, 'val': Dataset, 'test': Dataset}
        device   : 'cuda' | 'cpu' | None (자동 감지)
        """
        self.cfg = cfg
        self.scaler = scaler
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        train_cfg = cfg.get('train', {})
        self.batch_size  = int(train_cfg.get('batch_size', 256))
        self.max_epochs  = int(train_cfg.get('max_epochs', 200))
        self.num_workers = int(train_cfg.get('num_workers', 0))

        # DataLoader
        self.loaders = {
            'train': DataLoader(
                datasets['train'], batch_size=self.batch_size,
                shuffle=True,  num_workers=self.num_workers,
                pin_memory=(self.device == 'cuda'),
            ),
            'val': DataLoader(
                datasets['val'], batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
                pin_memory=(self.device == 'cuda'),
            ),
            'test': DataLoader(
                datasets['test'], batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
                pin_memory=(self.device == 'cuda'),
            ),
        }

        # Optimizer / scheduler
        self.optimizer = _build_optimizer(self.model, train_cfg.get('optimizer', {}))
        self.scheduler = _build_scheduler(self.optimizer, train_cfg.get('scheduler', {}))

        # Loss
        loss_name = train_cfg.get('loss', {}).get('name', 'mse').lower()
        if loss_name != 'mse':
            raise ValueError(f"unsupported loss: {loss_name}")
        self.criterion = nn.MSELoss(reduction='mean')

        # Early stopping
        es_cfg = train_cfg.get('early_stopping', {})
        self.es_enable    = bool(es_cfg.get('enable', True))
        self.es_patience  = int(es_cfg.get('patience', 20))
        self.es_min_delta = float(es_cfg.get('min_delta', 1e-5))

        # TensorBoard / checkpoint dirs
        log_cfg = cfg.get('logging', {})
        # csv 파일명에서 alpha 태그 자동 추출 (있으면 디렉토리명에 부착)
        #   예 : training_data_alpha0.0.csv → '_alpha0.0'
        import re
        csv_path = cfg.get('data', {}).get('csv_path', '')
        m = re.search(r'_alpha([0-9]+(?:\.[0-9]+)?)', os.path.basename(csv_path))
        alpha_tag = f"_alpha{m.group(1)}" if m else ""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + alpha_tag
        self.run_name = timestamp
        self.log_dir  = os.path.join(
            log_cfg.get('log_root', 'results/runs'), timestamp)
        self.ckpt_dir = os.path.join(
            log_cfg.get('ckpt_root', 'results/checkpoints'), timestamp)
        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.scatter_max_points    = int(log_cfg.get('scatter_max_points', 5000))
        self.scatter_every_n_epochs = int(log_cfg.get('scatter_every_n_epochs', 5))

        # 상태
        self.best_val_loss = float('inf')
        self.best_epoch    = -1
        self.epochs_no_improve = 0
        self.history = []   # list of dict per epoch

        # scaler / config 를 ckpt_dir 에 사본 저장
        self.scaler.save_npz(os.path.join(self.ckpt_dir, 'scaler.npz'))
        try:
            import yaml
            with open(os.path.join(self.ckpt_dir, 'network.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)
        except Exception:
            pass

    # ── 한 epoch ─────────────────────────────────────────
    def _run_epoch(self, loader, train_mode):
        """
        Returns
        -------
        dict {
            'loss_total': float,
            'loss_m_prop': float,
            'loss_TOF': float,
            'y_true_norm': np.array (N, 2),
            'y_pred_norm': np.array (N, 2),
        }
        """
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        running_total   = 0.0
        running_mprop   = 0.0
        running_tof     = 0.0
        n_seen          = 0

        y_true_list = []
        y_pred_list = []

        torch_grad = torch.enable_grad() if train_mode else torch.no_grad()
        with torch_grad:
            for X, Y in loader:
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)
                Y_pred = self.model(X)
                # 컴포넌트별 loss
                loss_m = self.criterion(Y_pred[:, 0], Y[:, 0])
                loss_t = self.criterion(Y_pred[:, 1], Y[:, 1])
                loss   = loss_m + loss_t

                if train_mode:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bs = X.shape[0]
                running_total += loss.item()   * bs
                running_mprop += loss_m.item() * bs
                running_tof   += loss_t.item() * bs
                n_seen        += bs

                # validation 에서만 모아둠
                if not train_mode:
                    y_true_list.append(Y.detach().cpu().numpy())
                    y_pred_list.append(Y_pred.detach().cpu().numpy())

        out = {
            'loss_total' : running_total / max(n_seen, 1),
            'loss_m_prop': running_mprop / max(n_seen, 1),
            'loss_TOF'   : running_tof   / max(n_seen, 1),
        }
        if not train_mode:
            out['y_true_norm'] = np.concatenate(y_true_list, axis=0)
            out['y_pred_norm'] = np.concatenate(y_pred_list, axis=0)
        return out

    # ── 학습 메인 ────────────────────────────────────────
    def fit(self):
        print(f"[Trainer] device = {self.device}")
        print(f"[Trainer] run = {self.run_name}")
        print(f"[Trainer] log  = {self.log_dir}")
        print(f"[Trainer] ckpt = {self.ckpt_dir}")
        print(f"[Trainer] model params = {self.model.count_parameters()}")
        print("-" * 70)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(self.loaders['train'], train_mode=True)
            va = self._run_epoch(self.loaders['val'],   train_mode=False)
            epoch_time = time.time() - t0

            # 원 단위 metric (val 만)
            y_true_orig = self.scaler.inverse_y(va['y_true_norm'])
            y_pred_orig = self.scaler.inverse_y(va['y_pred_norm'])
            r2_mp  = _r2( y_true_orig[:, 0], y_pred_orig[:, 0])
            r2_tof = _r2( y_true_orig[:, 1], y_pred_orig[:, 1])
            mae_mp  = float(np.mean(np.abs(y_true_orig[:, 0] - y_pred_orig[:, 0])))
            mae_tof = float(np.mean(np.abs(y_true_orig[:, 1] - y_pred_orig[:, 1])))
            err_mp_pct  = _mean_pct_err(y_true_orig[:, 0], y_pred_orig[:, 0])
            err_tof_pct = _mean_pct_err(y_true_orig[:, 1], y_pred_orig[:, 1])

            lr_now = self.optimizer.param_groups[0]['lr']

            # ── TensorBoard scalars ────────────────────
            self.writer.add_scalar('loss/train_total',  tr['loss_total'],  epoch)
            self.writer.add_scalar('loss/train_m_prop', tr['loss_m_prop'], epoch)
            self.writer.add_scalar('loss/train_TOF',    tr['loss_TOF'],    epoch)
            self.writer.add_scalar('loss/val_total',    va['loss_total'],  epoch)
            self.writer.add_scalar('loss/val_m_prop',   va['loss_m_prop'], epoch)
            self.writer.add_scalar('loss/val_TOF',      va['loss_TOF'],    epoch)
            self.writer.add_scalar('metric/val_m_prop_R2',     r2_mp,     epoch)
            self.writer.add_scalar('metric/val_TOF_R2',        r2_tof,    epoch)
            self.writer.add_scalar('metric/val_m_prop_MAE_kg', mae_mp,    epoch)
            self.writer.add_scalar('metric/val_TOF_MAE_days',  mae_tof,   epoch)
            self.writer.add_scalar('lr', lr_now, epoch)
            self.writer.add_scalar('epoch_time_sec', epoch_time, epoch)

            # ── histograms ─────────────────────────────
            if err_mp_pct.size > 0:
                self.writer.add_histogram(
                    'val/err_m_prop_pct', err_mp_pct, epoch)
            if err_tof_pct.size > 0:
                self.writer.add_histogram(
                    'val/err_TOF_pct',    err_tof_pct, epoch)

            # ── scatter (주기적으로만) ─────────────────
            if (epoch % self.scatter_every_n_epochs == 0) or (epoch == 1):
                fig_mp = _scatter_fig(
                    y_true_orig[:, 0], y_pred_orig[:, 0],
                    label='m_prop [kg]',
                    max_points=self.scatter_max_points, log_axes=False)
                self.writer.add_figure('val/scatter_m_prop', fig_mp, epoch)

                fig_tof = _scatter_fig(
                    y_true_orig[:, 1], y_pred_orig[:, 1],
                    label='TOF [days]',
                    max_points=self.scatter_max_points, log_axes=True)
                self.writer.add_figure('val/scatter_TOF', fig_tof, epoch)

            # ── 콘솔 ──────────────────────────────────
            print(
                f"epoch {epoch:>3d}/{self.max_epochs}  "
                f"tr {tr['loss_total']:.5f}  va {va['loss_total']:.5f}  "
                f"R²(m_prop) {r2_mp:.4f}  R²(TOF) {r2_tof:.4f}  "
                f"lr {lr_now:.2e}  {epoch_time:.1f}s"
            )

            # ── 기록 history ──────────────────────────
            self.history.append({
                'epoch'             : epoch,
                'lr'                : lr_now,
                'train_loss'        : tr['loss_total'],
                'val_loss'          : va['loss_total'],
                'val_m_prop_loss'   : va['loss_m_prop'],
                'val_TOF_loss'      : va['loss_TOF'],
                'val_m_prop_R2'     : r2_mp,
                'val_TOF_R2'        : r2_tof,
                'val_m_prop_MAE_kg' : mae_mp,
                'val_TOF_MAE_days'  : mae_tof,
                'epoch_time_sec'    : epoch_time,
            })

            # ── checkpoint / early stopping ────────────
            improved = (
                va['loss_total'] < self.best_val_loss - self.es_min_delta
            )
            if improved:
                self.best_val_loss = va['loss_total']
                self.best_epoch    = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, va['loss_total'], tag='best')
            else:
                self.epochs_no_improve += 1

            # ── scheduler step ─────────────────────────
            if self.scheduler is not None:
                self.scheduler.step(va['loss_total'])

            # ── early stop ─────────────────────────────
            if self.es_enable and self.epochs_no_improve >= self.es_patience:
                print(f"[early-stop] epoch {epoch} "
                      f"(no val improvement for {self.es_patience} epochs)")
                break

        # ── 학습 끝 — test 평가 + history 저장 ─────────
        test_metrics = self.evaluate_test()
        self._save_history(test_metrics)
        self.writer.close()
        return {
            'best_epoch'    : self.best_epoch,
            'best_val_loss' : self.best_val_loss,
            'test'          : test_metrics,
            'log_dir'       : self.log_dir,
            'ckpt_dir'      : self.ckpt_dir,
        }

    # ── test 평가 ───────────────────────────────────────
    def evaluate_test(self):
        # best checkpoint 가 있으면 로드
        best_path = os.path.join(self.ckpt_dir, 'best.pt')
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(state['model_state'])

        te = self._run_epoch(self.loaders['test'], train_mode=False)
        y_true_orig = self.scaler.inverse_y(te['y_true_norm'])
        y_pred_orig = self.scaler.inverse_y(te['y_pred_norm'])
        r2_mp   = _r2(y_true_orig[:, 0], y_pred_orig[:, 0])
        r2_tof  = _r2(y_true_orig[:, 1], y_pred_orig[:, 1])
        mae_mp  = float(np.mean(np.abs(y_true_orig[:, 0] - y_pred_orig[:, 0])))
        mae_tof = float(np.mean(np.abs(y_true_orig[:, 1] - y_pred_orig[:, 1])))
        err_mp_pct  = _mean_pct_err(y_true_orig[:, 0], y_pred_orig[:, 0])
        err_tof_pct = _mean_pct_err(y_true_orig[:, 1], y_pred_orig[:, 1])

        print("-" * 70)
        print(f"[TEST] loss_total {te['loss_total']:.5f}  "
              f"R²(m_prop) {r2_mp:.4f}  R²(TOF) {r2_tof:.4f}")
        print(f"       MAE m_prop {mae_mp:.4f} kg  MAE TOF {mae_tof:.4f} day")
        if err_mp_pct.size > 0:
            print(f"       err m_prop %  mean {np.mean(err_mp_pct):+.3f}  "
                  f"std {np.std(err_mp_pct):.3f}  "
                  f"|max| {np.max(np.abs(err_mp_pct)):.2f}")
        if err_tof_pct.size > 0:
            print(f"       err TOF    %  mean {np.mean(err_tof_pct):+.3f}  "
                  f"std {np.std(err_tof_pct):.3f}  "
                  f"|max| {np.max(np.abs(err_tof_pct)):.2f}")

        # TensorBoard
        self.writer.add_scalar('test/loss_total',    te['loss_total'])
        self.writer.add_scalar('test/m_prop_R2',     r2_mp)
        self.writer.add_scalar('test/TOF_R2',        r2_tof)
        self.writer.add_scalar('test/m_prop_MAE_kg', mae_mp)
        self.writer.add_scalar('test/TOF_MAE_days',  mae_tof)

        fig_mp = _scatter_fig(
            y_true_orig[:, 0], y_pred_orig[:, 0],
            label='m_prop [kg]', max_points=self.scatter_max_points,
            log_axes=False)
        self.writer.add_figure('test/scatter_m_prop', fig_mp)
        fig_tof = _scatter_fig(
            y_true_orig[:, 1], y_pred_orig[:, 1],
            label='TOF [days]', max_points=self.scatter_max_points,
            log_axes=True)
        self.writer.add_figure('test/scatter_TOF', fig_tof)
        if err_mp_pct.size > 0:
            self.writer.add_histogram('test/err_m_prop_pct',  err_mp_pct)
        if err_tof_pct.size > 0:
            self.writer.add_histogram('test/err_TOF_pct',     err_tof_pct)

        return {
            'loss_total'      : te['loss_total'],
            'loss_m_prop'     : te['loss_m_prop'],
            'loss_TOF'        : te['loss_TOF'],
            'm_prop_R2'       : r2_mp,
            'TOF_R2'          : r2_tof,
            'm_prop_MAE_kg'   : mae_mp,
            'TOF_MAE_days'    : mae_tof,
            'm_prop_err_pct_mean' : (float(np.mean(err_mp_pct))  if err_mp_pct.size else None),
            'TOF_err_pct_mean'    : (float(np.mean(err_tof_pct)) if err_tof_pct.size else None),
            'm_prop_err_pct_abs_max': (float(np.max(np.abs(err_mp_pct)))  if err_mp_pct.size else None),
            'TOF_err_pct_abs_max'   : (float(np.max(np.abs(err_tof_pct))) if err_tof_pct.size else None),
        }

    # ── 저장 ────────────────────────────────────────────
    def _save_checkpoint(self, epoch, val_loss, tag='best'):
        path = os.path.join(self.ckpt_dir, f'{tag}.pt')
        torch.save({
            'epoch'           : epoch,
            'val_loss'        : val_loss,
            'model_state'     : self.model.state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'model_config'    : self.model.config,
        }, path)

    def _save_history(self, test_metrics):
        path = os.path.join(self.ckpt_dir, 'history.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'history'    : self.history,
                'test'       : test_metrics,
                'best_epoch' : self.best_epoch,
            }, f, indent=2)