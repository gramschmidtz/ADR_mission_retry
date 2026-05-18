"""
plot_training_result.py
=======================
학습 결과 시각화. 두 가지 모드 :

  curves   : TensorBoard event 파일에서 train/val/test loss 시계열을 읽어
             3 figure (total / m_prop / TOF) 를 그린다. 논문 Fig. 8b 의 확장.

  scatter  : 학습된 ckpt 로 train/val/test split 각각에 대해 (y_true, y_pred)
             산점도 9 panel grid (3 row × 3 col) 를 그린다. 논문 Fig. 8a 의 확장.

  both     : 위 두 모드 모두 실행 (기본).

기본 사용 (run_id 위치인자 하나만) :
  python scripts/plot_training_result.py 20260514_003839_alpha0.0

  → results/runs/20260514_003839_alpha0.0/                  (event)
    results/checkpoints/20260514_003839_alpha0.0/best.pt    (model)
    data/training_data_alpha0.0.csv                          (split 재현)
    이 모두를 자동으로 찾아서 4 figure (loss 3개 + scatter 1개) 생성.

기타 :
  # 모드 한정
  python scripts/plot_training_result.py 20260514_003839_alpha0.0 --mode curves

  # 여러 run 비교 (curves 만)
  python scripts/plot_training_result.py --mode curves \\
      --logdir results/runs/<ts1>_alpha0.0  results/runs/<ts2>_alpha1.0

  # 저장 + 창 안 띄움
  python scripts/plot_training_result.py 20260514_003839_alpha0.0 \\
      --out-dir results/figures --no-show
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ──────────────────────────────────────────────────────────────
# tfevents → dict of (epochs, values)  per tag
# ──────────────────────────────────────────────────────────────
def load_scalars(logdir):
    """
    한 run 디렉토리의 tfevents 파일들에서 모든 scalar 를 읽어
    {tag: (np.array epochs, np.array values)} dict 로 반환.

    tensorboard.backend.event_processing.event_accumulator 또는
    tensorflow.python.summary.summary_iterator 둘 다 가능하지만
    tensorboard 의 EventAccumulator 가 안정적이라 그걸 쓴다.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError:
        raise SystemExit(
            "tensorboard 미설치 — `pip install tensorboard` 후 재시도. "
            "이 스크립트는 tfevents 파일을 읽기 위해 tensorboard 의 "
            "EventAccumulator 를 사용한다.")

    if not os.path.isdir(logdir):
        raise SystemExit(f"디렉토리가 아님: {logdir}")

    # size_guidance=0 → 모두 로드 (기본은 sampling 으로 압축)
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    out = {}
    for tag in tags:
        events = ea.Scalars(tag)
        # 각 event 는 step, value, wall_time
        steps  = np.array([e.step  for e in events], dtype=np.int64)
        values = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, values)
    return out


# ──────────────────────────────────────────────────────────────
# 한 component (total/m_prop/TOF) 의 figure 그리기
# ──────────────────────────────────────────────────────────────
def _plot_one_component(ax, scalars, component_key, run_label,
                        color_train='tab:blue',
                        color_val  ='tab:green',
                        color_test ='tab:red'):
    """
    component_key ∈ {'total', 'm_prop', 'TOF'}.
    한 ax 에 train/val/test 곡선을 그린다.
    여러 run 합본을 그리려면 호출자가 run 마다 색을 바꿔서 호출.

    Returns
    -------
    dict {'best_epoch': int|None, 'best_val_loss': float|None,
          'test_loss': float|None}
    """
    train_tag = f"loss/train_{component_key}"
    val_tag   = f"loss/val_{component_key}"
    test_tag  = f"test/loss_{component_key}"

    info = {'best_epoch': None, 'best_val_loss': None, 'test_loss': None}

    # train
    if train_tag in scalars:
        ep, v = scalars[train_tag]
        ax.plot(ep, v, color=color_train, lw=1.5,
                label=f"{run_label}Train")
    # val
    if val_tag in scalars:
        ep, v = scalars[val_tag]
        ax.plot(ep, v, color=color_val, lw=1.5,
                label=f"{run_label}Validation")
        # best (val 최저점)
        i_min = int(np.argmin(v))
        best_ep   = int(ep[i_min])
        best_val  = float(v[i_min])
        info['best_epoch']    = best_ep
        info['best_val_loss'] = best_val
        ax.axvline(best_ep, color='grey', ls=':', lw=1.0,
                   label=f"{run_label}Best (epoch {best_ep})")
        ax.scatter([best_ep], [best_val], color=color_val,
                   marker='o', s=40, edgecolors='black',
                   linewidths=0.8, zorder=5)
    # test (단일 값, step=0)
    if test_tag in scalars:
        _, v = scalars[test_tag]
        if v.size > 0:
            t_loss = float(v[-1])
            info['test_loss'] = t_loss
            # 수평 점선 + 끝점 marker
            ax.axhline(t_loss, color=color_test, ls='--', lw=1.0,
                       alpha=0.8,
                       label=f"{run_label}Test = {t_loss:.4g}")

    return info


def plot_training_curves(run_scalars_list, run_labels,
                          out_dir=None, show=True,
                          y_log=True):
    """
    run_scalars_list : List[Dict[str, (epochs, values)]]
      여러 run 의 scalars. 길이 1 이면 단일 run 모드.
    run_labels       : List[str]  (figure 의 legend prefix)
    """
    components = [
        ('total',  'Total MSE'),
        ('m_prop', 'm_prop MSE'),
        ('TOF',    'TOF MSE'),
    ]

    multi_run = len(run_scalars_list) > 1
    # 색 팔레트
    train_colors = plt.cm.Blues (np.linspace(0.5, 0.95, max(2, len(run_scalars_list))))
    val_colors   = plt.cm.Greens(np.linspace(0.5, 0.95, max(2, len(run_scalars_list))))
    test_colors  = plt.cm.Reds  (np.linspace(0.5, 0.95, max(2, len(run_scalars_list))))

    saved_paths = []
    for key, ylabel in components:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        for i, (scalars, lbl) in enumerate(zip(run_scalars_list, run_labels)):
            run_prefix = (f"[{lbl}] " if multi_run else "")
            info = _plot_one_component(
                ax, scalars, key, run_prefix,
                color_train = train_colors[i] if multi_run else 'tab:blue',
                color_val   = val_colors  [i] if multi_run else 'tab:green',
                color_test  = test_colors [i] if multi_run else 'tab:red',
            )
            if not multi_run and info['best_epoch'] is not None:
                ax.annotate(
                    f"Best val MSE = {info['best_val_loss']:.4g}\n"
                    f"epoch {info['best_epoch']}",
                    xy=(info['best_epoch'], info['best_val_loss']),
                    xytext=(15, 25), textcoords='offset points',
                    fontsize=9, color='black',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='grey', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.7),
                )

        ax.set_xlabel('Epochs')
        ax.set_ylabel(f'{ylabel}  (log)' if y_log else ylabel)
        if y_log:
            ax.set_yscale('log')
        ax.set_title(f'Training performance — {ylabel}')
        ax.grid(alpha=0.3, which='both')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        fig.tight_layout()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'training_curve_{key}.png')
            fig.savefig(out_path, dpi=140, bbox_inches='tight')
            print(f"  saved : {out_path}")
            saved_paths.append(out_path)

    if show:
        plt.show()
    else:
        plt.close('all')

    return saved_paths


# ──────────────────────────────────────────────────────────────
# Scatter mode — ckpt 로 train/val/test 예측 산점도 (9 panel grid)
# ──────────────────────────────────────────────────────────────
def _r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-20:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def predict_all_splits(ckpt_dir, csv_path):
    """
    ckpt_dir 의 best.pt + scaler.npz + network.yaml 로
    train/val/test 각 split 에 대해 (y_true, y_pred) 를 계산.

    학습 시점과 동일 split 을 재현하려면 같은 seed + split 비율을 사용해야 함.
    ckpt_dir/network.yaml 에 학습 시 사용된 yaml 사본이 있으므로 그걸 사용.

    Returns
    -------
    splits_data : dict
      {'train' | 'val' | 'test' :
        {'y_true_kg', 'y_pred_kg',           # m_prop, 원 단위 [kg]
         'y_true_day', 'y_pred_day',         # TOF, 원 단위 [day]
         'y_true_norm', 'y_pred_norm'}}      # 둘 stack, 정규화 도메인 (N, 2)
    info : dict   prepare_datasets 가 반환한 split 통계
    """
    import torch
    import yaml
    from src.ann_dataset import prepare_datasets, Scaler
    from src.ann_model   import build_model_from_config

    # ckpt_dir 의 network.yaml 우선 (학습 시점 설정), 없으면 configs/network.yaml
    ncfg_path = os.path.join(ckpt_dir, 'network.yaml')
    if not os.path.exists(ncfg_path):
        ncfg_path = 'configs/network.yaml'
        print(f"  [info] ckpt_dir 에 network.yaml 사본 없음 — {ncfg_path} 사용")
    with open(ncfg_path, 'r', encoding='utf-8') as f:
        ncfg = yaml.safe_load(f)
    ncfg['data']['csv_path'] = csv_path
    seed = int(ncfg.get('data', {}).get('seed', 42))

    # 동일 split 재현
    datasets, _, info = prepare_datasets(
        csv_path=csv_path,
        split_cfg=ncfg['data']['split'],
        norm_cfg=ncfg['normalization'],
        seed=seed,
    )
    # 정규화 파라미터는 학습 때 저장된 scaler.npz 가 ground truth
    scaler = Scaler.load_npz(os.path.join(ckpt_dir, 'scaler.npz'))

    # 모델 weight 로드
    state = torch.load(os.path.join(ckpt_dir, 'best.pt'),
                       map_location='cpu', weights_only=False)
    model = build_model_from_config(ncfg['model'])
    model.load_state_dict(state['model_state'])
    model.eval()

    splits_data = {}
    for split_name, ds in datasets.items():
        Y_norm = ds.Y.numpy()
        with torch.no_grad():
            Yp_norm = model(ds.X).numpy()
        Y_orig  = scaler.inverse_y(Y_norm)
        Yp_orig = scaler.inverse_y(Yp_norm)
        splits_data[split_name] = {
            'y_true_kg'   : Y_orig[:, 0],
            'y_pred_kg'   : Yp_orig[:, 0],
            'y_true_day'  : Y_orig[:, 1],
            'y_pred_day'  : Yp_orig[:, 1],
            'y_true_norm' : Y_norm,
            'y_pred_norm' : Yp_norm,
        }
    return splits_data, info


def _scatter_one(ax, y_true, y_pred, label, color,
                 log_axes=False, max_points=5000, rng=None):
    """단일 변수 산점도 + y=x + R² annotation."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = len(y_true)
    if n > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        ytp = y_true[idx]; ypp = y_pred[idx]
    else:
        ytp = y_true; ypp = y_pred

    ax.scatter(ytp, ypp, s=3, alpha=0.3, color=color)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], 'r-', lw=1, alpha=0.7)
    if log_axes:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.set_xlabel(f'target {label}')
    ax.set_ylabel(f'output {label}')

    r2 = _r2(y_true, y_pred)
    ax.text(0.04, 0.96, f"R² = {r2:.4f}\nn = {n}",
            transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='grey', alpha=0.85))


def _scatter_total(ax, y_true_norm, y_pred_norm,
                   max_points=5000, rng=None):
    """
    정규화 도메인의 (m_prop, TOF) 를 한 panel 에 색 구분으로 합침.
    논문 §3.2 식 (29) 의 MSE 가 이 도메인에서 정의 — total panel 의 자연스러운 표현.
    """
    yt = np.asarray(y_true_norm)
    yp = np.asarray(y_pred_norm)
    n = yt.shape[0]
    if n > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        ytp = yt[idx]; ypp = yp[idx]
    else:
        ytp = yt; ypp = yp

    ax.scatter(ytp[:, 0], ypp[:, 0], s=3, alpha=0.3,
               color='tab:blue', label='m_prop (norm)')
    ax.scatter(ytp[:, 1], ypp[:, 1], s=3, alpha=0.3,
               color='tab:orange', label='TOF (norm)')

    all_t = yt.reshape(-1)
    all_p = yp.reshape(-1)
    lo = float(min(all_t.min(), all_p.min()))
    hi = float(max(all_t.max(), all_p.max()))
    ax.plot([lo, hi], [lo, hi], 'r-', lw=1, alpha=0.7)
    ax.grid(alpha=0.3)
    ax.set_xlabel('target (normalized)')
    ax.set_ylabel('output (normalized)')
    ax.legend(loc='lower right', fontsize=8)

    r2_all = _r2(all_t, all_p)
    ax.text(0.04, 0.96, f"R² (all) = {r2_all:.4f}\nn = {n}",
            transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='grey', alpha=0.85))


def plot_scatter_grid(splits_data, max_points=5000,
                      save_path=None, show=True):
    """
    3 row (train/val/test) × 3 col (m_prop/TOF/total normalized) scatter grid.
    """
    rng = np.random.default_rng(0)
    row_order  = ['train', 'val', 'test']
    row_colors = {'train': 'tab:blue', 'val': 'tab:green', 'test': 'tab:red'}

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 13))
    for r, split in enumerate(row_order):
        if split not in splits_data:
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"{split}: n/a",
                                ha='center', va='center')
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            continue
        d = splits_data[split]
        c_col = row_colors[split]

        # col 0 : m_prop [kg]
        _scatter_one(axes[r, 0], d['y_true_kg'], d['y_pred_kg'],
                     label='m_prop [kg]', color=c_col,
                     log_axes=False, max_points=max_points, rng=rng)
        # col 1 : TOF [day], log-log
        _scatter_one(axes[r, 1], d['y_true_day'], d['y_pred_day'],
                     label='TOF [day]', color=c_col,
                     log_axes=True, max_points=max_points, rng=rng)
        # col 2 : total (normalized)
        _scatter_total(axes[r, 2], d['y_true_norm'], d['y_pred_norm'],
                       max_points=max_points, rng=rng)

        # row label
        axes[r, 0].annotate(
            split.upper(),
            xy=(-0.22, 0.5), xycoords='axes fraction',
            fontsize=14, fontweight='bold',
            color=c_col, rotation=90,
            ha='center', va='center',
        )

    col_titles = ['m_prop [kg]', 'TOF [day, log-log]',
                  'total (normalized)']
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12, fontweight='bold')

    fig.suptitle('ANN regression : output vs target',
                 fontsize=13, y=1.00)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"  saved : {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _resolve_paths(run_id, results_root='results', data_root='data'):
    """
    run_id (예: '20260514_003839_alpha0.0') 로부터 표준 경로 추론.
    여러 forms 를 지원 :
      - 단순 run_id : 'TS_alphaX.Y'           → 표준 results/runs, checkpoints, data 사용
      - 전체 경로  : 'results/runs/TS_alphaX' → 그대로 사용 + 다른 경로도 추론

    Returns
    -------
    dict {'run_id', 'logdir', 'ckpt_dir', 'csv', 'alpha_tag'}
        해당 파일이 존재하지 않으면 그 키의 값은 None.
    """
    import re

    # 사용자가 전체 경로를 줬을 수도 있음 — 그러면 basename 으로 정규화
    run_id_clean = os.path.basename(run_id.rstrip('/\\'))

    # alpha tag 추출 (예 : _alpha0.0)
    m = re.search(r'_alpha([0-9]+(?:\.[0-9]+)?)', run_id_clean)
    alpha_tag = m.group(1) if m else None

    logdir   = os.path.join(results_root, 'runs',        run_id_clean)
    ckpt_dir = os.path.join(results_root, 'checkpoints', run_id_clean)
    csv      = (os.path.join(data_root, f'training_data_alpha{alpha_tag}.csv')
                if alpha_tag is not None else None)

    return {
        'run_id'    : run_id_clean,
        'alpha_tag' : alpha_tag,
        'logdir'    : logdir   if os.path.isdir(logdir)   else None,
        'ckpt_dir'  : ckpt_dir if os.path.isdir(ckpt_dir) else None,
        'csv'       : csv      if (csv and os.path.exists(csv)) else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="학습 결과 시각화. run_id 만 주면 loss 곡선 (Fig. 8b) "
                    "3개 + 산점도 grid (Fig. 8a) 1개 = 4 figure 자동 생성.")
    ap.add_argument(
        'run_id', nargs='?', default=None,
        help='run identifier (예: 20260514_003839_alpha0.0). '
             'results/runs / results/checkpoints / data/ 의 표준 경로에서 '
             '필요한 파일을 자동으로 찾는다. '
             '여러 run 비교가 필요하면 --logdir 로 명시.',
    )
    ap.add_argument('--mode', choices=['curves', 'scatter', 'both'],
                    default='both',
                    help="기본 both (curves + scatter). 일부만 그리려면 지정.")
    # override 옵션 (고급 사용 — 보통 필요 없음)
    ap.add_argument('--logdir', nargs='+', default=None,
                    help='[override] curves mode 의 run 디렉토리들')
    ap.add_argument('--ckpt-dir', default=None,
                    help='[override] scatter mode 의 ckpt 폴더')
    ap.add_argument('--csv', default=None,
                    help='[override] scatter mode 의 csv 파일')
    ap.add_argument('--labels', nargs='*', default=None,
                    help='curves : 여러 run 비교 시 legend label')
    ap.add_argument('--linear-y', action='store_true',
                    help='curves : y 축 linear (기본 log)')
    ap.add_argument('--max-points', type=int, default=5000,
                    help='scatter : panel 당 sample 상한 (기본 5000)')
    ap.add_argument('--out-dir', default=None,
                    help='PNG 저장 디렉토리. 미지정 시 저장 안 함.')
    ap.add_argument('--no-show', action='store_true',
                    help='matplotlib 창 띄우기 끔')
    args = ap.parse_args()

    mode = args.mode

    # ── run_id 가 주어지면 표준 경로 자동 추론 ─────────────
    resolved = None
    if args.run_id:
        resolved = _resolve_paths(args.run_id)
        # override 가 없는 경우만 채움
        if args.logdir is None and resolved['logdir']:
            args.logdir = [resolved['logdir']]
        if args.ckpt_dir is None and resolved['ckpt_dir']:
            args.ckpt_dir = resolved['ckpt_dir']
        if args.csv is None and resolved['csv']:
            args.csv = resolved['csv']

    print("=" * 78)
    print(f"  plot_training_result : mode = {mode}")
    if resolved:
        print(f"  run_id    : {resolved['run_id']}")
        print(f"  alpha tag : {resolved['alpha_tag']}")
        print(f"  logdir    : {resolved['logdir']  or '(찾지 못함)'}")
        print(f"  ckpt-dir  : {resolved['ckpt_dir']or '(찾지 못함)'}")
        print(f"  csv       : {resolved['csv']    or '(찾지 못함)'}")
    print("=" * 78)

    # ─── curves mode ─────────────────────────────────────
    if mode in ('curves', 'both'):
        if not args.logdir:
            if mode == 'curves':
                raise SystemExit(
                    "--mode curves : --logdir 또는 run_id 위치인자 필요.")
            print("  [curves] logdir 없음 — curves 건너뜀.")
        else:
            if args.labels and len(args.labels) != len(args.logdir):
                raise SystemExit(
                    f"--labels 개수가 --logdir 와 다름 "
                    f"({len(args.labels)} vs {len(args.logdir)})")
            labels = args.labels or [os.path.basename(p.rstrip('/\\'))
                                      for p in args.logdir]
            run_scalars_list = []
            for path, lbl in zip(args.logdir, labels):
                print(f"  [curves] loading : {path}  ({lbl})")
                scalars = load_scalars(path)
                if not scalars:
                    print(f"    [WARN] scalar tag 없음")
                else:
                    tags_present = ', '.join(sorted(scalars.keys())[:5])
                    print(f"    tags : {len(scalars)} 개 (예: {tags_present} ...)")
                run_scalars_list.append(scalars)
            print("-" * 78)
            plot_training_curves(
                run_scalars_list, labels,
                out_dir=args.out_dir,
                show=(not args.no_show),
                y_log=(not args.linear_y),
            )

    # ─── scatter mode ────────────────────────────────────
    if mode in ('scatter', 'both'):
        if not args.ckpt_dir or not args.csv:
            if mode == 'scatter':
                raise SystemExit(
                    "--mode scatter : --ckpt-dir 와 --csv (또는 run_id) 필요.")
            print("  [scatter] ckpt-dir 또는 csv 없음 — scatter 건너뜀.")
        else:
            if not os.path.isdir(args.ckpt_dir):
                raise SystemExit(f"ckpt-dir 가 디렉토리가 아님: {args.ckpt_dir}")
            if not os.path.exists(args.csv):
                raise SystemExit(f"csv 없음: {args.csv}")

            print(f"  [scatter] ckpt : {args.ckpt_dir}")
            print(f"  [scatter] csv  : {args.csv}")
            splits_data, info = predict_all_splits(args.ckpt_dir, args.csv)
            print(f"    N_total={info['n_total']}  "
                  f"train={info['n_train']}  val={info['n_val']}  "
                  f"test={info['n_test']}")
            print("-" * 78)

            save_path = None
            if args.out_dir:
                save_path = os.path.join(
                    args.out_dir, 'regression_scatter_3x3.png')
            plot_scatter_grid(
                splits_data,
                max_points=args.max_points,
                save_path=save_path,
                show=(not args.no_show),
            )

    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)