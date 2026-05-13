"""
run_sequence_search.py
======================
논문 §4 의 SS-ANN sequence search 실행.

기능 :
  1. configs/random5000debris.yaml 의 5000 debris 에 대해 beam search.
  2. 최장 sequence 의 step 별 (debris, ANN-예측 m_prop, ANN-예측 TOF) 를 csv 로 저장.
  3. 콘솔에 sequence 요약 + step 별 ANN 예측값 출력.
  4. **그래프는 그리지 않는다.** Fig. 11 형식 시각화는 별도 스크립트
     scripts/evaluate_sequence.py 에서 (solver 로 재평가 + 오차율 포함) 처리.

α 별로 따로 실행:
  python scripts/run_sequence_search.py --alpha 0.0 --mode ann \
      --ckpt-dir results/checkpoints/20260514_003839_alpha0.0
  python scripts/run_sequence_search.py --alpha 0.0 --mode solver
"""

import os
import sys
import argparse
import csv
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader   import load_params
from src.sequence_search import (
    SequenceSearch, build_evaluator, DAY, T_STAY_SEC,
)


# ──────────────────────────────────────────────────────────────
# Sequence 결과 → csv (step 별 ANN 예측값 기록)
# ──────────────────────────────────────────────────────────────
def save_sequence_csv(seq, darr, out_path):
    cols = [
        'step',
        'src_idx', 'dst_idx', 'src_name', 'dst_name',
        'src_alt_km', 'dst_alt_km',
        't_start_day', 't_end_day', 't_capture_end_day',
        'ann_TOF_day', 'ann_m_prop_kg', 'cum_ann_m_prop_kg',
    ]
    cum = 0.0
    rows = []
    for i, step in enumerate(seq.steps, start=1):
        cum += step.m_prop_kg
        rows.append({
            'step'              : i,
            'src_idx'           : step.src_idx,
            'dst_idx'           : step.dst_idx,
            'src_name'          : darr.names[step.src_idx],
            'dst_name'          : darr.names[step.dst_idx],
            'src_alt_km'        : f"{step.src_alt_km:.3f}",
            'dst_alt_km'        : f"{step.dst_alt_km:.3f}",
            't_start_day'       : f"{step.t_start_s / DAY:.3f}",
            't_end_day'         : f"{step.t_end_s   / DAY:.3f}",
            't_capture_end_day' : f"{step.t_capture_end_s / DAY:.3f}",
            'ann_TOF_day'       : f"{step.TOF_s / DAY:.3f}",
            'ann_m_prop_kg'     : f"{step.m_prop_kg:.6f}",
            'cum_ann_m_prop_kg' : f"{cum:.6f}",
        })

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_sequence_summary(seq, darr, search):
    """콘솔에 step 별 ANN 예측 정리."""
    print()
    print("=" * 88)
    print(f"  BEST sequence : depth = {seq.depth} debris")
    print(f"    m_prop used = {seq.m_prop_used_kg:.3f} kg  "
          f"(잔여 {search.m_initial - seq.m_prop_used_kg - search.m_dry:.2f} kg)")
    print(f"    total TOF   = {seq.total_TOF_s / DAY:.2f} day  "
          f"(= {seq.total_TOF_s / DAY / 365.25:.3f} year)")
    print(f"    total time  = {seq.total_time_s / DAY:.2f} day  "
          f"(= {seq.total_time_s / DAY / 365.25:.3f} year, capture 포함)")
    print("=" * 88)
    print()
    print(f"  step  src→dst (idx)         src_alt   dst_alt    "
          f"ANN_TOF[day]  ANN_m_prop[kg]   cum[kg]")
    cum = 0.0
    for i, st in enumerate(seq.steps, start=1):
        cum += st.m_prop_kg
        print(f"  {i:>3d}   {st.src_idx:>5d}→{st.dst_idx:<5d}            "
              f"{st.src_alt_km:>7.2f}  {st.dst_alt_km:>7.2f}    "
              f"{st.TOF_s/DAY:>10.2f}    {st.m_prop_kg:>10.4f}   "
              f"{cum:>7.3f}")
    print()


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SS-ANN sequence search (논문 §4)")
    ap.add_argument('--debris-yaml', default='configs/random5000debris.yaml')
    ap.add_argument('--alpha', type=float, required=True,
                    help='alpha 값 (출력 파일명에 사용, solver 모드의 cost 가중치)')
    ap.add_argument('--mode', choices=['ann', 'solver'], default='ann')
    ap.add_argument('--ckpt-dir', default=None,
                    help="ANN 모드일 때 best.pt / scaler.npz 가 있는 폴더")
    ap.add_argument('--beam-width', type=int, default=100)
    ap.add_argument('--T-max-year', type=float, default=10.0)
    ap.add_argument('--starting-subsample', type=int, default=None,
                    help="첫 depth 의 시작점 수 (디버깅용. None = 모두)")
    ap.add_argument('--starting-seed', type=int, default=42)
    ap.add_argument('--out-prefix', default=None,
                    help="출력 파일 prefix. 기본: results/sequences/seq_alpha{α}_{mode}_{ts}")
    args = ap.parse_args()

    params, sim_cfg = load_params(
        constants_path='configs/constants.yaml',
        simulation_path='configs/simulation.yaml')

    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_prefix = (f"results/sequences/seq_alpha{args.alpha}_"
                      f"{args.mode}_{ts}")

    print("=" * 88)
    print(f"  SS-ANN sequence search  (α = {args.alpha}, mode = {args.mode})")
    print("=" * 88)
    if args.mode == 'ann':
        if not args.ckpt_dir:
            raise SystemExit(
                "--mode ann 이면 --ckpt-dir 가 필요합니다. "
                "예: --ckpt-dir results/checkpoints/<timestamp>_alpha0.0")
        evaluator = build_evaluator(
            mode='ann', params=params, ckpt_dir=args.ckpt_dir)
        print(f"  ANN  ckpt   : {args.ckpt_dir}")
        print(f"  ANN  device : {evaluator.device}")
    else:
        evaluator = build_evaluator(
            mode='solver', params=params, alpha=args.alpha)
        print(f"  solver h_P grid : "
              f"[{evaluator.h_P_grid_km[0]:.0f}, "
              f"{evaluator.h_P_grid_km[-1]:.0f}] km")
    print(f"  debris yaml      : {args.debris_yaml}")
    print(f"  beam_width N_S   : {args.beam_width}")
    print(f"  T_max            : {args.T_max_year} year")
    print(f"  t_stay           : {T_STAY_SEC/DAY:.1f} day")
    print(f"  starting subsample : "
          f"{args.starting_subsample if args.starting_subsample else 'all (= N)'}")
    print(f"  out prefix       : {out_prefix}")
    print("-" * 88)

    search = SequenceSearch(
        debris_yaml_path = args.debris_yaml,
        params           = params,
        evaluator        = evaluator,
        beam_width       = args.beam_width,
        mission_max_year = args.T_max_year,
        t_stay_s         = T_STAY_SEC,
        starting_subsample = args.starting_subsample,
        starting_seed    = args.starting_seed,
        verbose          = True,
    )
    sequences = search.run()

    if not sequences:
        print("[ERROR] 유효한 sequence 가 없습니다.")
        return 1

    best = sequences[0]
    print_sequence_summary(best, search.darr, search)

    csv_path = out_prefix + '.csv'
    save_sequence_csv(best, search.darr, csv_path)
    print(f"  sequence csv : {csv_path}")
    print("=" * 88)
    print()
    print("Fig. 11 형식 시각화 + solver 재평가 + 오차율 분석을 보려면:")
    print(f"  python scripts/evaluate_sequence.py --seq-csv {csv_path} "
          f"--alpha {args.alpha}")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)