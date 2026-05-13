"""
evaluate_sequence.py
====================
run_sequence_search.py 가 만든 sequence csv 를 받아서 :

  (1) 각 step 의 transfer 를 src.transfer_grid 의 솔버로 **재평가**.
      Solver 는 phase 별 시간 (T1, T2a, Tp, T2b) 과 최적 phasing 고도 h_P
      를 모두 알려주므로, 시간축에 펼친 altitude profile 을 그릴 수 있다.
  (2) ANN 예측값과 solver 정답 사이의 오차율 (m_prop, TOF) 콘솔 출력.
  (3) 논문 Fig. 11 형식의 dual-axis figure 저장 + (옵션) show :
      좌측 y = altitude [km],  우측 y = cumulative m_prop [kg].

m_SC 정책 :
  각 step 의 m_SC 는 **ANN 의 누적 m_prop** 으로 산출한 값을 사용한다
  (sequence search 가 결정한 chaser 질량 trajectory 와 동일). 이렇게 하면
  같은 입력에서의 ANN 예측 ↔ solver 정답 직접 비교가 됨.

CSV 입력 컬럼 (run_sequence_search.py 가 만든 형식) :
  step, src_idx, dst_idx, src_name, dst_name,
  src_alt_km, dst_alt_km, t_start_day, t_end_day, t_capture_end_day,
  ann_TOF_day, ann_m_prop_kg, cum_ann_m_prop_kg

사용 :
  python scripts/evaluate_sequence.py \
      --seq-csv results/sequences/seq_alpha0.0_ann_20260514_010000.csv \
      --alpha 0.0
"""

import os
import sys
import argparse
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader   import load_params, load_debris_list
from src.transfer_grid   import compute_transfer_grid
from src.transfer_solver import raan_drift_rate


DAY        = 86400.0
T_STAY_SEC = 30.0 * DAY


# ──────────────────────────────────────────────────────────────
# csv 로드
# ──────────────────────────────────────────────────────────────
def load_sequence_csv(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'step'             : int(r['step']),
                'src_idx'          : int(r['src_idx']),
                'dst_idx'          : int(r['dst_idx']),
                'src_name'         : r['src_name'],
                'dst_name'         : r['dst_name'],
                'src_alt_km'       : float(r['src_alt_km']),
                'dst_alt_km'       : float(r['dst_alt_km']),
                't_start_day'      : float(r['t_start_day']),
                't_end_day'        : float(r['t_end_day']),
                't_capture_end_day': float(r['t_capture_end_day']),
                'ann_TOF_day'      : float(r['ann_TOF_day']),
                'ann_m_prop_kg'    : float(r['ann_m_prop_kg']),
                'cum_ann_m_prop_kg': float(r['cum_ann_m_prop_kg']),
            })
    return rows


# ──────────────────────────────────────────────────────────────
# RAAN propagation 도우미
# ──────────────────────────────────────────────────────────────
def _raan_at(deg_initial, omega_dot_rad_per_s, t_s):
    """Ω(t) [deg] = Ω_0 + Ω̇ × t. mod 360 정규화."""
    return (deg_initial + np.rad2deg(omega_dot_rad_per_s * t_s)) % 360.0


# ──────────────────────────────────────────────────────────────
# Solver 재평가
# ──────────────────────────────────────────────────────────────
def reevaluate_with_solver(seq_rows, debris_dict, params, alpha):
    """
    각 step 에 대해 compute_transfer_grid 호출.
    m_SC 는 'ANN 의 누적 m_prop' 으로 계산한 값을 사용.

    Returns
    -------
    list of dict, 각 step 의 solver 결과 + ANN 비교.
        keys :
          step, src_idx, dst_idx,
          src_alt_km, dst_alt_km,
          t_start_s, m_SC_kg,
          # ANN 예측 (csv 원본값)
          ann_TOF_s, ann_m_prop_kg,
          # solver 정답
          sol_TOF_s, sol_m_prop_kg, sol_h_P_km,
          # solver phase 시간 [s] 절대 시각
          t_T1_end_s, t_T2a_end_s, t_Tp_end_s, t_T2b_end_s,
          t_capture_end_s,
          # 오차
          err_m_prop_pct, err_TOF_pct,
    """
    h_disp = params['disposal_alt_km']
    h_min  = max(h_disp, params['h_P_min_km'])
    h_max  = params['h_P_max_km']
    h_step = params['h_step_km']
    h_P_grid = np.arange(h_min, h_max + h_step / 2, h_step)
    m0 = params['m0']
    Re_m = params['Re']
    i_rad = np.deg2rad(params['inclination_deg'])

    # debris 의 RAAN drift 사전 계산
    names_list = list(debris_dict.keys())
    name_to_idx = {n: i for i, n in enumerate(names_list)}

    # 우리는 src_idx / dst_idx 로 접근 — debris yaml 의 키 순서가 sequence search
    # 에서 사용된 인덱스와 동일하다고 가정 (run_sequence_search.py 의 DebrisArray 와 일치)
    #
    # 시각은 solver TOF 누적 기준으로 재구성한다.
    #   csv 의 t_start_day 는 ANN TOF 누적으로 만들어진 값인데, ANN 과 solver TOF 가
    #   step 마다 다르면 (보통 1-5%) 누적되면서 phase 시각과 다음 step 시작 시각이
    #   어긋난다. 시각화에서 평평한 gap 또는 overlap 으로 보이게 됨.
    #   evaluate_sequence 는 solver 가 정답이므로 solver 의 시간 누적으로 figure 시각을
    #   구성한다.
    results = []
    t_cumulative_s = 0.0   # solver 기준 누적 시각 (이번 step 시작)
    for row in seq_rows:
        src_idx = row['src_idx']
        dst_idx = row['dst_idx']
        src_name = names_list[src_idx]
        dst_name = names_list[dst_idx]
        d_src = debris_dict[src_name]
        d_dst = debris_dict[dst_name]

        # 시각의 두 종류 :
        #   t_start_csv : ANN 누적 기준 (RAAN propagation 에 사용 — sequence search
        #                 가 ANN 기준 시각에 m_SC 등을 결정했으니 일관성 유지).
        #   t_start_sol : solver 누적 기준 (figure 시각화에 사용).
        t_start_csv = row['t_start_day'] * DAY
        t_start_sol = t_cumulative_s

        # 각 debris 의 ANN 기준 RAAN
        a_src = Re_m + d_src['alt0_km'] * 1000.0
        a_dst = Re_m + d_dst['alt0_km'] * 1000.0
        om_src = raan_drift_rate(a_src, d_src['e'], np.deg2rad(d_src['i']), params)
        om_dst = raan_drift_rate(a_dst, d_dst['e'], np.deg2rad(d_dst['i']), params)
        RAAN_src_now = _raan_at(d_src['RAAN'], om_src, t_start_csv)
        RAAN_dst_now = _raan_at(d_dst['RAAN'], om_dst, t_start_csv)

        # m_SC : sequence search 의 누적 ann m_prop 직전 시점
        cum_before = row['cum_ann_m_prop_kg'] - row['ann_m_prop_kg']
        m_SC = m0 - cum_before

        # solver 호출
        m_prop_s, TOF_s_total, info = compute_transfer_grid(
            h_D1        = d_src['alt0_km'],
            RAAN_D1_deg = RAAN_src_now,
            m_D1        = d_src['mass'],
            h_D2        = d_dst['alt0_km'],
            RAAN_D2_deg = RAAN_dst_now,
            m_SC        = m_SC,
            params      = params,
            alpha       = alpha,
            h_P_grid_km = h_P_grid,
        )

        if not info.get('feasible', False):
            # solver 가 infeasible 이라고 판단 → ANN 만으로 진행한 sequence 가
            # 사실은 invalid 일 가능성. 경고하고 ANN 값 그대로 대체.
            print(f"  [WARN] step {row['step']}: solver infeasible. ANN 값으로 대체.")
            T1 = T2a = Tp = T2b = 0.0
            sol_TOF = row['ann_TOF_day'] * DAY
            sol_mp  = row['ann_m_prop_kg']
            sol_hP  = float('nan')
            mp_T1 = mp_T2a = mp_T2b = mp_drag = 0.0
        else:
            T1  = info['T1']
            T2a = info['T2a']
            Tp  = info['Tp']
            T2b = info['T2b']
            sol_TOF = info['TOF'] - info['Ts']   # Ts (capture) 는 별도 처리
            sol_mp  = info['m_prop']
            sol_hP  = info['h_P_km']
            # phase 별 추진제 (Fig. 11 m_prop curve 정확화에 사용)
            mp_T1   = info.get('m_prop_T1',   0.0)
            mp_T2a  = info.get('m_prop_T2a',  0.0)
            mp_T2b  = info.get('m_prop_T2b',  0.0)
            mp_drag = info.get('m_prop_drag', 0.0)

        # 절대 시각 (solver 누적 기준)
        t_T1  = t_start_sol + T1
        t_T2a = t_T1        + T2a
        t_Tp  = t_T2a       + Tp
        t_T2b = t_Tp        + T2b
        t_cap_end = t_T2b + T_STAY_SEC

        # 다음 step 의 시작 시각 = 이번 step capture 끝
        t_cumulative_s = t_cap_end

        # 오차 (solver = ground truth)
        ann_mp  = row['ann_m_prop_kg']
        ann_TOF = row['ann_TOF_day'] * DAY
        err_mp  = (100.0 * (ann_mp  - sol_mp)  / sol_mp ) if sol_mp  > 0 else float('nan')
        err_TOF = (100.0 * (ann_TOF - sol_TOF) / sol_TOF) if sol_TOF > 0 else float('nan')

        results.append({
            'step'         : row['step'],
            'src_idx'      : src_idx,
            'dst_idx'      : dst_idx,
            'src_name'     : src_name,
            'dst_name'     : dst_name,
            'src_alt_km'   : d_src['alt0_km'],
            'dst_alt_km'   : d_dst['alt0_km'],
            't_start_s'    : t_start_sol,  # solver 기준 시각으로 저장
            't_start_csv_s': t_start_csv,  # 참고용 (ANN 기준)
            'm_SC_kg'      : m_SC,
            'ann_TOF_s'    : ann_TOF,
            'ann_m_prop_kg': ann_mp,
            'sol_TOF_s'    : sol_TOF,
            'sol_m_prop_kg': sol_mp,
            'sol_h_P_km'   : sol_hP,
            't_T1_end_s'   : t_T1,
            't_T2a_end_s'  : t_T2a,
            't_Tp_end_s'   : t_Tp,
            't_T2b_end_s'  : t_T2b,
            't_capture_end_s': t_cap_end,
            # phase 별 추진제 [kg] (T1 + T2a + drag + T2b = sol_m_prop_kg)
            'sol_mp_T1'    : mp_T1,
            'sol_mp_T2a'   : mp_T2a,
            'sol_mp_drag'  : mp_drag,
            'sol_mp_T2b'   : mp_T2b,
            'err_m_prop_pct'  : err_mp,
            'err_TOF_pct'     : err_TOF,
        })

    return results


# ──────────────────────────────────────────────────────────────
# Plot — Fig. 11 형식
# ──────────────────────────────────────────────────────────────
def _build_altitude_curve(steps, params):
    """
    각 step 의 phase 시각으로 (t [day], h [km]) 곡선 만들기.
    구성 :
        [t_start, src_alt]
        [t_T1_end, disposal_alt]
        [t_T2a_end, h_P]
        [t_Tp_end,  h_P]
        [t_T2b_end, dst_alt]
        [t_capture_end, dst_alt]
    """
    h_disp = float(params['disposal_alt_km'])
    if not steps:
        return np.zeros(0), np.zeros(0)
    t_pts = [steps[0]['t_start_s']]
    h_pts = [steps[0]['src_alt_km']]
    for s in steps:
        hP = s['sol_h_P_km']
        if not np.isfinite(hP):
            hP = h_disp
        t_pts.extend([s['t_T1_end_s'], s['t_T2a_end_s'],
                      s['t_Tp_end_s'], s['t_T2b_end_s'],
                      s['t_capture_end_s']])
        h_pts.extend([h_disp, hP, hP, s['dst_alt_km'], s['dst_alt_km']])
    return np.asarray(t_pts) / DAY, np.asarray(h_pts)


def _build_cum_mprop_curve(steps, use='sol'):
    """누적 추진제 곡선.
    use='sol' 이면 phase 별 (T1, T2a, drag, T2b) 정확한 누적을 그린다.
    use='ann' 이면 step 합 (ANN 출력은 phase 분해 없음) → T2b 끝에 한 번에 더해짐.

    물리 :
      - T1, T2a, T2b 구간 : 추력 ON → propellant_consumed_eq10 으로 큰 양 소비
      - Tp 구간 (phasing) : 추력 OFF, 대기항력 보정만 → 매우 작은 양 소비 (drag)
      - capture (t_stay) : 추력 OFF, drag 없음 → 0
    """
    if not steps:
        return np.zeros(0), np.zeros(0)
    t_pts = [steps[0]['t_start_s']]
    m_pts = [0.0]
    cum = 0.0
    for s in steps:
        # transfer 시작 시점 — 직전 cum 유지 (capture 평평 끝 점과 같음)
        t_pts.append(s['t_start_s']); m_pts.append(cum)

        if use == 'sol':
            # T1: t_start → t_T1_end, +mp_T1
            cum += s.get('sol_mp_T1', 0.0)
            t_pts.append(s['t_T1_end_s']);  m_pts.append(cum)
            # T2a: t_T1_end → t_T2a_end, +mp_T2a
            cum += s.get('sol_mp_T2a', 0.0)
            t_pts.append(s['t_T2a_end_s']); m_pts.append(cum)
            # Tp (phasing, drag): t_T2a_end → t_Tp_end, +mp_drag
            cum += s.get('sol_mp_drag', 0.0)
            t_pts.append(s['t_Tp_end_s']);  m_pts.append(cum)
            # T2b: t_Tp_end → t_T2b_end, +mp_T2b
            cum += s.get('sol_mp_T2b', 0.0)
            t_pts.append(s['t_T2b_end_s']); m_pts.append(cum)
        else:
            # ANN 은 phase 분해 없음 → step 전체 m_prop 을 T2b 끝에 한 번에
            cum += s['ann_m_prop_kg']
            t_pts.append(s['t_T2b_end_s']); m_pts.append(cum)

        # capture 평평
        t_pts.append(s['t_capture_end_s']); m_pts.append(cum)

    return np.asarray(t_pts) / DAY, np.asarray(m_pts)


def plot_fig11(steps, params, title, out_png=None, show=False,
               overlay_ann=True):
    fig, ax_h = plt.subplots(figsize=(11, 4.5))

    t_h, h_h = _build_altitude_curve(steps, params)
    ax_h.plot(t_h, h_h, color='tab:blue', lw=1.2, label='Altitude (solver)')
    ax_h.set_xlabel('Time, day')
    ax_h.set_ylabel('Altitude, km', color='tab:blue')
    ax_h.tick_params(axis='y', labelcolor='tab:blue')
    ax_h.grid(alpha=0.3)
    ax_h.set_title(title)

    ax_m = ax_h.twinx()
    t_m, m_m = _build_cum_mprop_curve(steps, use='sol')
    ax_m.plot(t_m, m_m, color='tab:orange', lw=1.4, label='m_prop (solver)')
    ax_m.set_ylabel('m_prop, kg', color='tab:orange')
    ax_m.tick_params(axis='y', labelcolor='tab:orange')

    if overlay_ann:
        t_a, m_a = _build_cum_mprop_curve(steps, use='ann')
        ax_m.plot(t_a, m_a, color='tab:orange', lw=1.0, ls='--', alpha=0.7,
                  label='m_prop (ANN)')

    fig.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        fig.savefig(out_png, dpi=140, bbox_inches='tight')
        print(f"  figure saved : {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 오차 요약 출력
# ──────────────────────────────────────────────────────────────
def print_error_summary(steps):
    print()
    print("=" * 100)
    print(f"  step 별 ANN ↔ solver 비교  (err% = (ann − sol) / sol × 100)")
    print("=" * 100)
    print(f"  step  src→dst              "
          f"ANN_m_prop   sol_m_prop   err%      "
          f"ANN_TOF[d]    sol_TOF[d]    err%      h_P[km]")
    for s in steps:
        print(f"  {s['step']:>3d}   "
              f"{s['src_idx']:>5d}→{s['dst_idx']:<5d}        "
              f"{s['ann_m_prop_kg']:>9.4f}   {s['sol_m_prop_kg']:>9.4f}  "
              f"{s['err_m_prop_pct']:>+7.2f}%   "
              f"{s['ann_TOF_s']/DAY:>10.2f}   {s['sol_TOF_s']/DAY:>10.2f}  "
              f"{s['err_TOF_pct']:>+7.2f}%   "
              f"{s['sol_h_P_km']:>6.1f}")
    # 통계
    err_mp  = np.array([s['err_m_prop_pct'] for s in steps
                        if np.isfinite(s['err_m_prop_pct'])])
    err_TOF = np.array([s['err_TOF_pct']    for s in steps
                        if np.isfinite(s['err_TOF_pct'])])
    print("-" * 100)
    cum_ann = sum(s['ann_m_prop_kg'] for s in steps)
    cum_sol = sum(s['sol_m_prop_kg'] for s in steps)
    cum_ann_TOF = sum(s['ann_TOF_s'] for s in steps) / DAY
    cum_sol_TOF = sum(s['sol_TOF_s'] for s in steps) / DAY
    print(f"  누적 m_prop  : ANN {cum_ann:.3f} kg   sol {cum_sol:.3f} kg   "
          f"err {100*(cum_ann-cum_sol)/cum_sol:+.3f}%")
    print(f"  누적 TOF     : ANN {cum_ann_TOF:.2f} day   "
          f"sol {cum_sol_TOF:.2f} day   "
          f"err {100*(cum_ann_TOF-cum_sol_TOF)/cum_sol_TOF:+.3f}%")
    if err_mp.size > 0:
        print(f"  step err m_prop %  :  "
              f"mean {err_mp.mean():+.3f}   std {err_mp.std():.3f}   "
              f"|max| {np.abs(err_mp).max():.3f}")
    if err_TOF.size > 0:
        print(f"  step err TOF    %  :  "
              f"mean {err_TOF.mean():+.3f}   std {err_TOF.std():.3f}   "
              f"|max| {np.abs(err_TOF).max():.3f}")

    # phase 별 추진제 분해 (solver)
    sum_T1   = sum(s.get('sol_mp_T1',   0.0) for s in steps)
    sum_T2a  = sum(s.get('sol_mp_T2a',  0.0) for s in steps)
    sum_drag = sum(s.get('sol_mp_drag', 0.0) for s in steps)
    sum_T2b  = sum(s.get('sol_mp_T2b',  0.0) for s in steps)
    print(f"  phase 별 m_prop (solver, 합산) :")
    print(f"     T1   (D_src → disposal) : {sum_T1:8.3f} kg")
    print(f"     T2a  (disposal → h_P)   : {sum_T2a:8.3f} kg")
    print(f"     Tp   (phasing drag)     : {sum_drag:8.3f} kg")
    print(f"     T2b  (h_P → D_dst)      : {sum_T2b:8.3f} kg")
    print(f"     ─────────────────────────────────────")
    print(f"     총합                    : {sum_T1+sum_T2a+sum_drag+sum_T2b:8.3f} kg")
    print("=" * 100)


# ──────────────────────────────────────────────────────────────
# Step 별 결과 csv 저장
# ──────────────────────────────────────────────────────────────
def save_eval_csv(steps, out_path):
    cols = [
        'step', 'src_idx', 'dst_idx', 'src_name', 'dst_name',
        'src_alt_km', 'dst_alt_km', 't_start_day', 'm_SC_kg',
        'ann_m_prop_kg', 'sol_m_prop_kg', 'err_m_prop_pct',
        'ann_TOF_day', 'sol_TOF_day', 'err_TOF_pct',
        'sol_h_P_km',
        # phase 별 추진제 분해 (solver) [kg]
        'sol_mp_T1', 'sol_mp_T2a', 'sol_mp_drag', 'sol_mp_T2b',
        # phase 별 시간 절대시각 [day]
        't_T1_end_day', 't_T2a_end_day', 't_Tp_end_day', 't_T2b_end_day',
        't_capture_end_day',
    ]
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in steps:
            w.writerow({
                'step'         : s['step'],
                'src_idx'      : s['src_idx'],
                'dst_idx'      : s['dst_idx'],
                'src_name'     : s['src_name'],
                'dst_name'     : s['dst_name'],
                'src_alt_km'   : f"{s['src_alt_km']:.3f}",
                'dst_alt_km'   : f"{s['dst_alt_km']:.3f}",
                't_start_day'  : f"{s['t_start_s']/DAY:.3f}",
                'm_SC_kg'      : f"{s['m_SC_kg']:.4f}",
                'ann_m_prop_kg': f"{s['ann_m_prop_kg']:.6f}",
                'sol_m_prop_kg': f"{s['sol_m_prop_kg']:.6f}",
                'err_m_prop_pct': (f"{s['err_m_prop_pct']:+.4f}"
                                    if np.isfinite(s['err_m_prop_pct']) else ''),
                'ann_TOF_day'  : f"{s['ann_TOF_s']/DAY:.3f}",
                'sol_TOF_day'  : f"{s['sol_TOF_s']/DAY:.3f}",
                'err_TOF_pct'  : (f"{s['err_TOF_pct']:+.4f}"
                                  if np.isfinite(s['err_TOF_pct']) else ''),
                'sol_h_P_km'   : (f"{s['sol_h_P_km']:.2f}"
                                  if np.isfinite(s['sol_h_P_km']) else ''),
                'sol_mp_T1'    : f"{s.get('sol_mp_T1',   0.0):.6f}",
                'sol_mp_T2a'   : f"{s.get('sol_mp_T2a',  0.0):.6f}",
                'sol_mp_drag'  : f"{s.get('sol_mp_drag', 0.0):.6f}",
                'sol_mp_T2b'   : f"{s.get('sol_mp_T2b',  0.0):.6f}",
                't_T1_end_day' : f"{s['t_T1_end_s']/DAY:.3f}",
                't_T2a_end_day': f"{s['t_T2a_end_s']/DAY:.3f}",
                't_Tp_end_day' : f"{s['t_Tp_end_s']/DAY:.3f}",
                't_T2b_end_day': f"{s['t_T2b_end_s']/DAY:.3f}",
                't_capture_end_day': f"{s['t_capture_end_s']/DAY:.3f}",
            })


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Re-evaluate ANN sequence with solver + Fig. 11 plot")
    ap.add_argument('--seq-csv', required=True,
                    help='run_sequence_search.py 가 만든 csv')
    ap.add_argument('--alpha', type=float, required=True,
                    help='solver 의 cost 가중치 (논문 식 28)')
    ap.add_argument('--debris-yaml', default='configs/random5000debris.yaml',
                    help='sequence search 에 쓴 debris yaml')
    ap.add_argument('--out-prefix', default=None,
                    help='출력 prefix. 기본: <seq-csv 의 prefix>_eval')
    ap.add_argument('--show', action='store_true', help='figure 창 띄우기')
    ap.add_argument('--no-overlay-ann', action='store_true',
                    help='m_prop 그래프에 ANN 누적 곡선 점선 overlay 끄기')
    args = ap.parse_args()

    if not os.path.exists(args.seq_csv):
        raise SystemExit(f"seq-csv not found: {args.seq_csv}")

    params, sim_cfg = load_params(
        constants_path='configs/constants.yaml',
        simulation_path='configs/simulation.yaml',
    )
    debris_dict = load_debris_list(args.debris_yaml)
    seq_rows = load_sequence_csv(args.seq_csv)

    if not seq_rows:
        raise SystemExit("seq-csv 가 비어 있습니다.")

    print("=" * 88)
    print(f"  evaluate_sequence : solver 로 재평가 + Fig. 11 figure")
    print("=" * 88)
    print(f"  seq csv        : {args.seq_csv}")
    print(f"  debris yaml    : {args.debris_yaml}")
    print(f"  α (solver)     : {args.alpha}")
    print(f"  N steps        : {len(seq_rows)}")
    print("-" * 88)

    steps = reevaluate_with_solver(seq_rows, debris_dict, params, args.alpha)

    print_error_summary(steps)

    # 출력 prefix
    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        base = args.seq_csv
        if base.lower().endswith('.csv'):
            base = base[:-4]
        out_prefix = base + '_eval'

    # 결과 csv
    eval_csv = out_prefix + '.csv'
    save_eval_csv(steps, eval_csv)
    print(f"  eval csv saved : {eval_csv}")

    # 그림
    depth = len(steps) + 1   # depth = visited debris 수
    title = (f"α = {args.alpha}    depth = {depth}    "
             f"m_prop(solver) = {sum(s['sol_m_prop_kg'] for s in steps):.2f} kg    "
             f"time = {steps[-1]['t_capture_end_s']/DAY/365.25:.2f} y")
    fig_png = out_prefix + '.png'
    plot_fig11(steps, params, title=title, out_png=fig_png,
               show=args.show,
               overlay_ann=(not args.no_overlay_ann))

    print("=" * 88)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)