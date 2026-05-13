"""
sweep_raan_d2.py
================
debris0001 → debris0002 transfer 에서 D2 의 RAAN 을 0°~360° 로 sweep 하며
optimize_phasing_orbit 결과를 그린다.

목적 : 추격(catch-up) ↔ 지연(wait-up) 전략 전환점을 시각적으로 확인.

각 α 값 (TOF 최소 / 균형 / ΔV 최소) 에 대해 4 패널 그래프 :
  (1) h_P [km]       — 전략 경계는 h_P = h_D2 수평선
  (2) Tp  [day]      — 전환점에서 Tp 가 0 근처로 떨어졌다가 점프
  (3) ΔV_PT [m/s]    — 전환점에서 점프
  (4) J              — 목적함수, 전환점에서 불연속

실행 :
  python scripts/sweep_raan_d2.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import optimize_phasing_orbit


# ──────────────────────────────────────────────────────────────
# 사용자 설정
# ──────────────────────────────────────────────────────────────
DEBRIS_PAIR = ("debris0001", "debris0002")
# RAAN_D2 sweep 범위/스텝은 simulation.yaml 의 search 섹션을 사용
#   search.raan_min_deg, search.raan_max_deg, search.raan_step_deg
ALPHA_LIST  = [0.0, 0.5, 1.0]   # TOF 최소 / 균형 / ΔV 최소


# ──────────────────────────────────────────────────────────────
# Sweep 실행 (1 개의 α 에 대해)
# ──────────────────────────────────────────────────────────────
def sweep_one_alpha(d1, d2, params, alpha, raan_d2_grid_deg):
    """
    한 α 값에 대해 RAAN_D2 grid 위에서 optimize_phasing_orbit 을 호출하고
    결과 배열을 반환한다.

    Returns
    -------
    h_P_arr    : (N,) [km]
    Tp_arr     : (N,) [day]    phasing orbit 대기 시간
    T_PT_arr   : (N,) [day]    전체 phasing 구간 시간 (T2a + Tp + T2b, 식 27)
    dv_PT_arr  : (N,) [m/s]
    J_arr      : (N,)
    fail_mask  : (N,) bool (True = optimize 실패 또는 phasing 불가)
    """
    i_rad = np.deg2rad(params['inclination_deg'])
    N     = len(raan_d2_grid_deg)
    h_P_arr   = np.full(N, np.nan)
    Tp_arr    = np.full(N, np.nan)
    T_PT_arr  = np.full(N, np.nan)
    dv_PT_arr = np.full(N, np.nan)
    J_arr     = np.full(N, np.nan)
    fail_mask = np.zeros(N, dtype=bool)

    for j, raan_d2_deg in enumerate(raan_d2_grid_deg):
        try:
            res = optimize_phasing_orbit(
                h_D1_km    = d1['alt0_km'],
                RAAN_D1_0  = np.deg2rad(d1['RAAN']),
                m_D1       = d1['mass'],
                h_disp_km  = params['disposal_alt_km'],
                h_D2_km    = d2['alt0_km'],
                RAAN_D2_0  = np.deg2rad(raan_d2_deg),
                m_SC_start = params['m0'],
                params     = params,
                alpha      = alpha,
                i_rad      = i_rad,
            )
            if res and np.isfinite(res.get('J', np.nan)):
                h_P_arr[j]   = res['h_P_km']
                Tp_arr[j]    = res['Tp'] / 86400.0
                T_PT_arr[j]  = (res['tof_T2a'] + res['Tp']
                                + res['tof_T2b']) / 86400.0
                dv_PT_arr[j] = res['dv_T2a'] + res['dv_T2b'] + res['dv_drag_P']
                J_arr[j]     = res['J']
            else:
                fail_mask[j] = True
        except Exception:
            fail_mask[j] = True

    return h_P_arr, Tp_arr, T_PT_arr, dv_PT_arr, J_arr, fail_mask


# ──────────────────────────────────────────────────────────────
# 전환점 검출
# ──────────────────────────────────────────────────────────────
def find_transitions(raan_grid, h_P_arr, h_D2):
    """
    h_P 가 h_D2 를 가로지르는 (전략이 바뀌는) 인접 grid 인덱스 쌍을 반환.
    """
    valid = ~np.isnan(h_P_arr)
    transitions = []
    for j in range(len(raan_grid) - 1):
        if not (valid[j] and valid[j + 1]):
            continue
        side_j  = (h_P_arr[j]     - h_D2)
        side_j1 = (h_P_arr[j + 1] - h_D2)
        # 부호 변경 또는 큰 점프 (>200 km)
        if side_j * side_j1 < 0 or abs(h_P_arr[j + 1] - h_P_arr[j]) > 200:
            transitions.append((j, j + 1))
    return transitions


# ──────────────────────────────────────────────────────────────
# 메인 : sweep + plot
# ──────────────────────────────────────────────────────────────
def main():
    # 설정 로드
    params, sim_cfg = load_params(
        constants_path='configs/constants.yaml',
        simulation_path='configs/simulation.yaml',
    )
    debris_all = load_debris_list(sim_cfg['training_data']['debris_yaml'])
    d1_name, d2_name = DEBRIS_PAIR
    d1 = dict(debris_all[d1_name]); d1['name'] = d1_name
    d2 = dict(debris_all[d2_name]); d2['name'] = d2_name

    raan_min  = params['raan_min_deg']
    raan_max  = params['raan_max_deg']
    raan_step = params['raan_step_deg']
    raan_d2_grid = np.arange(
        raan_min,
        raan_max + raan_step / 2,
        raan_step,
    )

    print('=' * 78)
    print(f'  RAAN_D2 sweep : {d1_name} → {d2_name}')
    print('=' * 78)
    print(f'  D1 : alt={d1["alt0_km"]:7.2f} km, RAAN={d1["RAAN"]:7.2f}°, m={d1["mass"]:6.2f} kg')
    print(f'  D2 : alt={d2["alt0_km"]:7.2f} km, RAAN=sweep,     m={d2["mass"]:6.2f} kg')
    print(f'  RAAN_D2 범위 : [{raan_min:.2f}, {raan_max:.2f}]°, '
          f'step {raan_step}° ({len(raan_d2_grid)} 점)')
    print(f'  α 값         : {ALPHA_LIST}')
    print('-' * 78)

    # 각 α 별 sweep 결과 보관
    results = {}
    for alpha in ALPHA_LIST:
        print(f'  sweeping α={alpha} ...', end=' ', flush=True)
        h_P, Tp, T_PT, dv, J, fail = sweep_one_alpha(
            d1, d2, params, alpha, raan_d2_grid)
        results[alpha] = {
            'h_P': h_P, 'Tp': Tp, 'T_PT': T_PT,
            'dv_PT': dv, 'J': J, 'fail': fail,
        }
        n_fail = int(fail.sum())
        print(f'완료 (실패 {n_fail}/{len(raan_d2_grid)})')

    # ────────────────────────────────────────────────
    # plot
    # ────────────────────────────────────────────────
    colors = {0.0: 'tab:blue', 0.5: 'tab:green', 1.0: 'tab:red'}
    labels = {0.0: 'α=0 (TOF min)', 0.5: 'α=0.5 (balanced)', 1.0: 'α=1 (ΔV min)'}

    h_D2 = d2['alt0_km']

    # 전환점 검출 (α=0.5 기준, 두 figure 공통으로 표시)
    trans_05 = find_transitions(raan_d2_grid, results[0.5]['h_P'], h_D2)
    transition_x = [
        0.5 * (raan_d2_grid[j] + raan_d2_grid[j1]) for (j, j1) in trans_05
    ]

    # ════════════════════════════════════════════════
    # Figure 1 : J, ΔV_PT, T_PT, Tp  (4x1)
    #   T_PT = tof_T2a + Tp + tof_T2b  (식 27)
    # ════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    # ── Panel 1 : J ──
    ax = axes1[0]
    for alpha in ALPHA_LIST:
        r = results[alpha]
        ax.plot(raan_d2_grid, r['J'], color=colors[alpha],
                label=labels[alpha], lw=1.5)
    ax.set_ylabel('J = α·ΔV/1000 + (1-α)·TOF/365')
    ax.set_title(f'RAAN_D2 sweep : phasing metrics '
                 f'({d1_name} -> {d2_name})', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 2 : ΔV_PT ──
    ax = axes1[1]
    for alpha in ALPHA_LIST:
        r = results[alpha]
        ax.plot(raan_d2_grid, r['dv_PT'], color=colors[alpha],
                label=labels[alpha], lw=1.5)
    ax.set_ylabel('ΔV_PT [m/s]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 3 : T_PT (log) ──
    ax = axes1[2]
    for alpha in ALPHA_LIST:
        r = results[alpha]
        ax.semilogy(raan_d2_grid, r['T_PT'], color=colors[alpha],
                    label=labels[alpha], lw=1.5)
    ax.set_ylabel('T_PT [day]  (log scale)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # ── Panel 4 : Tp (log) ──
    ax = axes1[3]
    for alpha in ALPHA_LIST:
        r = results[alpha]
        ax.semilogy(raan_d2_grid, r['Tp'], color=colors[alpha],
                    label=labels[alpha], lw=1.5)
    ax.set_ylabel('Tp [day]  (log scale)')
    ax.set_xlabel('RAAN_D2 [deg]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # 공통 : x 축, 전환점 선, D1 RAAN
    # x 축 tick 간격: 범위 길이에 따라 자동 선택
    raan_span = raan_max - raan_min
    if raan_span <= 30:
        tick_step = 2
    elif raan_span <= 90:
        tick_step = 10
    elif raan_span <= 180:
        tick_step = 20
    else:
        tick_step = 30
    # tick 시작점을 tick_step 의 배수로 정렬
    xtick_start = np.floor(raan_min / tick_step) * tick_step
    xticks = np.arange(xtick_start, raan_max + tick_step / 2, tick_step)

    for axx in axes1:
        axx.set_xlim(raan_min, raan_max)
        axx.set_xticks(xticks)
        for x_mid in transition_x:
            axx.axvline(x_mid, color='k', ls='--', lw=0.8, alpha=0.4)
        axx.axvline(d1['RAAN'], color='tab:orange', ls=':', lw=1, alpha=0.5)

    fig1.tight_layout()

    # ════════════════════════════════════════════════
    # Figure 2 : optimal h_P
    # ════════════════════════════════════════════════
    fig2, ax = plt.subplots(1, 1, figsize=(11, 4.5))

    for alpha in ALPHA_LIST:
        r = results[alpha]
        ax.plot(raan_d2_grid, r['h_P'], color=colors[alpha],
                label=labels[alpha], lw=1.5)
    ax.axhline(h_D2, color='k', ls=':', lw=1, alpha=0.6,
               label=f'h_D2 = {h_D2:.0f} km (strategy boundary)')
    ax.set_ylabel('optimal phasing orbit h_P* [km]')
    ax.set_xlabel('RAAN_D2 [deg]')
    ax.set_title(f'RAAN_D2 sweep : optimal h_P '
                 f'({d1_name} -> {d2_name})', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Strategy regions (catch-up below, wait-up above)
    ymin, ymax = ax.get_ylim()
    ax.axhspan(ymin, h_D2, alpha=0.05, color='blue', zorder=0)
    ax.axhspan(h_D2, ymax, alpha=0.05, color='red', zorder=0)
    text_x = raan_min + 0.014 * (raan_max - raan_min)
    ax.text(text_x, h_D2 - (h_D2 - ymin) * 0.1, 'catch-up\nh_P < h_D2',
            fontsize=8, color='blue', alpha=0.7, va='top')
    ax.text(text_x, h_D2 + (ymax - h_D2) * 0.1, 'wait-up\nh_P > h_D2',
            fontsize=8, color='red', alpha=0.7, va='bottom')

    # 전환점 선 + 주석
    annot_dx = 0.022 * (raan_max - raan_min)
    for x_mid in transition_x:
        ax.axvline(x_mid, color='k', ls='--', lw=0.8, alpha=0.4)
        ax.annotate(f'transition\nRAAN~{x_mid:.0f} deg',
                    xy=(x_mid, h_D2), xytext=(x_mid + annot_dx, h_D2 * 1.3),
                    fontsize=8, color='k', alpha=0.7,
                    arrowprops=dict(arrowstyle='->', color='k', alpha=0.4))

    # D1 RAAN 위치
    ax.axvline(d1['RAAN'], color='tab:orange', ls=':', lw=1, alpha=0.5)
    ax.text(d1['RAAN'] + 0.005 * (raan_max - raan_min),
            ymin + (ymax - ymin) * 0.02,
            f"D1 RAAN ({d1['RAAN']:.1f} deg)",
            fontsize=8, color='tab:orange', alpha=0.8, rotation=90,
            va='bottom')

    ax.set_xlim(raan_min, raan_max)
    ax.set_xticks(xticks)

    fig2.tight_layout()

    # 전환점 요약 출력
    print('-' * 78)
    print('전환점 요약 (α=0.5 기준) :')
    for j, j1 in trans_05:
        r = results[0.5]
        print(f'  RAAN_D2 = {raan_d2_grid[j]:.1f}° → {raan_d2_grid[j1]:.1f}° :'
              f'   h_P {r["h_P"][j]:.0f} → {r["h_P"][j1]:.0f} km,'
              f'   J {r["J"][j]:.3f} → {r["J"][j1]:.3f}')

    plt.show()


if __name__ == '__main__':
    main()