r"""
sweep_alpha_raan_hp.py
======================
(α, RAAN_D2) 2D grid 위에서 각각 J(h_P) 를 h_P grid sweep 으로 최소화하여
얻은 **최적 phasing 고도 h_P\*(α, RAAN_D2)** 를 z 축으로 3D surface 로 그린다.

목적
----
- α 가 변할 때 최적 h_P\* 가 어떻게 추격(h_P<h_D2) ↔ 지연(h_P>h_D2) 영역을
  옮겨다니는지 (전략 전환 표면) 시각화
- sweep_raan_d2.py 의 Brent 가 잡지 못하던 다봉 J(h_P) 의 진짜 전역 최저를
  보여주기 위해 **h_P 는 고정 grid sweep** 으로 처리 → 톱니/jitter 없음.

축
--
    x = α            [-]   (0 = TOF 최소, 1 = ΔV 최소)
    y = RAAN_D2      [deg]
    z = optimal h_P  [km]

설정 (simulation.yaml 의 search 섹션):
    search.alpha_min,    alpha_max,    alpha_step
    search.raan_min_deg, raan_max_deg, raan_step_deg
    search.h_P_min_km,   h_P_max_km,   h_step_km

debris pair 는 스크립트 상단 DEBRIS_PAIR 상수.

실행 :
  python scripts/sweep_alpha_raan_hp.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import (
    evaluate_phasing_orbit,
    tof_tsiolkovsky,
    propellant_consumed_eq10,
)


# ──────────────────────────────────────────────────────────────
# 사용자 설정
# ──────────────────────────────────────────────────────────────
DEBRIS_PAIR = ("debris0001", "debris0002")

# 모든 sweep 범위/스텝은 simulation.yaml 의 search 섹션 사용:
#   search.alpha_min,    alpha_max,    alpha_step
#   search.raan_min_deg, raan_max_deg, raan_step_deg
#   search.h_P_min_km,   h_P_max_km,   h_step_km

# Figure 저장 옵션
SAVE_FIG = True


# ──────────────────────────────────────────────────────────────
# T1 단계 (m_SC_after_T1 산출, 정보 출력용)
# ──────────────────────────────────────────────────────────────
def compute_m_SC_after_T1(debris1, m_SC_start, params):
    h_D1   = debris1["alt0_km"]
    m_D1   = debris1["mass"]
    h_disp = params["disposal_alt_km"]

    m_total_T1 = m_SC_start + m_D1
    tof_T1     = tof_tsiolkovsky(h_D1, h_disp, m_total_T1, params)
    m_prop_T1  = propellant_consumed_eq10(tof_T1, params)
    m_SC_after_T1 = m_SC_start - m_prop_T1
    return m_SC_after_T1, tof_T1, m_prop_T1


# ──────────────────────────────────────────────────────────────
# 단일 (α, RAAN_D2) 에서 J(h_P) grid 최저 → 최적 h_P 반환
# ──────────────────────────────────────────────────────────────
def best_hP_grid(d1, d2, params, alpha, raan_d2_deg, h_P_grid, i_rad):
    """
    주어진 (α, RAAN_D2) 에서 h_P 를 h_P_grid 전체에 대해 sweep 하고
    J 가 최소가 되는 h_P 와 그때의 J, ΔV_PT, T_PT 를 반환한다.

    NaN 반환 시 phasing 불가능한 케이스.
    """
    m_SC_start = params['m0']
    best_J     = np.inf
    best_hP    = np.nan
    best_dv    = np.nan
    best_T     = np.nan

    for h_P in h_P_grid:
        try:
            J, res = evaluate_phasing_orbit(
                h_P_km     = float(h_P),
                h_D1_km    = d1['alt0_km'],
                RAAN_D1_0  = np.deg2rad(d1['RAAN']),
                m_D1       = d1['mass'],
                h_disp_km  = params['disposal_alt_km'],
                h_D2_km    = d2['alt0_km'],
                RAAN_D2_0  = np.deg2rad(float(raan_d2_deg)),
                m_SC_start = m_SC_start,
                params     = params,
                alpha      = alpha,
                i_rad      = i_rad,
            )
        except Exception:
            continue
        if (res is None) or (not np.isfinite(J)):
            continue
        if J < best_J:
            best_J  = J
            best_hP = float(h_P)
            best_dv = res['dv_PT']
            best_T  = res['tof_PT'] / 86400.0  # day
    return best_hP, best_J, best_dv, best_T


# ──────────────────────────────────────────────────────────────
# 2D grid sweep over (α, RAAN_D2)
# ──────────────────────────────────────────────────────────────
def sweep_alpha_raan(d1, d2, params, alpha_grid, raan_grid_deg, h_P_grid):
    """
    (α, RAAN_D2) 격자 위에서 best_hP_grid 를 호출하고
    optimal_hP, J, dv_PT, T_PT 의 2D 배열 (shape: (N_a, N_r)) 을 반환한다.

    배열 인덱싱 : Z[i_alpha, j_raan]
    """
    i_rad = np.deg2rad(params['inclination_deg'])

    N_a = len(alpha_grid)
    N_r = len(raan_grid_deg)

    opt_hP   = np.full((N_a, N_r), np.nan)
    opt_J    = np.full((N_a, N_r), np.nan)
    opt_dv   = np.full((N_a, N_r), np.nan)
    opt_T    = np.full((N_a, N_r), np.nan)

    for i, alpha in enumerate(alpha_grid):
        for j, raan_deg in enumerate(raan_grid_deg):
            hP, J, dv, T = best_hP_grid(
                d1, d2, params, alpha, raan_deg, h_P_grid, i_rad)
            if np.isfinite(hP):
                opt_hP[i, j] = hP
                opt_J [i, j] = J
                opt_dv[i, j] = dv
                opt_T [i, j] = T

    return opt_hP, opt_J, opt_dv, opt_T


# ──────────────────────────────────────────────────────────────
# 히트맵 — pcolormesh + 정확한 격자 경계
# ──────────────────────────────────────────────────────────────
def _bin_edges(centers):
    """격자 중심 좌표 → pcolormesh 용 경계 좌표 (N+1 개) 반환."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    step = centers[1] - centers[0]
    edges = np.concatenate([
        [centers[0] - step / 2],
        centers + step / 2,
    ])
    return edges


def plot_optimal_hP_heatmap(alpha_grid, raan_grid, opt_hP,
                            d1_name, d2_name, h_D2_km):
    """
    optimal h_P*(α, RAAN_D2) 를 2D heatmap 으로 그린다.
    h_P = h_D2 에 해당하는 colorbar tick 을 표시하여 전략 경계 가시화.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    # NaN mask
    Z_masked = np.ma.masked_invalid(opt_hP)

    # 격자 경계
    x_edges = _bin_edges(alpha_grid)
    y_edges = _bin_edges(raan_grid)

    # 배열은 [i_alpha, j_raan] 이므로 pcolormesh 에 넘기려면 transpose:
    #   row → y (RAAN), col → x (α)
    Z_for_plot = Z_masked.T

    mesh = ax.pcolormesh(
        x_edges, y_edges, Z_for_plot,
        cmap='viridis', shading='auto',
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label('optimal h_P* [km]')

    # colorbar 위에 h_P = h_D2 경계 표시
    cbar.ax.axhline(h_D2_km, color='red', lw=1.2)
    # cbar.ax.text(
    #     1.05, h_D2_km, f' h_D2={h_D2_km:.0f}',
    #     transform=cbar.ax.get_yaxis_transform(),
    #     color='red', fontsize=8, va='center',
    # )

    ax.set_xlabel('α  (0 = TOF min,  1 = ΔV min)')
    ax.set_ylabel('RAAN_D2 [deg]')
    ax.set_title(
        f'optimal h_P*(α, RAAN_D2) — {d1_name} → {d2_name}\n'
        f'(red colorbar tick : h_P = h_D2 = {h_D2_km:.0f} km, '
        f'strategy boundary)',
        fontsize=11,
    )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.grid(False)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 메인
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

    # h_P grid (simulation.yaml 의 search 섹션)
    h_min  = max(params['disposal_alt_km'], params['h_P_min_km'])
    h_max  = params['h_P_max_km']
    h_step = params['h_step_km']
    h_P_grid = np.arange(h_min, h_max + h_step / 2, h_step)

    # RAAN_D2 grid (simulation.yaml 의 search 섹션)
    raan_min  = params['raan_min_deg']
    raan_max  = params['raan_max_deg']
    raan_step = params['raan_step_deg']
    raan_grid = np.arange(raan_min, raan_max + raan_step / 2, raan_step)

    # α grid (simulation.yaml 의 search 섹션)
    a_min   = params['alpha_min']
    a_max   = params['alpha_max']
    a_step  = params['alpha_step']
    alpha_grid = np.arange(a_min, a_max + a_step / 2, a_step)
    alpha_grid = np.clip(alpha_grid, 0.0, 1.0)  # 부동소수 오버슛 방지

    # T1 단계 (정보 출력용)
    m_SC_after_T1, tof_T1, m_prop_T1 = compute_m_SC_after_T1(
        d1, params['m0'], params)

    # 헤더
    print('=' * 88)
    print(f'  (α, RAAN_D2) sweep → optimal h_P* surface : {d1_name} → {d2_name}')
    print('=' * 88)
    print(f'  D1                 : alt={d1["alt0_km"]:7.2f} km, '
          f'RAAN={d1["RAAN"]:7.2f}°, m={d1["mass"]:6.2f} kg')
    print(f'  D2                 : alt={d2["alt0_km"]:7.2f} km, '
          f'RAAN=sweep,        m={d2["mass"]:6.2f} kg')
    print(f'  disposal alt       : {params["disposal_alt_km"]:.2f} km')
    print(f'  m_SC after T1      : {m_SC_after_T1:.2f} kg '
          f'(T1 사용 추진제 {m_prop_T1:.2f} kg)')
    print(f'  α grid             : [{a_min:.2f}, {a_max:.2f}], '
          f'step {a_step} ({len(alpha_grid)} 점)')
    print(f'  RAAN_D2 grid       : [{raan_min:+.2f}, {raan_max:+.2f}]°, '
          f'step {raan_step}° ({len(raan_grid)} 점)')
    print(f'  h_P grid           : [{h_min:.1f}, {h_max:.1f}] km, '
          f'step {h_step} km ({len(h_P_grid)} 점)')
    n_eval = len(alpha_grid) * len(raan_grid) * len(h_P_grid)
    print(f'  내부 evaluate 횟수  : {n_eval} 개')
    print('-' * 88)

    # Sweep 실행 (진행 상황 표시)
    print('  sweeping ...', flush=True)
    opt_hP, opt_J, opt_dv, opt_T = sweep_alpha_raan(
        d1, d2, params, alpha_grid, raan_grid, h_P_grid)
    n_total = opt_hP.size
    n_valid = int(np.isfinite(opt_hP).sum())
    print(f'  완료 ({n_valid}/{n_total} 유효)')

    # 통계
    if n_valid > 0:
        i_a, j_r = np.unravel_index(np.nanargmin(opt_J), opt_J.shape)
        print('-' * 88)
        print(f'  grid 전체 최저 J : J = {opt_J[i_a, j_r]:.4f}')
        print(f'    @ α = {alpha_grid[i_a]:.2f}, '
              f'RAAN_D2 = {raan_grid[j_r]:+7.2f}°, '
              f'optimal h_P* = {opt_hP[i_a, j_r]:.1f} km')
        print(f'    ΔV_PT = {opt_dv[i_a, j_r]:8.2f} m/s, '
              f'T_PT = {opt_T[i_a, j_r]:9.2f} day')

        # 전략 영역 통계
        n_catch = int(np.sum(opt_hP < d2['alt0_km']))
        n_wait  = int(np.sum(opt_hP > d2['alt0_km']))
        n_equal = int(np.sum(opt_hP == d2['alt0_km']))
        print(f'  전략 분포 : 추격(h_P<h_D2) {n_catch}, '
              f'지연(h_P>h_D2) {n_wait}, '
              f'경계(h_P=h_D2) {n_equal}  / 총 {n_valid}')
    print('=' * 88)

    # 그래프
    fig = plot_optimal_hP_heatmap(
        alpha_grid, raan_grid, opt_hP,
        d1_name, d2_name, d2['alt0_km'])

    # 저장
    if SAVE_FIG:
        out_dir = sim_cfg.get('visualization', {}).get(
            'output_dir', 'results/figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f'sweep_alpha_raan_hp_{d1_name}_to_{d2_name}_optimal_hP.png'
        )
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        print(f'  그래프 저장 : {out_path}')

    plt.show()


if __name__ == '__main__':
    main()