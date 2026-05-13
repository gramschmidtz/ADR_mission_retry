"""
sweep_raan_hp.py
================
debris0001 → debris0002 transfer 에서 (RAAN_D2, h_P) 2D grid 를 sweep 하여
ΔV_PT, T_PT, J 의 **히트맵 (2D heatmap)** 3 장을 그린다.

각 (RAAN_D2, h_P) 격자점에서 `evaluate_phasing_orbit` 을 직접 호출하므로
sweep_raan_d2.py 와 달리 h_P 는 **고정 grid** 를 사용한다 (Brent 최적화 X).
이렇게 하면 sweep_raan_d2 에서 발생하던 톱니 (Brent 가 다봉 J 의 차선 골짜기로
미끄러져 생기는 jitter) 없이 J 의 진짜 표면 모양을 볼 수 있다.

Figure 1 : ΔV_PT (RAAN_D2, h_P) [m/s]               linear colormap
Figure 2 : T_PT  (RAAN_D2, h_P) [day]               log colormap (LogNorm)
Figure 3 : J     (RAAN_D2, h_P)                     linear colormap

축 정의:
    x = RAAN_D2 [deg]
    y = h_P    [km]
    color = ΔV_PT / T_PT / J

RAAN_D2 sweep 범위/스텝 :  simulation.yaml → search.raan_min_deg / raan_max_deg / raan_step_deg
h_P     sweep 범위/스텝 :  simulation.yaml → search.h_P_min_km / h_P_max_km / h_step_km
α                       :  simulation.yaml → training_data.alpha
debris pair             :  스크립트 상단 DEBRIS_PAIR 상수

실행 :
  python scripts/sweep_raan_hp.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
#   search.raan_min_deg, search.raan_max_deg, search.raan_step_deg
#   search.h_P_min_km,   search.h_P_max_km,   search.h_step_km

# Figure 저장 옵션
SAVE_FIG = True


# ──────────────────────────────────────────────────────────────
# T1 단계 진행 (m_SC_after_T1 산출) — test_J.py 와 동일
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
# 2D grid sweep
# ──────────────────────────────────────────────────────────────
def sweep_grid(d1, d2, params, alpha, raan_grid_deg, h_P_grid_km):
    """
    (RAAN_D2, h_P) 격자 위에서 evaluate_phasing_orbit 을 호출하고
    dv_PT, tof_PT, J 의 2D 배열 (shape: (N_h, N_raan)) 을 반환한다.

    배열 인덱싱 규약 : Z[i_h, j_raan]
        i_h   : h_P  방향 (행)
        j_raan: RAAN 방향 (열)
    """
    i_rad      = np.deg2rad(params['inclination_deg'])
    m_SC_start = params['m0']

    N_h    = len(h_P_grid_km)
    N_raan = len(raan_grid_deg)

    dv_PT = np.full((N_h, N_raan), np.nan)
    T_PT  = np.full((N_h, N_raan), np.nan)
    J_arr = np.full((N_h, N_raan), np.nan)

    for j, raan_deg in enumerate(raan_grid_deg):
        for i, h_P in enumerate(h_P_grid_km):
            try:
                J, res = evaluate_phasing_orbit(
                    h_P_km     = float(h_P),
                    h_D1_km    = d1['alt0_km'],
                    RAAN_D1_0  = np.deg2rad(d1['RAAN']),
                    m_D1       = d1['mass'],
                    h_disp_km  = params['disposal_alt_km'],
                    h_D2_km    = d2['alt0_km'],
                    RAAN_D2_0  = np.deg2rad(float(raan_deg)),
                    m_SC_start = m_SC_start,
                    params     = params,
                    alpha      = alpha,
                    i_rad      = i_rad,
                )
                if res and np.isfinite(J):
                    dv_PT[i, j] = res['dv_PT']
                    T_PT [i, j] = res['tof_PT'] / 86400.0   # day
                    J_arr[i, j] = J
            except Exception:
                pass

    return dv_PT, T_PT, J_arr


# ──────────────────────────────────────────────────────────────
# 히트맵 유틸 — pcolormesh + 정확한 격자 경계
# ──────────────────────────────────────────────────────────────
def _bin_edges(centers):
    """
    격자 중심 좌표 (등간격) → pcolormesh 용 경계 좌표 (N+1 개) 반환.
    """
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        # 1점이면 임의로 ±0.5 폭
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    step = centers[1] - centers[0]
    edges = np.concatenate([
        [centers[0] - step / 2],
        centers + step / 2,
    ])
    return edges


def plot_heatmap(raan_grid, h_P_grid, Z, clabel, title,
                 cmap='viridis', use_log=False, h_D2_km=None,
                 d1_RAAN_deg=None, col_min_overlay=False):
    """
    (raan_grid, h_P_grid, Z) → 2D heatmap (pcolormesh).
    use_log=True 면 LogNorm 으로 colormap 자체를 log 스케일링.

    Parameters
    ----------
    Z         : shape (len(h_P_grid), len(raan_grid)) 2D 배열
    h_D2_km   : 표시할 경우 h_P = h_D2 수평선
    d1_RAAN_deg : 표시할 경우 D1 RAAN 수직선
    col_min_overlay : True 면 각 RAAN_D2 열에서 Z 가 최소가 되는 h_P 를
                      빨간 점으로 표시 (J heatmap 의 RAAN 별 최적 h_P 가시화)
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 6))

    # NaN mask
    Z_masked = np.ma.masked_invalid(Z)

    # pcolormesh 격자 경계
    x_edges = _bin_edges(raan_grid)
    y_edges = _bin_edges(h_P_grid)

    if use_log:
        # log scale colormap : 양수만 사용
        positive = np.where(Z > 0, Z, np.nan)
        vmin = np.nanmin(positive)
        vmax = np.nanmax(positive)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        mesh = ax.pcolormesh(x_edges, y_edges, Z_masked,
                             cmap=cmap, norm=norm,
                             shading='auto')
    else:
        mesh = ax.pcolormesh(x_edges, y_edges, Z_masked,
                             cmap=cmap, shading='auto')

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(clabel)

    ax.set_xlabel('RAAN_D2 [deg]')
    ax.set_ylabel('h_P [km]')
    ax.set_title(title, fontsize=12)

    # h_P = h_D2 수평선 (전략 경계)
    if h_D2_km is not None:
        ax.axhline(h_D2_km, color='white', ls='--', lw=1.0, alpha=0.85,
                   label=f'h_P = h_D2 = {h_D2_km:.0f} km')
    # D1 RAAN 수직선
    if d1_RAAN_deg is not None:
        ax.axvline(d1_RAAN_deg, color='white', ls=':', lw=1.0, alpha=0.7,
                   label=f'D1 RAAN = {d1_RAAN_deg:.1f}°')

    # 각 RAAN_D2 열에서 Z 최저점 (column-wise argmin) 을 빨간 점으로 표시
    if col_min_overlay:
        N_raan = Z.shape[1]
        opt_h = np.full(N_raan, np.nan)
        for j in range(N_raan):
            col = Z[:, j]
            if np.isfinite(col).any():
                opt_h[j] = h_P_grid[np.nanargmin(col)]
        ax.plot(raan_grid, opt_h,
                linestyle='None', marker='o', color='red',
                ms=3.0, mew=0, alpha=0.9,
                label='argmin h_P per RAAN_D2')

    if (h_D2_km is not None) or (d1_RAAN_deg is not None) or col_min_overlay:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

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
    alpha = sim_cfg['training_data']['alpha']
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

    # T1 단계 (정보 출력용)
    m_SC_after_T1, tof_T1, m_prop_T1 = compute_m_SC_after_T1(
        d1, params['m0'], params)

    # 헤더 출력
    print('=' * 88)
    print(f'  (RAAN_D2, h_P) 2D heatmap sweep : {d1_name} → {d2_name}')
    print('=' * 88)
    print(f'  α                  : {alpha}')
    print(f'  D1                 : alt={d1["alt0_km"]:7.2f} km, '
          f'RAAN={d1["RAAN"]:7.2f}°, m={d1["mass"]:6.2f} kg')
    print(f'  D2                 : alt={d2["alt0_km"]:7.2f} km, '
          f'RAAN=sweep,        m={d2["mass"]:6.2f} kg')
    print(f'  disposal alt       : {params["disposal_alt_km"]:.2f} km')
    print(f'  m_SC after T1      : {m_SC_after_T1:.2f} kg '
          f'(T1 사용 추진제 {m_prop_T1:.2f} kg)')
    print(f'  RAAN_D2 grid       : [{raan_min:+.2f}, {raan_max:+.2f}]°, '
          f'step {raan_step}° ({len(raan_grid)} 점)')
    print(f'  h_P grid           : [{h_min:.1f}, {h_max:.1f}] km, '
          f'step {h_step} km ({len(h_P_grid)} 점)')
    print(f'  총 평가 횟수       : {len(raan_grid) * len(h_P_grid)} 개')
    print('-' * 88)

    # Sweep 실행
    print('  sweeping ...', end=' ', flush=True)
    dv_PT, T_PT, J_arr = sweep_grid(d1, d2, params, alpha, raan_grid, h_P_grid)
    n_total = dv_PT.size
    n_valid = int(np.isfinite(J_arr).sum())
    print(f'완료 ({n_valid}/{n_total} 유효)')

    # ──── 통계: grid 최저 J 위치 (정보 출력용) ────
    if n_valid > 0:
        i_h, j_r = np.unravel_index(np.nanargmin(J_arr), J_arr.shape)
        print('-' * 88)
        print(f'  grid 최저 J : J = {J_arr[i_h, j_r]:.4f}')
        print(f'    @ RAAN_D2 = {raan_grid[j_r]:+7.2f}°, '
              f'h_P = {h_P_grid[i_h]:7.2f} km')
        print(f'    ΔV_PT = {dv_PT[i_h, j_r]:8.2f} m/s, '
              f'T_PT = {T_PT[i_h, j_r]:9.2f} day')
    print('=' * 88)

    # ──── 그래프 생성 ────
    figs = []

    # Figure 1 : ΔV_PT (linear)
    fig1 = plot_heatmap(
        raan_grid, h_P_grid, dv_PT,
        clabel='ΔV_PT  [m/s]',
        title=f'ΔV_PT heatmap ({d1_name} → {d2_name}, α={alpha})',
        cmap='viridis',
        h_D2_km=d2['alt0_km'],
        d1_RAAN_deg=d1['RAAN'],
    )
    figs.append(('dv_PT', fig1))

    # Figure 2 : T_PT (log colormap)
    fig2 = plot_heatmap(
        raan_grid, h_P_grid, T_PT,
        clabel='T_PT  [day]  (log color)',
        title=f'T_PT heatmap ({d1_name} → {d2_name}, α={alpha})',
        cmap='plasma',
        use_log=True,
        h_D2_km=d2['alt0_km'],
        d1_RAAN_deg=d1['RAAN'],
    )
    figs.append(('T_PT_log', fig2))

    # Figure 3 : J (linear) — RAAN 별 argmin h_P 를 빨간 점으로 오버레이
    fig3 = plot_heatmap(
        raan_grid, h_P_grid, J_arr,
        clabel=f'J  (α={alpha})',
        title=f'J heatmap ({d1_name} → {d2_name}, α={alpha})',
        cmap='magma',
        h_D2_km=d2['alt0_km'],
        d1_RAAN_deg=d1['RAAN'],
        col_min_overlay=True,
    )
    figs.append(('J', fig3))

    # ──── 저장 ────
    if SAVE_FIG:
        out_dir = sim_cfg.get('visualization', {}).get(
            'output_dir', 'results/figures')
        os.makedirs(out_dir, exist_ok=True)
        for tag, fig in figs:
            out_path = os.path.join(
                out_dir,
                f'sweep_raan_hp_{d1_name}_to_{d2_name}_'
                f'alpha{alpha}_{tag}.png'
            )
            fig.savefig(out_path, dpi=140, bbox_inches='tight')
            print(f'  그래프 저장 : {out_path}')

    plt.show()


if __name__ == '__main__':
    main()