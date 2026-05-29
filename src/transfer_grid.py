"""
transfer_grid.py
================
h_D1, Ω_D1, m_D1, h_D2, Ω_D2, m_SC  →  m_prop, TOF

논문 §3.1 의 ANN 입출력 매핑과 동일한 인터페이스를 제공한다 (식 30, 31).
즉 ANN 학습 데이터 생성용 ground-truth 함수.

기존 `src.transfer_solver.solve_transfer` 와의 차이:
    - 본 모듈은 (h_D1, Ω_D1, m_D1, h_D2, Ω_D2, m_SC) 를 입력으로 받아 한
      transfer 의 m_prop, TOF, 부가정보(dict) 를 반환하는 ANN 학습용 인터페이스.
    - 내부 h_P 최적화는 transfer_solver.optimize_phasing_orbit 와 **동일한
      grid search** (simulation.yaml 의 h_P_min_km / h_P_max_km / h_step_km).
      두 솔버는 같은 grid 와 같은 evaluate_phasing_orbit 을 호출하므로
      동일 입력에 대해 비트 단위로 동일한 결과를 낸다.

논문 §2.2 의 mission profile 전체 (T1 + T2a + Tp + T2b + Ts) 를 합산한
m_prop, TOF 를 반환한다. ANN 학습용 target 으로 그대로 사용 가능.

사용 예
-------
    from src.config_loader import load_params
    from src.transfer_grid import compute_transfer_grid

    params, sim_cfg = load_params()
    m_prop, TOF, info = compute_transfer_grid(
        h_D1=500.0, RAAN_D1_deg=0.0, m_D1=270.0,
        h_D2=800.0, RAAN_D2_deg=10.0,
        m_SC=400.0,
        params=params,
        alpha=1.0,
    )
"""

from __future__ import annotations

import numpy as np

from src.transfer_solver import (
    evaluate_phasing_orbit,
    tof_tsiolkovsky,
    propellant_consumed_eq10,
    delta_v_from_mass,
)


# ──────────────────────────────────────────────────────────────
# h_P grid search → 최적 phasing orbit
# ──────────────────────────────────────────────────────────────
def _find_best_hP_grid(
    h_D1, RAAN_D1_0, m_D1,
    h_disp,
    h_D2, RAAN_D2_0,
    m_SC_start, params, alpha, i_rad,
    h_P_grid,
):
    """
    주어진 h_P_grid 위에서 evaluate_phasing_orbit 을 호출하고
    J 가 최소가 되는 h_P 와 해당 result dict 를 반환한다.

    반환
    ----
    best_h_P : float [km]  최적 phasing orbit 고도
    best_res : dict        evaluate_phasing_orbit 가 반환한 상세 dict
                           (dv_T2a, dv_T2b, dv_drag_P, tof_T2a, Tp, tof_T2b,
                            m_after_T2a, m_after_drag, m_final, delta_RAAN 등)
    best_J   : float       해당 h_P 에서의 J 값
    """
    best_J  = np.inf
    best_hP = np.nan
    best_res = None

    for h_P in h_P_grid:
        J, res = evaluate_phasing_orbit(
            h_P_km     = float(h_P),
            h_D1_km    = h_D1,
            RAAN_D1_0  = RAAN_D1_0,
            m_D1       = m_D1,
            h_disp_km  = h_disp,
            h_D2_km    = h_D2,
            RAAN_D2_0  = RAAN_D2_0,
            m_SC_start = m_SC_start,
            params     = params,
            alpha      = alpha,
            i_rad      = i_rad,
        )
        if (res is None) or (not np.isfinite(J)):
            continue
        if J < best_J:
            best_J   = J
            best_hP  = float(h_P)
            best_res = res

    return best_hP, best_res, best_J


# ──────────────────────────────────────────────────────────────
# Public API : (입력 6 → 출력 2)
# ──────────────────────────────────────────────────────────────
def compute_transfer_grid(
    h_D1, RAAN_D1_deg, m_D1,
    h_D2, RAAN_D2_deg,
    m_SC,
    params, alpha,
    *,
    h_P_grid_km=None,
):
    """
    한 transfer 의 m_prop, TOF 를 계산한다. 논문 §3.1 식 (30) → 식 (31).

    Parameters
    ----------
    h_D1        : 출발 debris 고도 [km]
    RAAN_D1_deg : 출발 debris RAAN [deg] (현재 epoch 기준)
    m_D1        : 출발 debris 질량 [kg]
    h_D2        : 도착 debris 고도 [km]
    RAAN_D2_deg : 도착 debris RAAN [deg] (현재 epoch 기준)
    m_SC        : chaser 초기 질량 [kg]  (T1 시작 직전, D1 docking 전)
    params      : load_params() 로 얻은 물리 파라미터 dict
                  ( mu, Re, ge, J2, CD, S, T_max, Isp, m0, m_prop_max,
                    disposal_alt_km, inclination_deg,
                    h_P_min_km, h_P_max_km, h_step_km, ... 포함)
    alpha       : float ∈ [0, 1]  목적함수 가중치 (식 28)

    h_P_grid_km : (선택) phasing 고도 grid [km].
                  None 이면 params 의 [h_P_min_km, h_P_max_km] 범위를
                  h_step_km 간격으로 자동 생성.

    Returns
    -------
    m_prop : float [kg]  전체 mission 의 총 추진제 소비
    TOF    : float [s]   전체 mission 시간 (T1 + T2a + Tp + T2b + Ts)
    info   : dict        상세 결과 (각 phase 의 dv, tof, mass; h_P*; ΔΩ 등)

    참고
    ----
    - phasing orbit 고도 h_P 는 **grid search** 로 찾는다.
      grid step 은 simulation.yaml 의 search.h_step_km 로 지정.
    - eclipse / duty cycle 은 모델링하지 않음 (논문 §2.2.1 의 40% 미반영).
      이 솔버는 transfer_solver.solve_transfer 의 단순화와 동일하다.
    - Ts (capture time) 는 30 일 고정 (논문 §2.2).
    """
    # 물리 파라미터 unpack
    h_disp = params['disposal_alt_km']
    i_rad  = np.deg2rad(params['inclination_deg'])
    Ts     = 30.0 * 86400.0   # 30 일 (논문 §2.2)

    # h_P grid 자동 생성 (호출자가 미리 만들었으면 그걸 사용)
    if h_P_grid_km is None:
        h_min  = max(h_disp, params['h_P_min_km'])
        h_max  = params['h_P_max_km']
        h_step = params['h_step_km']
        h_P_grid_km = np.arange(h_min, h_max + h_step / 2, h_step)

    # ── Phase T1: D1 → disposal orbit (chaser + D1 합산 질량으로 하강) ──
    m_total_T1       = m_SC + m_D1
    tof_T1           = tof_tsiolkovsky(h_D1, h_disp, m_total_T1, params)
    m_prop_T1        = propellant_consumed_eq10(tof_T1, params)
    m_total_after_T1 = m_total_T1 - m_prop_T1
    dv_T1            = delta_v_from_mass(m_total_T1, m_prop_T1, params)
    # D1 release 후 chaser 단독 질량
    m_SC_after_T1    = m_SC - m_prop_T1

    # ── Phase T2a + Tp + T2b 의 최적 phasing 고도 grid search ──
    RAAN_D1_0 = np.deg2rad(RAAN_D1_deg)
    RAAN_D2_0 = np.deg2rad(RAAN_D2_deg)
    h_P_star, phasing, J_star = _find_best_hP_grid(
        h_D1=h_D1, RAAN_D1_0=RAAN_D1_0, m_D1=m_D1,
        h_disp=h_disp,
        h_D2=h_D2, RAAN_D2_0=RAAN_D2_0,
        m_SC_start=m_SC, params=params, alpha=alpha, i_rad=i_rad,
        h_P_grid=h_P_grid_km,
    )

    if phasing is None or not np.isfinite(h_P_star):
        # phasing orbit 결정 불가 — feasibility 없음
        return np.nan, np.nan, {
            'feasible': False,
            'reason'  : 'no valid h_P found in grid',
            'h_P_grid_km': np.asarray(h_P_grid_km),
        }

    # 각 phase 의 추진제 소비 (chronological 순서: T2a → Tp(drag) → T2b)
    m_prop_T2a   = m_SC_after_T1            - phasing['m_after_T2a']
    m_prop_drag  = phasing['m_after_T2a']   - phasing['m_after_drag']
    m_prop_T2b   = phasing['m_after_drag']  - phasing['m_final']
    m_prop_total = m_prop_T1 + m_prop_T2a + m_prop_drag + m_prop_T2b

    # 총 비행 시간
    TOF = tof_T1 + phasing['tof_T2a'] + phasing['Tp'] + phasing['tof_T2b'] + Ts

    info = {
        'feasible'      : True,
        # 구간별 시간 [s]
        'T1'            : tof_T1,
        'T2a'           : phasing['tof_T2a'],
        'Tp'            : phasing['Tp'],
        'T2b'           : phasing['tof_T2b'],
        'Ts'            : Ts,
        'TOF'           : TOF,
        # ΔV [m/s]
        'dv_T1'         : dv_T1,
        'dv_T2a'        : phasing['dv_T2a'],
        'dv_T2b'        : phasing['dv_T2b'],
        'dv_drag'       : phasing['dv_drag_P'],
        # 질량 [kg]
        'm_SC_start'    : m_SC,
        'm_SC_after_T1' : m_SC_after_T1,
        'm_SC_end'      : phasing['m_final'],
        'm_prop_T1'     : m_prop_T1,
        'm_prop_T2a'    : m_prop_T2a,
        'm_prop_T2b'    : m_prop_T2b,
        'm_prop_drag'   : m_prop_drag,
        'm_prop'        : m_prop_total,
        # phasing 결과
        'h_P_km'        : h_P_star,
        'delta_RAAN_rad': phasing['delta_RAAN'],
        'J_star'        : J_star,
        # grid 정보
        'h_P_grid_km'   : np.asarray(h_P_grid_km),
        'n_grid_points' : len(h_P_grid_km),
    }
    return m_prop_total, TOF, info