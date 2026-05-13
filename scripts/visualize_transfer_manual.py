"""
visualize_transfer_manual.py
============================
visualize_transfer.py 와 동일한 4-패널 그래프(논문 Fig. 2)를 그리되,
phasing orbit 고도 h_P 를 자동 최적화(minimize_scalar)하지 않고
사용자가 직접 지정한 값으로 전체 transfer 를 계산한다.

기존 visualize_transfer.py 는 건드리지 않는다.

사용 :
  - 스크립트 상단의 H_P_KM 상수를 원하는 phasing 고도 [km] 로 설정
  - 그 외 입력(debris pair, alpha)은 기존과 동일하게 simulation.yaml 사용

실행 :
  python scripts/visualize_transfer_manual.py
"""

import sys
import os
import numpy as np

import matplotlib.pyplot as plt  # noqa: F401  (plot_transfer 가 사용)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import (
    evaluate_phasing_orbit,
    tof_hohmann,
    propellant_consumed_eq10,
    delta_v_from_mass,
)

# 기존 visualize_transfer 의 build_timeline, plot_transfer 재사용
from scripts.visualize_transfer import plot_transfer


# ──────────────────────────────────────────────────────────────
# 사용자 설정
# ──────────────────────────────────────────────────────────────
H_P_KM = 500.0   # ← 여기를 원하는 phasing orbit 고도 [km] 로 변경


# ──────────────────────────────────────────────────────────────
# h_P 고정 버전 transfer 계산
# ──────────────────────────────────────────────────────────────
def solve_transfer_fixed_hP(debris1, debris2, m_SC, params, alpha, h_P_km):
    """
    solve_transfer 와 동일한 result dict 를 반환하되
    phasing orbit 고도 h_P_km 를 사용자가 직접 지정한다.

    내부 구조는 transfer_solver.solve_transfer 와 동일하지만
    optimize_phasing_orbit (minimize_scalar) 대신
    evaluate_phasing_orbit 를 단일 h_P_km 에서 한 번만 호출한다.

    Parameters
    ----------
    debris1, debris2 : dict     debris 정보
    m_SC             : float    chaser 초기 질량 [kg]
    params           : dict     물리/우주선 파라미터
    alpha            : float    목적함수 가중치 (J 출력용 — transfer 계산 자체는 영향 없음)
    h_P_km           : float    사용자 지정 phasing orbit 고도 [km]

    Returns
    -------
    result : dict   solve_transfer 와 동일한 형식
    """
    Re      = params['Re']
    i_rad   = np.deg2rad(params['inclination_deg'])
    h_disp  = params['disposal_alt_km']
    Ts      = 30.0 * 86400.0   # 30일 stay (논문)

    h_D1    = debris1['alt0_km']
    RAAN_D1 = np.deg2rad(debris1['RAAN'])
    m_D1    = debris1['mass']

    h_D2    = debris2['alt0_km']
    RAAN_D2 = np.deg2rad(debris2['RAAN'])

    # ── Phase T1: D1 orbit → disposal orbit ──
    # (solve_transfer 와 동일한 식 10 경로)
    m_total_T1       = m_SC + m_D1
    tof_T1           = tof_hohmann(h_D1, h_disp, params)
    m_prop_T1        = propellant_consumed_eq10(tof_T1, params)
    m_total_after_T1 = m_total_T1 - m_prop_T1
    dv_T1            = delta_v_from_mass(m_total_T1, m_prop_T1, params)
    m_SC_after_T1    = m_SC - m_prop_T1

    # ── Phase T2,a + Tp + T2,b : 사용자 지정 h_P 로 단일 평가 ──
    if h_P_km < h_disp:
        raise ValueError(
            f"H_P_KM ({h_P_km:.1f} km) 은 disposal 고도 "
            f"({h_disp:.1f} km) 이상이어야 합니다."
        )

    J, phasing = evaluate_phasing_orbit(
        h_P_km     = float(h_P_km),
        h_D1_km    = h_D1,
        RAAN_D1_0  = RAAN_D1,
        h_disp_km  = h_disp,
        h_D2_km    = h_D2,
        RAAN_D2_0  = RAAN_D2,
        m_after_T1 = m_SC_after_T1,
        params     = params,
        alpha      = alpha,
        i_rad      = i_rad,
    )

    if not np.isfinite(J) or not phasing:
        raise RuntimeError(
            f"H_P_KM = {h_P_km:.1f} km 에서 phasing 불가 (Tp < 0). "
            f"다른 고도를 선택하세요. "
            f"test_J.py 로 가능한 범위를 먼저 확인할 수 있습니다."
        )

    # ── 추진제 소비 합산 (solve_transfer 동일) ──
    m_prop_T2a   = m_SC_after_T1            - phasing['m_after_T2a']
    m_prop_T2b   = phasing['m_after_T2a']   - phasing['m_after_T2b']
    m_prop_drag  = phasing['m_after_T2b']   - phasing['m_final']
    m_prop_total = m_prop_T1 + m_prop_T2a + m_prop_T2b + m_prop_drag

    TOF = tof_T1 + phasing['tof_T2a'] + phasing['Tp'] + phasing['tof_T2b'] + Ts

    # solve_transfer 와 동일한 dict 구조로 반환 → plot_transfer 그대로 사용 가능
    return {
        'T1'      : tof_T1,
        'T2a'     : phasing['tof_T2a'],
        'Tp'      : phasing['Tp'],
        'T2b'     : phasing['tof_T2b'],
        'Ts'      : Ts,
        'TOF'     : TOF,
        'dv_T1'   : dv_T1,
        'dv_T2a'  : phasing['dv_T2a'],
        'dv_T2b'  : phasing['dv_T2b'],
        'dv_drag' : phasing['dv_drag_P'],
        'm_prop'  : m_prop_total,
        'm_SC_start'    : m_SC,
        'm_SC_end'      : phasing['m_final'],
        'h_P_km'        : phasing['h_P_km'],
        'delta_RAAN_rad': phasing['delta_RAAN'],
        'J'             : J,
        'phase': {
            'T1' : {
                'h_start': h_D1, 'h_end': h_disp,
                'dv': dv_T1, 'tof': tof_T1,
                'm_start': m_total_T1, 'm_end': m_total_after_T1,
            },
            'T2a': {
                'h_start': h_disp, 'h_end': phasing['h_P_km'],
                'dv': phasing['dv_T2a'], 'tof': phasing['tof_T2a'],
                'm_start': m_SC_after_T1, 'm_end': phasing['m_after_T2a'],
            },
            'Tp' : {
                'h': phasing['h_P_km'],
                'tof': phasing['Tp'],
                'm_start': phasing['m_after_T2a'], 'm_end': phasing['m_after_T2b'],
            },
            'T2b': {
                'h_start': phasing['h_P_km'], 'h_end': h_D2,
                'dv': phasing['dv_T2b'], 'tof': phasing['tof_T2b'],
                'm_start': phasing['m_after_T2b'], 'm_end': phasing['m_final'],
            },
            'Ts' : {
                'h': h_D2,
                'tof': Ts,
                'm_start': phasing['m_final'], 'm_end': phasing['m_final'],
            },
        }
    }


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    params, sim_cfg = load_params(
        constants_path='configs/constants.yaml',
        simulation_path='configs/simulation.yaml',
    )
    debris_all = load_debris_list(sim_cfg['training_data']['debris_yaml'])

    pair    = sim_cfg['visualization']['debris_pair']
    d1_name = pair[0]
    d2_name = pair[1]
    alpha   = sim_cfg['training_data']['alpha']

    params['alpha'] = alpha

    d1 = debris_all[d1_name]
    d2 = debris_all[d2_name]
    d1['name'] = d1_name
    d2['name'] = d2_name

    print(f"\n{'='*60}")
    print(f"Transfer (manual h_P): {d1_name} → disposal → phasing → {d2_name}")
    print(f"  D1: alt={d1['alt0_km']:.1f} km, RAAN={d1['RAAN']:.1f}°, mass={d1['mass']:.1f} kg")
    print(f"  D2: alt={d2['alt0_km']:.1f} km, RAAN={d2['RAAN']:.1f}°, mass={d2['mass']:.1f} kg")
    print(f"  α       = {alpha}")
    print(f"  h_P (manual) = {H_P_KM:.1f} km")
    print(f"{'='*60}")

    m_SC = params['m0']

    # 사용자 지정 h_P 로 transfer 계산
    result = solve_transfer_fixed_hP(d1, d2, m_SC, params, alpha, H_P_KM)

    # 결과 출력
    print(f"\n결과 (h_P = {H_P_KM:.1f} km 고정):")
    print(f"  T1  (D1→disposal)     : {result['T1']/86400:.1f} days,  ΔV={result['dv_T1']:.1f} m/s")
    print(f"  T2a (disposal→Phasing): {result['T2a']/86400:.1f} days,  ΔV={result['dv_T2a']:.1f} m/s")
    print(f"  Tp  (phasing wait)    : {result['Tp']/86400:.1f} days")
    print(f"  T2b (phasing→D2)      : {result['T2b']/86400:.1f} days,  ΔV={result['dv_T2b']:.1f} m/s")
    print(f"  Ts  (stay at D2)      : {result['Ts']/86400:.1f} days")
    print(f"  ─────────────────────────────")
    print(f"  TOF total             : {result['TOF']/86400:.1f} days")
    print(f"  ΔRAAN needed          : {np.rad2deg(result['delta_RAAN_rad']):.2f}°")
    print(f"  ΔV_drag (phasing)     : {result['dv_drag']:.2f} m/s")
    print(f"  m_prop total          : {result['m_prop']:.3f} kg")
    print(f"  m_SC start/end        : {result['m_SC_start']:.1f} / {result['m_SC_end']:.1f} kg")
    print(f"  J (this h_P)          : {result['J']:.4f}")

    # 시각화 (기존 plot_transfer 그대로 사용)
    plot_transfer(result, d1, d2, params)


if __name__ == '__main__':
    main()