"""
visualize_transfer.py
=====================
논문 Fig. 2를 재현한다.
debris0001 → disposal → phasing → debris0002 전이를 계산하고
4-패널 그래프(고도, RAAN, Thrust ON/OFF, 질량)를 출력한다.

실행:
  cd adr_project
  python scripts/visualize_transfer.py
"""

import sys
import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import (
    solve_transfer, raan_drift_rate,
    delta_v_altitude_change, mass_after_burn,
    evaluate_phasing_orbit, tof_tsiolkovsky,
    propellant_consumed_eq10, delta_v_from_mass,
    drag_dv_circular,
)


# ────────────────────────────────────────────────
# h_P 고정 버전 transfer 계산 (manual mode)
# ────────────────────────────────────────────────

def solve_transfer_fixed_hP(debris1, debris2, m_SC, params, alpha, h_P_km):
    """
    solve_transfer 와 동일한 result dict 를 반환하되
    phasing orbit 고도 h_P_km 를 사용자가 직접 지정한다.

    내부 구조는 transfer_solver.solve_transfer 와 동일하지만
    optimize_phasing_orbit (grid search) 대신
    evaluate_phasing_orbit 를 단일 h_P_km 에서 한 번만 호출한다.

    Parameters
    ----------
    debris1, debris2 : dict     debris 정보
    m_SC             : float    chaser 초기 질량 [kg]
    params           : dict
    alpha            : float    목적함수 가중치 (J 출력용 — transfer 계산 자체엔 영향 없음)
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

    # ── Phase T1: D1 orbit → disposal orbit (Tsiolkovsky 해석해 + 식 10) ──
    m_total_T1       = m_SC + m_D1
    tof_T1           = tof_tsiolkovsky(h_D1, h_disp, m_total_T1, params)
    m_prop_T1        = propellant_consumed_eq10(tof_T1, params)
    m_total_after_T1 = m_total_T1 - m_prop_T1
    dv_T1            = delta_v_from_mass(m_total_T1, m_prop_T1, params)
    m_SC_after_T1    = m_SC - m_prop_T1

    # ── 사전 검증 ──
    if h_P_km < h_disp:
        raise ValueError(
            f"h_P_km ({h_P_km:.1f} km) 은 disposal 고도 "
            f"({h_disp:.1f} km) 이상이어야 합니다."
        )

    # ── Phase T2,a + Tp + T2,b : 사용자 지정 h_P 로 단일 평가 ──
    J, phasing = evaluate_phasing_orbit(
        h_P_km     = float(h_P_km),
        h_D1_km    = h_D1,
        RAAN_D1_0  = RAAN_D1,
        m_D1       = m_D1,
        h_disp_km  = h_disp,
        h_D2_km    = h_D2,
        RAAN_D2_0  = RAAN_D2,
        m_SC_start = m_SC,
        params     = params,
        alpha      = alpha,
        i_rad      = i_rad,
    )

    if not np.isfinite(J) or not phasing:
        raise RuntimeError(
            f"h_P_km = {h_P_km:.1f} km 에서 phasing 불가 (Tp < 0). "
            f"다른 고도를 선택하세요. "
            f"(test_J.py 로 가능한 범위를 먼저 확인할 수 있습니다.)"
        )

    # ── 추진제 소비 합산 (chronological 순서: T2a → Tp(drag) → T2b → Ts(drag)) ──
    m_prop_T2a   = m_SC_after_T1            - phasing['m_after_T2a']
    m_prop_drag  = phasing['m_after_T2a']   - phasing['m_after_drag']
    m_prop_T2b   = phasing['m_after_drag']  - phasing['m_final']

    # Ts (D2 stay) 동안 대기항력 보상
    dv_Ts_drag   = drag_dv_circular(h_D2, Ts, phasing['m_final'], params)
    m_SC_end     = mass_after_burn(phasing['m_final'], dv_Ts_drag, params)
    m_prop_Ts    = phasing['m_final'] - m_SC_end

    m_prop_total = (m_prop_T1 + m_prop_T2a + m_prop_drag
                    + m_prop_T2b + m_prop_Ts)

    TOF = tof_T1 + phasing['tof_T2a'] + phasing['Tp'] + phasing['tof_T2b'] + Ts

    # solve_transfer 와 동일한 dict 구조로 반환 → plot_transfer/build_timeline 그대로 사용 가능
    return {
        'T1'      : tof_T1,
        'T2a'     : phasing['tof_T2a'],
        'Tp'      : phasing['Tp'],
        'T2b'     : phasing['tof_T2b'],
        'Ts'      : Ts,
        'TOF'     : TOF,
        'dv_T1'      : dv_T1,
        'dv_T2a'     : phasing['dv_T2a'],
        'dv_T2b'     : phasing['dv_T2b'],
        'dv_drag'    : phasing['dv_drag_P'],   # Tp drag
        'dv_Ts_drag' : dv_Ts_drag,              # Ts drag
        'm_prop'     : m_prop_total,
        'm_SC_start' : m_SC,
        'm_SC_end'   : m_SC_end,
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
                # Tp 동안의 drag 보상 (chronological 첫 번째)
                'dv': phasing['dv_drag_P'],
                'm_start': phasing['m_after_T2a'], 'm_end': phasing['m_after_drag'],
            },
            'T2b': {
                'h_start': phasing['h_P_km'], 'h_end': h_D2,
                'dv': phasing['dv_T2b'], 'tof': phasing['tof_T2b'],
                'm_start': phasing['m_after_drag'], 'm_end': phasing['m_final'],
            },
            'Ts' : {
                'h': h_D2,
                'tof': Ts,
                # Ts 동안의 drag 보상
                'dv': dv_Ts_drag,
                'm_start': phasing['m_final'], 'm_end': m_SC_end,
            },
        }
    }


# ────────────────────────────────────────────────
# RAAN 시간 적분 헬퍼 (해석적 평균 Ω̇ 근사 대신 사용)
# ────────────────────────────────────────────────
#
# 추력 구간(T1, T2a, T2b)에서는 a(t)가 시간에 따라 변하므로
# 매 시각의 Ω̇(a(t))를 평가하고 시간 적분(누적 사다리꼴) 한다.
# 추력 OFF 구간(Tp, Ts)에서는 a 일정 → Ω̇·t 로 단순.
#
# 구간 시작점에서의 누적 RAAN 변화를 0으로 하고, 끝점까지의
# 누적값(rad)을 길이 n_sample 의 ndarray 로 반환한다.

def _integrate_raan_thrust_leg(a_start, a_end, T_leg, e, i_rad, params, n_sample):
    """
    추력 구간의 RAAN 변화량 누적 시계열을 식 (22) 시간 적분으로 계산.
    a(t) 는 a_start → a_end 로 시간 선형 보간 (저추력 근사).

    Returns
    -------
    raan_inc : ndarray [rad], 길이 n_sample, 시작값 0
    """
    t_arr = np.linspace(0.0, T_leg, n_sample)
    if T_leg <= 0:
        return np.zeros_like(t_arr)
    a_arr = a_start + (a_end - a_start) * (t_arr / T_leg)
    omega_dot = np.array([
        raan_drift_rate(a, e, i_rad, params) for a in a_arr
    ])
    raan_inc = np.zeros_like(t_arr)
    raan_inc[1:] = np.cumsum(0.5 * (omega_dot[:-1] + omega_dot[1:]) * np.diff(t_arr))
    return raan_inc


def _integrate_raan_constant_leg(a_const, T_leg, e, i_rad, params, n_sample):
    """
    추력 OFF 구간 (a 일정) 의 RAAN 누적 변화. Ω̇·t 의 해석적 형태이지만,
    인터페이스를 _integrate_raan_thrust_leg 와 동일하게 맞춤.
    """
    t_arr = np.linspace(0.0, T_leg, n_sample)
    if T_leg <= 0:
        return np.zeros_like(t_arr)
    od = raan_drift_rate(a_const, e, i_rad, params)
    return od * t_arr


# ────────────────────────────────────────────────
# 시각화용 궤적 재구성
# ────────────────────────────────────────────────

def build_timeline(result, debris1, debris2, params):
    """
    solve_transfer 결과로부터 시간-고도/RAAN/추력/질량 이력을 생성한다.
    논문 Fig. 2와 동일한 4개 패널용 데이터.

    각 구간을 시간에 따라 선형 보간하여 연속 곡선으로 표현한다.
    RAAN은 각 구간에서 해석적으로 전파한다 (식 22 기반).

    Returns
    -------
    t        : ndarray [days]
    alt      : ndarray [km]
    RAAN_SC  : ndarray [deg]
    RAAN_D2  : ndarray [deg]   (D2의 RAAN 전파값, 비교용)
    thrust   : ndarray [0/1]   (1=ON, 0=OFF)
    mass     : ndarray [kg]
    segments : list of dict    (각 구간 경계 정보)
    """
    i_rad  = np.deg2rad(params['inclination_deg'])
    Re     = params['Re']
    N_pts  = 300   # 전체 포인트 수

    phase  = result['phase']
    T1     = result['T1']
    T2a    = result['T2a']
    Tp     = result['Tp']
    T2b    = result['T2b']
    Ts     = result['Ts']
    TOF    = result['TOF']

    # 각 구간 시간 경계 [s]
    t_bounds = [
        0,
        T1,
        T1 + T2a,
        T1 + T2a + Tp,
        T1 + T2a + Tp + T2b,
        TOF
    ]
    labels = ['T1', 'T2a', 'Tp', 'T2b', 'Ts']

    # 포인트 배분 (시간 비율에 따라)
    n_pts_list = []
    for k in range(5):
        dt = t_bounds[k+1] - t_bounds[k]
        n  = max(2, int(N_pts * dt / TOF))
        n_pts_list.append(n)

    # ── 고도 이력 ──
    # T1 : h_D1 → h_disp  (선형 근사)
    # T2a: h_disp → h_P   (선형 근사)
    # Tp : h_P (일정)
    # T2b: h_P → h_D2     (선형 근사)
    # Ts : h_D2 (일정)
    h_segs = [
        np.linspace(debris1['alt0_km'], params['disposal_alt_km'], n_pts_list[0]),
        np.linspace(params['disposal_alt_km'], result['h_P_km'],   n_pts_list[1]),
        np.full(n_pts_list[2], result['h_P_km']),
        np.linspace(result['h_P_km'], debris2['alt0_km'],          n_pts_list[3]),
        np.full(n_pts_list[4], debris2['alt0_km']),
    ]

    # ── 질량 이력 ──
    # T1: chaser+D1 합산 질량이 추진제 소비만큼 감소
    #     T1 마지막 포인트에서 D1 질량이 즉시 드롭 (점프)
    #     → T1 구간 마지막을 m_end_T1(합산)으로 끝내고,
    #       T2a 구간 첫 포인트를 m_start_T2a(chaser 단독)로 시작
    m_segs = []

    # T1: chaser+D1 연소 (추진제만 감소, D1은 T1 끝에 방출)
    m_start_T1 = phase['T1']['m_start']   # chaser+D1+propellant
    m_end_T1   = phase['T1']['m_end']     # chaser+D1 (추진제 소비 후)
    m_segs.append(np.linspace(m_start_T1, m_end_T1, n_pts_list[0]))
    # → T1 마지막값 = m_end_T1 (합산)
    # → T2a 첫값   = m_start_T2a (chaser 단독, D1 방출 후 즉시 점프)

    # T2a: chaser 단독 연소
    m_start_T2a = phase['T2a']['m_start']  # D1 방출 직후 chaser 단독 질량
    m_end_T2a   = phase['T2a']['m_end']
    m_segs.append(np.linspace(m_start_T2a, m_end_T2a, n_pts_list[1]))

    # Tp: thrust OFF, drag 보정 연소 (미세 감소)
    m_start_Tp  = phase['Tp']['m_start']
    m_end_Tp    = phase['Tp']['m_end']
    m_segs.append(np.linspace(m_start_Tp, m_end_Tp, n_pts_list[2]))

    # T2b: chaser 연소
    m_start_T2b = phase['T2b']['m_start']
    m_end_T2b   = phase['T2b']['m_end']
    m_segs.append(np.linspace(m_start_T2b, m_end_T2b, n_pts_list[3]))

    # Ts: thrust ON (drag 보상), 질량 미세 감소
    m_segs.append(np.linspace(phase['Ts']['m_start'], phase['Ts']['m_end'], n_pts_list[4]))

    # ── RAAN 이력 (시간 적분 기반) ──
    # 추력 구간(T1, T2a, T2b): a(t) 시간 선형 보간 + Ω̇(a(t)) 사다리꼴 적분
    # 추력 OFF 구간(Tp, Ts):   a 일정 → Ω̇·t
    # 모든 누적값은 rad 로 계산하고 마지막에 deg 로 변환.
    e = 0.0   # 원궤도 가정 (debris.yaml 모두 e=0)

    a_D1   = Re + debris1['alt0_km'] * 1e3
    a_disp = Re + params['disposal_alt_km'] * 1e3
    a_P    = Re + result['h_P_km'] * 1e3
    a_D2   = Re + debris2['alt0_km'] * 1e3

    # D2 의 RAAN drift rate (D2 는 항상 같은 궤도, 일정)
    od_D2 = raan_drift_rate(a_D2, e, i_rad, params)

    # 각 구간 누적 RAAN 증가량 [rad], 길이 = n_pts_list[k]
    seg_a = [
        ('thrust',   a_D1,   a_disp, T1),    # T1
        ('thrust',   a_disp, a_P,    T2a),   # T2,a
        ('constant', a_P,    a_P,    Tp),    # Tp
        ('thrust',   a_P,    a_D2,   T2b),   # T2,b
        ('constant', a_D2,   a_D2,   Ts),    # Ts
    ]
    dRAAN_segs_rad = []
    for k, (kind, a_s, a_e, T_leg) in enumerate(seg_a):
        n = n_pts_list[k]
        if kind == 'thrust':
            dRAAN_segs_rad.append(
                _integrate_raan_thrust_leg(a_s, a_e, T_leg, e, i_rad, params, n)
            )
        else:
            dRAAN_segs_rad.append(
                _integrate_raan_constant_leg(a_s, T_leg, e, i_rad, params, n)
            )

    # 누적 RAAN (chaser, D2) — 각 구간을 이어붙임
    RAAN_0    = np.deg2rad(debris1['RAAN'])    # chaser 시작 = D1 의 RAAN
    RAAN_D2_0 = np.deg2rad(debris2['RAAN'])

    RAAN_segs    = []
    RAAN_D2_segs = []
    RAAN_cur     = RAAN_0
    RAAN_D2_cur  = RAAN_D2_0

    for k, T_leg in enumerate([T1, T2a, Tp, T2b, Ts]):
        n        = n_pts_list[k]
        t_local  = np.linspace(0.0, T_leg, n)

        # chaser : 적분으로 구한 누적 ΔΩ 사용
        sc_seg   = RAAN_cur + dRAAN_segs_rad[k]
        # D2 : 항상 일정한 Ω̇_D2
        d2_seg   = RAAN_D2_cur + od_D2 * t_local

        RAAN_segs.append   (np.rad2deg(sc_seg))
        RAAN_D2_segs.append(np.rad2deg(d2_seg))

        # 다음 구간 시작값
        RAAN_cur    = sc_seg[-1]
        RAAN_D2_cur = d2_seg[-1]

    # ── Thrust ON/OFF ──
    thrust_segs = [
        np.ones(n_pts_list[0]),    # T1: ON
        np.ones(n_pts_list[1]),    # T2a: ON (0이면 OFF)
        np.zeros(n_pts_list[2]),   # Tp: OFF
        np.ones(n_pts_list[3]),    # T2b: ON
        np.zeros(n_pts_list[4]),   # Ts: OFF
    ]
    # T2a=0인 경우 (phasing = disposal orbit)
    if T2a < 1.0:
        thrust_segs[1] = np.zeros(n_pts_list[1])

    # ── 시간 축 생성 [days] ──
    t_segs = []
    t_cur  = 0.0
    for k, (dt_total, n) in enumerate(zip([T1, T2a, Tp, T2b, Ts], n_pts_list)):
        t_segs.append(np.linspace(t_cur, t_cur + dt_total, n) / 86400.0)
        t_cur += dt_total

    # 연결 (마지막 포인트 중복 제거)
    def concat_segs(segs):
        result_arr = [segs[0]]
        for s in segs[1:]:
            result_arr.append(s[1:])
        return np.concatenate(result_arr)

    t       = concat_segs(t_segs)
    alt     = concat_segs(h_segs)
    RAAN_SC = concat_segs(RAAN_segs)
    RAAN_D2_arr = concat_segs(RAAN_D2_segs)
    thrust  = concat_segs(thrust_segs)

    # 질량은 T1/T2a 경계에서 점프가 있으므로 별도 처리.
    # mass_raw 에는 이미 T1 마지막 점(=m_end_T1, chaser+D1 합산) 이 들어있다.
    # T2a 시작 점(=m_start_T2a, chaser 단독) 만 같은 시각에 추가로 삽입하면
    # matplotlib 이 두 점 사이를 수직 선분으로 그려 D1 방출 점프를 표현한다.
    # (이전엔 [m_high, m_low] 두 점을 모두 삽입해서 m_high 가 두 번 나타났고,
    #  그 결과 D1 방출 후에도 한동안 합산 질량으로 머무르는 비스듬한 선이
    #  그려지는 버그가 있었음.)
    mass_raw   = concat_segs(m_segs)
    t_jump     = t_segs[0][-1]           # T1 끝 시각 [days]
    m_low      = m_segs[1][0]            # T2a 시작 질량 (chaser 단독, 낮은 값)
    # concat_segs 결과에서 t_jump 의 위치 (T1 마지막 점 바로 다음)
    jump_idx   = np.searchsorted(t, t_jump, side='right')
    mass       = np.insert(mass_raw, jump_idx, [m_low])
    t          = np.insert(t,        jump_idx, [t_jump])
    alt        = np.insert(alt,      jump_idx, [alt[jump_idx-1]])
    RAAN_SC    = np.insert(RAAN_SC,  jump_idx, [RAAN_SC[jump_idx-1]])
    RAAN_D2_arr= np.insert(RAAN_D2_arr, jump_idx, [RAAN_D2_arr[jump_idx-1]])
    thrust     = np.insert(thrust,   jump_idx, [thrust[jump_idx-1]])

    # ── 누적 추진제 m_prop 이력 ──
    # phase dict 의 mass 변화는 chronological 순서이므로 그대로 합산.
    #   T1   : chaser+D1 시스템의 T1 추력 소비
    #   T2a  : chaser 의 T2a 추력 소비
    #   Tp   : phasing 동안 drag 보상
    #   T2b  : chaser 의 T2b 추력 소비
    #   Ts   : D2 stay 동안 drag 보상
    dm_T1     = phase['T1']['m_start']  - phase['T1']['m_end']
    dm_T2a    = phase['T2a']['m_start'] - phase['T2a']['m_end']
    dm_drag   = phase['Tp']['m_start']  - phase['Tp']['m_end']
    dm_T2b    = phase['T2b']['m_start'] - phase['T2b']['m_end']
    dm_Ts     = phase['Ts']['m_start']  - phase['Ts']['m_end']
    dm_phase  = [dm_T1, dm_T2a, dm_drag, dm_T2b, dm_Ts]

    # 각 phase 시작 시의 누적 m_prop
    cum_starts = np.concatenate([[0.0], np.cumsum(dm_phase)[:-1]])
    m_prop_segs = [
        np.linspace(cum_starts[k], cum_starts[k] + dm_phase[k], n_pts_list[k])
        for k in range(5)
    ]
    m_prop_raw = concat_segs(m_prop_segs)
    # D1 방출 시점에서 m_prop 은 점프 없음 (debris 방출 ≠ 추진제 소비) →
    # 이전 값을 그대로 한 번 더 삽입해 t/alt/mass 와 인덱스 정렬만 맞춤.
    m_prop_cum = np.insert(m_prop_raw, jump_idx, [m_prop_raw[jump_idx - 1]])

    # 구간 경계 [days]
    t_bd = np.array(t_bounds) / 86400.0
    segments = [
        {'label': lbl, 't_start': t_bd[k], 't_end': t_bd[k+1]}
        for k, lbl in enumerate(labels)
    ]

    return t, alt, RAAN_SC, RAAN_D2_arr, thrust, mass, m_prop_cum, segments


# ────────────────────────────────────────────────
# 그래프 출력 (논문 Fig. 2 재현)
# ────────────────────────────────────────────────

def plot_transfer(result, debris1, debris2, params):
    """
    논문 Fig. 2와 동일한 4-패널 그래프를 생성한다.

    패널 1: 고도 [km] vs 시간 [days]
    패널 2: RAAN [deg] vs 시간
    패널 3: Thrust ON/OFF vs 시간
    패널 4: 누적 추진제 소비량 m_prop [kg] vs 시간
    """
    t, alt, RAAN_SC, RAAN_D2, thrust, mass, m_prop_cum, segments = \
        build_timeline(result, debris1, debris2, params)

    h_disp = params['disposal_alt_km']
    h_P    = result['h_P_km']
    TOF_d  = result['TOF'] / 86400.0

    # 구간 경계 시각 [days]
    t_T1_end  = result['T1'] / 86400.0
    t_T2a_end = (result['T1'] + result['T2a']) / 86400.0
    t_Tp_end  = (result['T1'] + result['T2a'] + result['Tp']) / 86400.0
    t_T2b_end = (result['T1'] + result['T2a'] + result['Tp'] + result['T2b']) / 86400.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f"Transfer: {debris1.get('name','D1')} → disposal → phasing → {debris2.get('name','D2')}\n"
        f"TOF = {TOF_d:.1f} days  |  "
        f"m_prop = {result['m_prop']:.2f} kg  |  "
        f"h_P = {h_P:.1f} km  |  "
        f"α = {params.get('alpha', 0.5)}",
        fontsize=11, y=0.98
    )

    # ── 구간 배경색 ──
    seg_colors = ['#d4e8ff', '#c8f0c8', '#fff3cc', '#c8f0c8', '#f0d4f0']
    seg_labels_kr = ['T1\n(Descent)', 'T2a\n(→Phasing)', 'Tp\n(phasing)', 'T2b\n(→D2)', 'Ts\n(Coast)']
    t_bounds_d = [0, t_T1_end, t_T2a_end, t_Tp_end, t_T2b_end, TOF_d]

    def draw_backgrounds(ax):
        for k in range(5):
            ax.axvspan(t_bounds_d[k], t_bounds_d[k+1],
                       alpha=0.3, color=seg_colors[k], linewidth=0)

    def draw_vlines(ax):
        for t_bd in [t_T1_end, t_T2a_end, t_Tp_end, t_T2b_end]:
            ax.axvline(t_bd, color='gray', ls='--', lw=0.8, alpha=0.7)

    # ── 패널 1: 고도 ──
    ax = axes[0]
    draw_backgrounds(ax)
    draw_vlines(ax)
    ax.plot(t, alt, 'b-', lw=2, label='Chaser altitude')
    ax.axhline(h_disp, color='r',  ls=':', lw=1.2, label=f'Disposal ({h_disp:.0f} km)')
    ax.axhline(h_P,    color='g',  ls=':', lw=1.2, label=f'Phasing ({h_P:.1f} km)')
    ax.axhline(debris1['alt0_km'], color='orange', ls=':', lw=1.0,
               label=f'D1 ({debris1["alt0_km"]:.1f} km)')
    ax.axhline(debris2['alt0_km'], color='purple', ls=':', lw=1.0,
               label=f'D2 ({debris2["alt0_km"]:.1f} km)')
    # D1 Release 시점 표시
    ax.axvline(t_T1_end, color='red', lw=1.5, alpha=0.8)
    ax.annotate('D1 Release', xy=(t_T1_end, h_disp), xytext=(t_T1_end + TOF_d*0.02, h_disp + 50),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.set_ylabel('Altitude [km]')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 구간 레이블 추가
    for k in range(5):
        t_mid = (t_bounds_d[k] + t_bounds_d[k+1]) / 2
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else debris1['alt0_km'] * 1.05
        ax.text(t_mid, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02,
                seg_labels_kr[k], ha='center', fontsize=7, color='#333333')

    # ── 패널 2: RAAN ──
    # 0~360 범위로 래핑하고, 점프 구간은 NaN으로 끊어서 선이 이어지지 않게 함
    def wrap_360_with_breaks(arr):
        """
        누적 RAAN을 0~360 범위로 mod 변환하되,
        인접 포인트 간 차이가 180° 이상이면 불연속(점프)으로 간주해
        NaN을 삽입하여 선을 끊는다.
        """
        wrapped = arr % 360.0
        result  = wrapped.copy().astype(float)
        diff    = np.abs(np.diff(wrapped))
        jumps   = np.where(diff > 180)[0]
        for j in jumps:
            result = np.insert(result, j + 1, np.nan)
        t_out = t.copy()
        for j in jumps:
            t_out = np.insert(t_out, j + 1, t[j])   # 시간 축도 맞춰 삽입
        return t_out, result

    t_sc,  raan_sc_w  = wrap_360_with_breaks(RAAN_SC)
    t_d2,  raan_d2_w  = wrap_360_with_breaks(RAAN_D2)

    ax = axes[1]
    draw_backgrounds(ax)
    draw_vlines(ax)
    ax.plot(t_sc, raan_sc_w, 'b-',  lw=2,   label='Ω_SC (chaser)')
    ax.plot(t_d2, raan_d2_w, 'r--', lw=1.5, label='Ω_D2 (debris 2)')
    ax.axhline(debris1['RAAN'] % 360, color='orange', ls=':', lw=1.0,
               label=f'Ω_D1 ({debris1["RAAN"] % 360:.1f}°)')
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel('RAAN [deg]')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── 패널 3: Thrust ON/OFF ──
    ax = axes[2]
    draw_backgrounds(ax)
    draw_vlines(ax)
    ax.fill_between(t, 0, thrust, step='mid', alpha=0.7, color='steelblue', label='Thrust')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OFF', 'ON'])
    ax.set_ylim(-0.1, 1.3)
    ax.set_ylabel('Thrust')
    ax.grid(True, alpha=0.3, axis='x')

    # ── 패널 4: 누적 추진제 소비량 m_prop ──
    ax = axes[3]
    draw_backgrounds(ax)
    draw_vlines(ax)
    ax.plot(t, m_prop_cum, 'b-', lw=2, label='Cumulative m_prop')

    # phase 별 추진제 소비량 (phase dict 이 이미 chronological)
    phase_d  = result['phase']
    dm_T1    = phase_d['T1']['m_start']  - phase_d['T1']['m_end']
    dm_T2a   = phase_d['T2a']['m_start'] - phase_d['T2a']['m_end']
    dm_drag  = phase_d['Tp']['m_start']  - phase_d['Tp']['m_end']
    dm_T2b   = phase_d['T2b']['m_start'] - phase_d['T2b']['m_end']
    dm_Ts    = phase_d['Ts']['m_start']  - phase_d['Ts']['m_end']
    m_prop_total = result['m_prop']

    # 각 phase 끝점에서의 누적값을 점선으로 표시
    cum_end = [dm_T1,
               dm_T1 + dm_T2a,
               dm_T1 + dm_T2a + dm_drag,
               dm_T1 + dm_T2a + dm_drag + dm_T2b]
    for c in cum_end:
        ax.axhline(c, color='#888', ls=':', lw=0.7, alpha=0.5)

    # 총량 가이드 라인
    ax.axhline(m_prop_total, color='red', ls='--', lw=0.8, alpha=0.7,
               label=f'total = {m_prop_total:.2f} kg')

    # phase 별 소비량 텍스트
    seg_dm     = [dm_T1, dm_T2a, dm_drag, dm_T2b, dm_Ts]
    seg_lbls   = ['T1', 'T2a', 'Tp(drag)', 'T2b', 'Ts(drag)']
    cum_starts_plot = np.concatenate([[0.0], np.cumsum(seg_dm)[:-1]])
    # 라벨 표시 임계값 — 너무 작아서 보이지 않는 소비는 라벨 생략
    dm_label_threshold = max(1e-4, m_prop_total * 1e-3)
    for k in range(5):
        if seg_dm[k] < dm_label_threshold:
            continue
        t_mid   = (t_bounds_d[k] + t_bounds_d[k+1]) / 2
        cum_mid = cum_starts_plot[k] + seg_dm[k] / 2
        ax.text(t_mid, cum_mid,
                f'{seg_lbls[k]}\n+{seg_dm[k]:.2f}kg',
                ha='center', va='center', fontsize=7, color='#222',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray',
                          lw=0.5, alpha=0.85))

    ax.set_ylabel('Cumulative m_prop [kg]')
    ax.set_xlabel('Time [days]')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    # y 범위에 여유 확보 (라벨/타이틀 가독성)
    y_top = max(m_prop_total * 1.15, m_prop_total + 0.5)
    ax.set_ylim(-m_prop_total * 0.05, y_top)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────

def _resolve_h_P_setting(viz_cfg):
    """
    simulation.yaml 의 visualization.h_P_km 값을 해석한다.

      "optimal"  → ('optimal', None)
      <숫자>    → ('fixed',   float(h_P))
      그 외     → ValueError

    키 자체가 없으면 기본 'optimal' 로 처리.
    """
    raw = viz_cfg.get('h_P_km', 'optimal')

    if isinstance(raw, str):
        if raw.strip().lower() == 'optimal':
            return 'optimal', None
        # 숫자 문자열도 허용 (예: "500" → 500.0)
        try:
            return 'fixed', float(raw)
        except ValueError:
            raise ValueError(
                f"visualization.h_P_km 의 문자열 값은 'optimal' 또는 숫자만 허용됩니다. "
                f"입력값: {raw!r}"
            )

    if isinstance(raw, (int, float)):
        return 'fixed', float(raw)

    raise ValueError(
        f"visualization.h_P_km 형식이 잘못됨: {raw!r} (type={type(raw).__name__}). "
        f"'optimal' 또는 숫자를 사용하세요."
    )


def main():
    # 설정 로드
    params, sim_cfg = load_params(
        constants_path='configs/constants.yaml',
        simulation_path='configs/simulation.yaml',
    )
    debris_all = load_debris_list(sim_cfg['visualization']['debris_file'])

    # 시각화 대상 debris 쌍
    viz_cfg = sim_cfg['visualization']
    pair    = viz_cfg['debris_pair']
    d1_name = pair[0]
    d2_name = pair[1]
    alpha   = sim_cfg['training_data']['alpha']

    params['alpha'] = alpha

    d1 = debris_all[d1_name]
    d2 = debris_all[d2_name]
    d1['name'] = d1_name
    d2['name'] = d2_name

    # h_P 모드 결정
    mode, h_P_user = _resolve_h_P_setting(viz_cfg)

    print(f"\n{'='*60}")
    print(f"Transfer: {d1_name} → disposal → phasing → {d2_name}")
    print(f"  D1: alt={d1['alt0_km']:.1f} km, RAAN={d1['RAAN']:.1f}°, mass={d1['mass']:.1f} kg")
    print(f"  D2: alt={d2['alt0_km']:.1f} km, RAAN={d2['RAAN']:.1f}°, mass={d2['mass']:.1f} kg")
    print(f"  α = {alpha}")
    if mode == 'optimal':
        print(f"  h_P : optimal (grid search 로 자동 탐색)")
    else:
        print(f"  h_P : {h_P_user:.1f} km (사용자 지정)")
    print(f"{'='*60}")

    m_SC = params['m0']

    # 전이 계산 (모드에 따라 분기)
    if mode == 'optimal':
        result = solve_transfer(d1, d2, m_SC, params, alpha)
    else:
        result = solve_transfer_fixed_hP(d1, d2, m_SC, params, alpha, h_P_user)

    # 결과 출력
    print(f"\n결과:")
    print(f"  T1  (D1→disposal)   : {result['T1']/86400:.1f} days,  ΔV={result['dv_T1']:.1f} m/s")
    print(f"  T2a (disposal→Phasing): {result['T2a']/86400:.1f} days,  ΔV={result['dv_T2a']:.1f} m/s")
    print(f"  Tp  (phasing wait)  : {result['Tp']/86400:.1f} days,  ΔV_drag={result['dv_drag']:.2f} m/s")
    print(f"  T2b (phasing→D2)    : {result['T2b']/86400:.1f} days,  ΔV={result['dv_T2b']:.1f} m/s")
    print(f"  Ts  (stay at D2)    : {result['Ts']/86400:.1f} days")
    print(f"  ─────────────────────────────")
    print(f"  TOF total           : {result['TOF']/86400:.1f} days")
    print(f"  h_P (phasing orbit) : {result['h_P_km']:.1f} km")
    print(f"  ΔRAAN needed        : {np.rad2deg(result['delta_RAAN_rad']):.2f}°")
    print(f"  m_prop total        : {result['m_prop']:.3f} kg")
    print(f"  m_SC start/end      : {result['m_SC_start']:.1f} / {result['m_SC_end']:.1f} kg")

    # 시각화
    plot_transfer(result, d1, d2, params)


if __name__ == '__main__':
    main()