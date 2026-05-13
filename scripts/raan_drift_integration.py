"""
raan_drift_integration.py
=========================
SC 와 D2 의 RAAN 을 식 (22), (23), (24) 를 직접 시간 적분하여 추적한다.
SC 와 D2 의 RAAN 차이가 ±1° (또는 ±361°, ±721°, ...) 이내가 되는 첫 시점을
찾고 그 시점까지의 두 RAAN 곡선을 그래프로 그린다.

식 정리 (논문) :
    Ω̇(a) = -(3/2) · J2 · sqrt(μ) · Re² / [a^(7/2) · (1-e²)²] · cos(i)        … (22)
    Ω_2,f  = Ω_2,0 + Ω̇_2 · (T1 + T2 + Tp)                                    … (23)
    Ω_SC,f = Ω_1,0 + Ω̇_T1·T1 + Ω̇_T2·T2 + Ω̇_p·Tp                              … (24)

본 스크립트는 식 (24) 의 각 항(T1, T2,a, T2,b, Tp 구간) 을
"평균 a 의 일정 Ω̇ × 구간 시간" 근사가 아니라
**a(t) 를 시간에 선형으로 보간하여 식 (22) 를 매 시각 평가** 한 뒤
시간에 대해 적분한다 (사다리꼴 / scipy quad).

"두 RAAN 이 다시 일치하는 시점" 의 정의 :
    Δ(t) = Ω_SC(t) - Ω_D2(t)   (누적 unwrap 값, deg)
    Δ(t) mod 360°  ∈  [-1°, +1°]   (= 0° 또는 360° 의 정수배에 ±1° 이내)
    이 조건을 처음으로 만족하는 t* 를 이분법으로 정밀 탐색.

설정 :
    - debris pair : configs/simulation.yaml 의 visualization.debris_pair 사용
    - h_P_km      : 스크립트 상단 H_P_KM 직접 지정 (기본 500 km)
    - tolerance   : ±1°
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import (
    tof_tsiolkovsky,
    propellant_consumed_eq10,
    delta_v_from_mass,
    raan_drift_rate,
    evaluate_phasing_orbit,
)


# ──────────────────────────────────────────────────────────────
# 사용자 설정
# ──────────────────────────────────────────────────────────────
H_P_KM        = 500.0    # phasing orbit 고도 [km]
TOL_DEG       = 1.0      # RAAN 일치 판정 톨러런스 [deg]
T_MAX_DAYS    = 5000.0   # 최대 탐색 시간 [days] (이 시간 안에 못 만나면 포기)
N_SAMPLES_LEG = 400      # 각 구간(T1,T2a,Tp,T2b)마다 시간 샘플 수
N_SAMPLES_TS  = 60       # Ts (D2 stay) 구간 샘플 수


# ──────────────────────────────────────────────────────────────
# 핵심 적분 루틴
# ──────────────────────────────────────────────────────────────
def integrate_raan_thrust_leg(a_start, a_end, T_leg, e, i_rad, params, n_sample):
    """
    추력 구간(T1, T2,a, T2,b)의 RAAN 변화량을 식 (22) 시간 적분으로 계산한다.

    a(t) 는 0 → T_leg 동안 a_start → a_end 로 시간 선형 보간 (저추력 근사).
    각 시각의 Ω̇(a(t)) 를 평가하고 사다리꼴 적분.

    Parameters
    ----------
    a_start, a_end : float [m]
    T_leg          : float [s]
    e              : float
    i_rad          : float [rad]
    params         : dict
    n_sample       : int   적분 샘플 수 (≥ 2)

    Returns
    -------
    t_arr      : ndarray [s], shape (n_sample,)   0 → T_leg
    raan_inc   : ndarray [rad], 누적 RAAN 변화 (시작점 0)
    omega_dot  : ndarray [rad/s], 각 시각 Ω̇
    """
    t_arr = np.linspace(0.0, T_leg, n_sample)
    if T_leg <= 0:
        return t_arr, np.zeros_like(t_arr), np.zeros_like(t_arr)

    # a(t) 선형 보간
    a_arr = a_start + (a_end - a_start) * (t_arr / T_leg)

    # 매 시각 Ω̇(a) 계산
    omega_dot = np.array([
        raan_drift_rate(a, e, i_rad, params) for a in a_arr
    ])

    # 누적 적분 (사다리꼴)
    # cumulative trapezoidal : raan_inc[k] = ∫_0^{t_k} Ω̇(τ) dτ
    raan_inc = np.zeros_like(t_arr)
    raan_inc[1:] = np.cumsum(0.5 * (omega_dot[:-1] + omega_dot[1:]) * np.diff(t_arr))

    return t_arr, raan_inc, omega_dot


def integrate_raan_constant_leg(a_const, T_leg, e, i_rad, params, n_sample):
    """
    추력 OFF 구간 (Tp, Ts) 의 RAAN 변화량.
    a 가 일정하므로 Ω̇ 도 일정 → 단순 선형 증가.
    적분 형식은 유지 (시간 샘플 + 누적값).
    """
    t_arr = np.linspace(0.0, T_leg, n_sample)
    if T_leg <= 0:
        return t_arr, np.zeros_like(t_arr), np.zeros_like(t_arr)

    od = raan_drift_rate(a_const, e, i_rad, params)
    raan_inc = od * t_arr
    omega_dot = np.full_like(t_arr, od)
    return t_arr, raan_inc, omega_dot


# ──────────────────────────────────────────────────────────────
# 전체 미션 동안 SC, D2 의 RAAN 시계열 생성
# ──────────────────────────────────────────────────────────────
def build_raan_timeseries(
    debris1, debris2, params, h_P_km, m_SC_start, alpha
):
    """
    T1 → T2,a → Tp → T2,b → Ts 구간을 순차적으로 적분하여
    chaser RAAN, D2 RAAN 의 누적 시계열을 만든다 (단위 rad, unwrap).

    각 구간 시간(T1, T2a, Tp, T2b, Ts) 은 transfer_solver 의
    solve_transfer 와 동일한 방식으로 계산한다.

    Returns
    -------
    t          : ndarray [s]                미션 시작부터 누적 시간
    raan_sc    : ndarray [rad]              chaser 누적 RAAN (unwrap)
    raan_d2    : ndarray [rad]              D2 누적 RAAN     (unwrap)
    seg_bounds : list of (t_start, t_end, label)   구간 경계 (그래프 색칠용)
    info       : dict                       추가 정보 (h_P, ΔRAAN_phasing 등)
    """
    Re      = params['Re']
    i_rad   = np.deg2rad(params['inclination_deg'])
    h_disp  = params['disposal_alt_km']
    e       = 0.0   # 원궤도 가정 (논문, debris.yaml 모두)

    # 반장축
    a_D1   = Re + debris1['alt0_km'] * 1e3
    a_disp = Re + h_disp             * 1e3
    a_P    = Re + h_P_km             * 1e3
    a_D2   = Re + debris2['alt0_km'] * 1e3

    # 구간별 시간 (transfer_solver 와 동일 로직)
    # T1 : D1 → disposal (chaser+D1 합산 질량으로 저추력)
    m_total_T1       = m_SC_start + debris1['mass']
    T1               = tof_tsiolkovsky(debris1['alt0_km'], h_disp, m_total_T1, params)

    # T1 후 chaser 단독 질량 (D1 방출) — 식 10 경로
    m_prop_T1        = propellant_consumed_eq10(T1, params)
    m_total_after_T1 = m_total_T1 - m_prop_T1
    m_SC_after_T1    = m_SC_start - m_prop_T1

    # phasing 단계 평가 (h_P 고정) → T2a, Tp, T2b 시간 얻음
    J, ph = evaluate_phasing_orbit(
        h_P_km     = float(h_P_km),
        h_D1_km    = debris1['alt0_km'],
        RAAN_D1_0  = np.deg2rad(debris1['RAAN']),
        m_D1       = debris1['mass'],
        h_disp_km  = h_disp,
        h_D2_km    = debris2['alt0_km'],
        RAAN_D2_0  = np.deg2rad(debris2['RAAN']),
        m_SC_start = m_SC_start,
        params     = params,
        alpha      = alpha,
        i_rad      = i_rad,
    )
    if not np.isfinite(J) or not ph:
        raise RuntimeError(
            f"H_P_KM = {h_P_km:.1f} km 에서 phasing 불가 (Tp < 0). "
            f"다른 고도를 선택하세요."
        )

    T2a = ph['tof_T2a']
    Tp  = ph['Tp']
    T2b = ph['tof_T2b']
    Ts  = 30.0 * 86400.0   # 30일 stay (논문)

    # ── 구간별 적분 ──
    legs = [
        # (label, leg type, args)
        ('T1',  'thrust',   a_D1,   a_disp, T1,  N_SAMPLES_LEG),
        ('T2a', 'thrust',   a_disp, a_P,    T2a, N_SAMPLES_LEG),
        ('Tp',  'constant', a_P,    a_P,    Tp,  N_SAMPLES_LEG),
        ('T2b', 'thrust',   a_P,    a_D2,   T2b, N_SAMPLES_LEG),
        ('Ts',  'constant', a_D2,   a_D2,   Ts,  N_SAMPLES_TS),
    ]

    t_global   = []
    raan_sc    = []   # chaser 누적 RAAN [rad]
    raan_d2    = []   # D2 누적 RAAN [rad]
    seg_bounds = []

    t_offset      = 0.0
    raan_sc_acc   = np.deg2rad(debris1['RAAN'])    # chaser 시작 RAAN = D1 의 RAAN
    raan_d2_acc   = np.deg2rad(debris2['RAAN'])    # D2 시작 RAAN

    od_d2 = raan_drift_rate(a_D2, e, i_rad, params)   # D2 는 항상 일정

    for label, kind, a_s, a_e, T_leg, n in legs:
        if T_leg <= 0:
            seg_bounds.append((t_offset, t_offset, label))
            continue

        if kind == 'thrust':
            t_loc, dRAAN_sc, _ = integrate_raan_thrust_leg(
                a_s, a_e, T_leg, e, i_rad, params, n
            )
        else:
            t_loc, dRAAN_sc, _ = integrate_raan_constant_leg(
                a_s, T_leg, e, i_rad, params, n
            )

        # SC 누적
        sc_seg = raan_sc_acc + dRAAN_sc
        # D2 누적 (이 구간 시작 RAAN + Ω̇_D2 · t_loc)
        d2_seg = raan_d2_acc + od_d2 * t_loc

        # 첫 점 중복 처리: 이전 구간 마지막 점과 같으므로 두 번째 점부터 append
        if len(t_global) == 0:
            t_global.append(t_loc + t_offset)
            raan_sc.append(sc_seg)
            raan_d2.append(d2_seg)
        else:
            t_global.append((t_loc + t_offset)[1:])
            raan_sc.append(sc_seg[1:])
            raan_d2.append(d2_seg[1:])

        # 구간 끝 누적값 갱신
        raan_sc_acc = sc_seg[-1]
        raan_d2_acc = d2_seg[-1]
        seg_bounds.append((t_offset, t_offset + T_leg, label))
        t_offset   += T_leg

    t       = np.concatenate(t_global)
    raan_sc = np.concatenate(raan_sc)
    raan_d2 = np.concatenate(raan_d2)

    info = {
        'h_P_km'        : ph['h_P_km'],
        'delta_RAAN_rad': ph['delta_RAAN'],
        'T1'  : T1,  'T2a' : T2a, 'Tp'  : Tp,
        'T2b' : T2b, 'Ts'  : Ts,
        'TOF' : T1 + T2a + Tp + T2b + Ts,
    }
    return t, raan_sc, raan_d2, seg_bounds, info


# ──────────────────────────────────────────────────────────────
# RAAN 일치 시점 탐색
# ──────────────────────────────────────────────────────────────
def find_meeting_time(t, raan_sc, raan_d2, tol_deg):
    """
    Δ(t) = Ω_SC(t) - Ω_D2(t)   (deg, unwrap)
    Δ(t) mod 360°   ∈   [-tol, +tol]   (= 0° 또는 ±360°·k 에 tol 이내)

    이 조건을 처음 만족하는 t* 를 찾는다.
    샘플 격자에서 부호 변경(또는 톨러런스 진입) 인덱스를 잡고,
    인접 두 점 사이를 선형 보간(이분법 1회) 으로 정밀화한다.

    Returns
    -------
    t_star    : float [s]   (없으면 None)
    delta_deg : float [deg]  t_star 에서 (Ω_SC - Ω_D2) mod 360, [-180,180] 로 정규화
    idx_hit   : int          맞은 샘플 인덱스 (없으면 -1)
    """
    delta_deg = np.rad2deg(raan_sc - raan_d2)

    # mod 360 으로 [-180, +180] 범위에 정규화
    delta_mod = ((delta_deg + 180.0) % 360.0) - 180.0   # [-180, 180]

    # |delta_mod| ≤ tol 인 첫 인덱스
    inside = np.where(np.abs(delta_mod) <= tol_deg)[0]
    if len(inside) == 0:
        return None, None, -1

    idx_hit = inside[0]

    if idx_hit == 0:
        # 시작부터 톨러런스 안 (드문 경우)
        return float(t[0]), float(delta_mod[0]), 0

    # 인접 샘플 사이 선형 보간 → t_star
    t0, t1 = t[idx_hit - 1], t[idx_hit]
    d0, d1 = delta_mod[idx_hit - 1], delta_mod[idx_hit]

    # mod 경계에서 점프(예: +179 → -179) 가 일어나면 보간이 의미 없음
    # → 그 경우 idx_hit 자체를 그냥 사용
    if abs(d1 - d0) > 180.0:
        return float(t[idx_hit]), float(d1), idx_hit

    # 부호가 바뀌었거나 한쪽이 이미 톨러런스 안일 때 0 으로 보간
    if d0 * d1 < 0:
        # 0 통과
        frac = abs(d0) / (abs(d0) + abs(d1))
        t_star = t0 + frac * (t1 - t0)
        return float(t_star), 0.0, idx_hit
    else:
        # 같은 부호인데 |d1| ≤ tol → 그냥 t1 사용
        return float(t[idx_hit]), float(d1), idx_hit


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    params, sim_cfg = load_params(
        constants_path = 'configs/constants.yaml',
        simulation_path = 'configs/simulation.yaml',
    )
    debris_all = load_debris_list(sim_cfg['training_data']['debris_yaml'])

    pair    = sim_cfg['visualization']['debris_pair']
    d1_name = pair[0]
    d2_name = pair[1]
    alpha   = sim_cfg['training_data']['alpha']

    d1 = debris_all[d1_name]
    d2 = debris_all[d2_name]
    d1['name'] = d1_name
    d2['name'] = d2_name

    print(f"\n{'='*60}")
    print(f"RAAN drift 적분 :  {d1_name} → phasing(h_P={H_P_KM:.0f} km) → {d2_name}")
    print(f"  D1 RAAN (= SC 시작) : {d1['RAAN']:.2f}°,  alt = {d1['alt0_km']:.1f} km")
    print(f"  D2 RAAN             : {d2['RAAN']:.2f}°,  alt = {d2['alt0_km']:.1f} km")
    print(f"  α                   : {alpha}")
    print(f"  tol                 : ±{TOL_DEG}°")
    print(f"{'='*60}")

    m_SC = params['m0']

    # ── 1) 미션 정상 구간 (T1+T2a+Tp+T2b+Ts) 동안 적분 ──
    t_mis, raan_sc_mis, raan_d2_mis, seg_bounds, info = build_raan_timeseries(
        d1, d2, params, H_P_KM, m_SC, alpha
    )

    print(f"\n미션 구간 시간 :")
    print(f"  T1  = {info['T1']/86400:8.2f} day")
    print(f"  T2a = {info['T2a']/86400:8.2f} day")
    print(f"  Tp  = {info['Tp']/86400:8.2f} day")
    print(f"  T2b = {info['T2b']/86400:8.2f} day")
    print(f"  Ts  = {info['Ts']/86400:8.2f} day")
    print(f"  TOF = {info['TOF']/86400:8.2f} day")

    # ── 2) 미션 끝나도 ±tol 내 못 만나면 D2 stay 고도(=D2 궤도)에서 더 적분 ──
    #     실제 미션 시나리오는 아니지만,
    #     "두 RAAN 이 처음 일치하는 시점" 을 보려는 분석 목적이므로 연장 추적.
    Re      = params['Re']
    i_rad   = np.deg2rad(params['inclination_deg'])
    a_D2    = Re + d2['alt0_km'] * 1e3
    a_P_m   = Re + H_P_KM * 1e3
    od_sc_extend = raan_drift_rate(a_D2, 0.0, i_rad, params)   # 미션 끝 = D2 궤도
    od_d2        = raan_drift_rate(a_D2, 0.0, i_rad, params)   # D2 도 같음 → 미션 후엔 0

    # 미션 끝까지에서 이미 만났는지 먼저 확인
    t_star, delta_at, idx_hit = find_meeting_time(
        t_mis, raan_sc_mis, raan_d2_mis, TOL_DEG
    )

    if t_star is not None:
        # 미션 안에서 만남 → 그 지점까지만 그림
        t_plot       = t_mis [: idx_hit + 1]
        raan_sc_plot = raan_sc_mis[: idx_hit + 1]
        raan_d2_plot = raan_d2_mis[: idx_hit + 1]
        within_mission = True
    else:
        # 미션이 끝났는데도 못 만남 → 안내
        # (미션 종료 후엔 SC, D2 모두 a_D2 궤도라 Ω̇ 차이가 0 → 영원히 못 만남)
        within_mission = False
        t_plot       = t_mis
        raan_sc_plot = raan_sc_mis
        raan_d2_plot = raan_d2_mis
        print(f"\n[알림] 미션 전체 구간({info['TOF']/86400:.1f} day) 안에서 "
              f"|Δ| ≤ {TOL_DEG}° 도달 못함.")
        print(f"       미션 종료 후 SC 와 D2 모두 같은 궤도(a_D2)라 "
              f"RAAN 차이가 더 이상 줄지 않습니다.")
        print(f"       (이 phasing 설계 자체로는 일치 불가능 — 다른 h_P 또는 "
              f"다른 phasing 시간이 필요)")

    # ── 결과 출력 ──
    print(f"\n결과 :")
    if t_star is not None:
        print(f"  RAAN 일치 시점 t*    = {t_star/86400:.3f} day  "
              f"({t_star:.1f} s)")
        print(f"  그 시각 Δ(mod 360)   = {delta_at:+.3f}°  (|Δ| ≤ {TOL_DEG}°)")
        # 누적 차이 (몇 바퀴 돌았는지) 확인용
        delta_cum_deg = np.rad2deg(raan_sc_plot[-1] - raan_d2_plot[-1])
        print(f"  누적 차이 (unwrap)   = {delta_cum_deg:+.2f}°  "
              f"→ 약 {delta_cum_deg/360:+.2f} 바퀴")
        print(f"  Ω_SC(t*) = {np.rad2deg(raan_sc_plot[-1]) % 360:.2f}°")
        print(f"  Ω_D2(t*) = {np.rad2deg(raan_d2_plot[-1]) % 360:.2f}°")
    else:
        print(f"  RAAN 일치 시점 : (찾지 못함)")

    # ── 그래프 ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # 구간 배경색
    seg_colors = {
        'T1' : '#d4e8ff', 'T2a': '#c8f0c8', 'Tp' : '#fff3cc',
        'T2b': '#c8f0c8', 'Ts' : '#f0d4f0',
    }

    def shade_segments(ax, t_max):
        for (t_s, t_e, lbl) in seg_bounds:
            if t_s > t_max:
                break
            t_e_clip = min(t_e, t_max)
            if t_e_clip <= t_s:
                continue
            ax.axvspan(t_s/86400, t_e_clip/86400,
                       color=seg_colors.get(lbl, '#eeeeee'),
                       alpha=0.35, linewidth=0)
            # 라벨
            ax.text((t_s + t_e_clip)/2/86400,
                    ax.get_ylim()[1] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                    lbl, ha='center', fontsize=8, color='#333')

    # ── 패널 1 : 두 RAAN (mod 360 wrap, 끊어 그리기) ──
    ax = axes[0]
    raan_sc_wrap = np.rad2deg(raan_sc_plot) % 360.0
    raan_d2_wrap = np.rad2deg(raan_d2_plot) % 360.0

    def plot_with_breaks(ax, t_arr, y_arr, **kwargs):
        # 인접점 차이가 180° 이상이면 NaN 끼워서 선 끊기
        diff = np.abs(np.diff(y_arr))
        breaks = np.where(diff > 180.0)[0]
        y_b = y_arr.astype(float).copy()
        t_b = t_arr.astype(float).copy()
        # 뒤에서부터 삽입
        for j in reversed(breaks):
            y_b = np.insert(y_b, j+1, np.nan)
            t_b = np.insert(t_b, j+1, t_arr[j])
        ax.plot(t_b/86400, y_b, **kwargs)

    plot_with_breaks(ax, t_plot, raan_sc_wrap,
                     color='tab:blue', lw=2, label='Ω_SC (chaser)')
    plot_with_breaks(ax, t_plot, raan_d2_wrap,
                     color='tab:red', lw=1.6, ls='--', label='Ω_D2')

    if t_star is not None:
        ax.axvline(t_star/86400, color='k', lw=1.0, ls=':',
                   label=f't* = {t_star/86400:.2f} day')
        ax.scatter([t_star/86400], [np.rad2deg(raan_sc_plot[-1]) % 360],
                   color='tab:blue', zorder=5, s=40)
        ax.scatter([t_star/86400], [np.rad2deg(raan_d2_plot[-1]) % 360],
                   color='tab:red',  zorder=5, s=40)

    ax.set_ylabel('RAAN [deg]  (mod 360)')
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"RAAN drift integration (식 22~24) :  "
        f"{d1_name} → {d2_name}, h_P = {H_P_KM:.0f} km"
    )
    shade_segments(ax, t_plot[-1])

    # ── 패널 2 : Δ(t) = Ω_SC - Ω_D2 (mod 360, [-180,180]) ──
    ax = axes[1]
    delta_mod = ((np.rad2deg(raan_sc_plot - raan_d2_plot) + 180.0) % 360.0) - 180.0
    plot_with_breaks(ax, t_plot, delta_mod,
                     color='tab:purple', lw=1.6, label='Δ (mod 360)')

    ax.axhline( TOL_DEG, color='gray', ls=':', lw=1.0,
               label=f'±{TOL_DEG}° tol')
    ax.axhline(-TOL_DEG, color='gray', ls=':', lw=1.0)
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.5)
    if t_star is not None:
        ax.axvline(t_star/86400, color='k', lw=1.0, ls=':')
        ax.scatter([t_star/86400], [delta_at], color='tab:purple',
                   zorder=5, s=50)

    ax.set_ylabel('Δ = Ω_SC − Ω_D2 [deg]\n(mod 360, [-180,180])')
    ax.set_xlabel('time [day]')
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, -TOL_DEG, 0, TOL_DEG, 90, 180])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    shade_segments(ax, t_plot[-1])

    plt.tight_layout()

    # 저장
    out_dir = sim_cfg.get('visualization', {}).get('output_dir', 'results/figures')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"raan_drift_{d1_name}_{d2_name}_hP{int(H_P_KM)}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    print(f"\n  그래프 저장 : {out_path}")
    plt.show()


if __name__ == '__main__':
    main()