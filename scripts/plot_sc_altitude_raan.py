"""
scripts/plot_sc_altitude_raan.py
================================
Sequence A 의 spacecraft (SC) 시나리오에서 시간에 따른
  - SC 고도 [km]
  - SC RAAN [deg]   (0~360°, wrap)
을 동시에 그래프로 그린다 (첨부 사진과 동일한 dual-axis 형식,
빨간색 = RAAN, 파란색 = altitude).

시나리오 (debris0001 ~ debris0013, 약 3800일):
  사용자 명시 (d1~d4) :
    t=0       : d1(637.16km) 출발 → 하강
    t=72.95   : 390km 도달
    t=309.61  : 390km 떠나 d2 향해 상승
    t=334.52  : d2 도달
    t=364.52  : d2 떠남 (30일 머묾)
    t=405.69  : 390km
    t=500.00  : 390km 떠나 d3
    t=540.92  : d3 도달
    t=570.92  : d3 떠남
    t=622.78  : 390km (즉시 상승)
    t=685.05  : phasing orbit(708.14km) 도달
    t=725.98  : phasing 떠남
    t=759.79  : d4 도달
    t=789.79  : d4 떠남
    t=829.18  : 390km

  이미지에서 추출 (d5~d13, ±5일 정확도) :
    각 debris 머묾 30일 (이미지 peak center ± 15일).
    390km coast 구간은 이미지 평탄 검출 결과 사용.
    d12 → d13 사이 평탄은 검출되지 않아 alt 변화량 비례로 시각 추정.

사용법:
  python scripts/plot_sc_altitude_raan.py
  python scripts/plot_sc_altitude_raan.py --n_samples 2000
  python scripts/plot_sc_altitude_raan.py --debris_yaml configs/debris_sequenceA.yaml
  # 특정 debris 의 RAAN 을 함께 plot (0일 ~ SC 가 그 debris 에 도달한 시각까지) :
  python scripts/plot_sc_altitude_raan.py --debris 3
  python scripts/plot_sc_altitude_raan.py --debris 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 프로젝트 루트 import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import (
    raan_drift_rate,
    integrate_raan_thrust_leg,
)


# ────────────────────────────────────────────────
# 시나리오 정의
# ────────────────────────────────────────────────

def build_scenario(debris_dict):
    """
    사용자가 명시한 d1~d4 timeline + 이미지에서 추출한 d5~d13 timeline 으로
    phase 리스트를 구성한다. 각 phase 는 dict 형식:
      {
        't0_day' : 시작 시각 [day]
        't1_day' : 종료 시각 [day]
        'h0_km'  : 시작 고도 [km]
        'h1_km'  : 종료 고도 [km]   (coast 면 h0 와 동일)
        'kind'   : 'thrust' (고도 변화) | 'coast' (고도 일정)
      }

    debris 고도는 YAML 에서 읽음. 시각 일관성 :
      - d1~d4 : 사용자 제공값 사용 (정확함)
      - d5~d13 : 이미지 peak center 기준 ± 15일 (30일 머묾 가정)
      - 390km coast 구간 : 이미지에서 평탄 구간 픽셀 추적으로 추출
      - d12 → d13 사이 평탄은 검출 안 됨 → alt 변화량 비례로 추정
    """
    da = {k: debris_dict[f'debris{k:04d}']['alt0_km'] for k in range(1, 14)}
    h_disp    = 390.0
    h_phasing = 708.14   # 사용자 명시 phasing orbit (d3 → d4 사이)

    # debris 머묾 중심 시각 (이미지에서 추출, peak center)
    peak_centers = {
        5:  1176.65, 6: 1508.40, 7: 1962.40, 8: 2222.15,
        9:  2667.40, 10: 2870.35, 11: 3267.60, 12: 3625.60, 13: 3752.20,
    }
    # 390km coast 구간 (이미지 평탄 검출). 각 원소 = (start, end, 다음에 가는 debris index)
    flats_after = [
        (829.18,  1106.80, 5),
        (1272.70, 1425.50, 6),
        (1626.30, 1910.00, 7),
        (1997.30, 2171.90, 8),
        (2289.80, 2625.90, 9),
        (2717.60, 2787.40, 10),
        (2970.80, 3219.60, 11),
        (3350.60, 3573.20, 12),
    ]

    phases = []

    # === d1 ~ d4 (사용자 제공값) ===
    phases += [
        {'t0_day': 0.00,   't1_day':  72.95, 'h0_km': da[1],     'h1_km': h_disp,    'kind': 'thrust'},
        {'t0_day':  72.95, 't1_day': 309.61, 'h0_km': h_disp,    'h1_km': h_disp,    'kind': 'coast'},
        {'t0_day': 309.61, 't1_day': 334.52, 'h0_km': h_disp,    'h1_km': da[2],     'kind': 'thrust'},
        {'t0_day': 334.52, 't1_day': 364.52, 'h0_km': da[2],     'h1_km': da[2],     'kind': 'coast'},
        {'t0_day': 364.52, 't1_day': 405.69, 'h0_km': da[2],     'h1_km': h_disp,    'kind': 'thrust'},
        {'t0_day': 405.69, 't1_day': 500.00, 'h0_km': h_disp,    'h1_km': h_disp,    'kind': 'coast'},
        {'t0_day': 500.00, 't1_day': 540.92, 'h0_km': h_disp,    'h1_km': da[3],     'kind': 'thrust'},
        {'t0_day': 540.92, 't1_day': 570.92, 'h0_km': da[3],     'h1_km': da[3],     'kind': 'coast'},
        {'t0_day': 570.92, 't1_day': 622.78, 'h0_km': da[3],     'h1_km': h_disp,    'kind': 'thrust'},
        # phasing orbit (d3 → phasing → d4)
        {'t0_day': 622.78, 't1_day': 685.05, 'h0_km': h_disp,    'h1_km': h_phasing, 'kind': 'thrust'},
        {'t0_day': 685.05, 't1_day': 725.98, 'h0_km': h_phasing, 'h1_km': h_phasing, 'kind': 'coast'},
        {'t0_day': 725.98, 't1_day': 759.79, 'h0_km': h_phasing, 'h1_km': da[4],     'kind': 'thrust'},
        {'t0_day': 759.79, 't1_day': 789.79, 'h0_km': da[4],     'h1_km': da[4],     'kind': 'coast'},
        {'t0_day': 789.79, 't1_day': 829.18, 'h0_km': da[4],     'h1_km': h_disp,    'kind': 'thrust'},
    ]

    # === d5 ~ d12 (이미지 추출, 각 cycle: coast390 + 상승 + 머묾 + 하강) ===
    for k, (coast_s, coast_e, d_idx) in enumerate(flats_after):
        pc = peak_centers[d_idx]
        arr, dep = pc - 15.0, pc + 15.0
        h_d = da[d_idx]
        # 390km coast
        phases.append({'t0_day': coast_s, 't1_day': coast_e,
                       'h0_km': h_disp, 'h1_km': h_disp, 'kind': 'coast'})
        # 상승
        phases.append({'t0_day': coast_e, 't1_day': arr,
                       'h0_km': h_disp, 'h1_km': h_d, 'kind': 'thrust'})
        # debris 머묾 (30일)
        phases.append({'t0_day': arr, 't1_day': dep,
                       'h0_km': h_d, 'h1_km': h_d, 'kind': 'coast'})
        # 하강 → 다음 390 coast 까지 (마지막 cycle 은 d12 → d13 처리로 별도)
        if k + 1 < len(flats_after):
            next_coast_s = flats_after[k + 1][0]
            phases.append({'t0_day': dep, 't1_day': next_coast_s,
                           'h0_km': h_d, 'h1_km': h_disp, 'kind': 'thrust'})

    # === d12 → d13 (평탄 검출 안 됨 → alt 비례로 추정) ===
    # d12 떠남 = 3640.6, d13 도달 = 3737.2 → 96.6일 사이.
    # 하강 + 상승 시간 비를 고도 변화량 비로 가정.
    d12_dep = peak_centers[12] + 15.0
    d13_arr = peak_centers[13] - 15.0
    gap = d13_arr - d12_dep
    desc_dur = gap * (da[12] - h_disp) / ((da[12] - h_disp) + (da[13] - h_disp))
    t_390 = d12_dep + desc_dur
    phases += [
        {'t0_day': d12_dep, 't1_day': t_390,   'h0_km': da[12],  'h1_km': h_disp,  'kind': 'thrust'},
        {'t0_day': t_390,   't1_day': d13_arr, 'h0_km': h_disp,  'h1_km': da[13],  'kind': 'thrust'},
        {'t0_day': d13_arr, 't1_day': peak_centers[13] + 15,
         'h0_km': da[13], 'h1_km': da[13], 'kind': 'coast'},
        {'t0_day': peak_centers[13] + 15, 't1_day': 3800.0,
         'h0_km': da[13], 'h1_km': h_disp, 'kind': 'thrust'},
    ]

    return phases


# ────────────────────────────────────────────────
# 시점 t 에서의 (altitude, RAAN) 계산
# ────────────────────────────────────────────────

def find_phase(t_day, phases):
    """t_day 가 속한 phase index 반환. 마지막 시점은 마지막 phase 로 처리."""
    for k, ph in enumerate(phases):
        if ph['t0_day'] <= t_day <= ph['t1_day']:
            return k
    # 시퀀스 종료 후
    return len(phases) - 1


def altitude_at(t_day, phases):
    """시점 t_day 에서의 SC 고도 [km]. thrust 구간은 시간 선형 보간."""
    k = find_phase(t_day, phases)
    ph = phases[k]
    if ph['kind'] == 'coast':
        return ph['h0_km']
    # thrust : t 비율로 선형 보간 (transfer_solver.py 의 a(t) 선형 보간 가정과 일치)
    t0, t1 = ph['t0_day'], ph['t1_day']
    if t1 - t0 <= 0:
        return ph['h1_km']
    frac = (t_day - t0) / (t1 - t0)
    return ph['h0_km'] + frac * (ph['h1_km'] - ph['h0_km'])


def compute_phase_endpoint_raan(phases, params, i_rad, e):
    """
    각 phase 종료 시점에서의 누적 RAAN (rad) 을 미리 계산.

    이렇게 두면 시점별 RAAN 계산 시 :
      해당 phase 시작 시점의 누적 RAAN  +  phase 내부에서 [t0, t] 적분
    한 번으로 끝낼 수 있어 효율적.

    Returns
    -------
    raan_at_phase_start : list[float] [rad]
        len = n_phases (각 phase 의 시작 시점 누적 RAAN; index 0 은 SC 초기값)
    """
    Re = params['Re']
    raan_starts = [0.0]   # phase 0 시작 = SC 초기 RAAN, 외부에서 더해 줌

    for ph in phases:
        a0 = Re + ph['h0_km'] * 1e3
        a1 = Re + ph['h1_km'] * 1e3
        T_leg = (ph['t1_day'] - ph['t0_day']) * 86400.0

        if ph['kind'] == 'thrust':
            d_raan = integrate_raan_thrust_leg(a0, a1, T_leg, e, i_rad, params)
        else:
            # coast : a 일정 → 해석식
            d_raan = raan_drift_rate(a0, e, i_rad, params) * T_leg

        raan_starts.append(raan_starts[-1] + d_raan)

    # raan_starts 는 (phase 시작 시점 누적값) 만 의미를 가지므로
    # 마지막 원소는 시나리오 전체 끝 시점의 누적값
    return raan_starts


def raan_at(t_day, phases, raan_starts, params, i_rad, e, raan_init_rad):
    """
    시점 t_day 에서의 SC RAAN [rad] (unwrap 누적).

    raan_starts[k] = phase k 시작 시점의 누적 ΔΩ (raan_init 제외)
    """
    k = find_phase(t_day, phases)
    ph = phases[k]
    Re = params['Re']

    # phase 시작까지의 누적
    raan_acc = raan_starts[k]

    # phase 내부 [t0, t] 추가 적분
    t0 = ph['t0_day']
    delta_t_sec = (t_day - t0) * 86400.0
    if delta_t_sec > 0:
        a0 = Re + ph['h0_km'] * 1e3
        a1 = Re + ph['h1_km'] * 1e3
        T_leg_total = (ph['t1_day'] - t0) * 86400.0

        if ph['kind'] == 'thrust':
            # [t0, t] 구간 끝 고도는 a(t) 선형 보간으로
            if T_leg_total > 0:
                frac = delta_t_sec / T_leg_total
                a_t = a0 + frac * (a1 - a0)
            else:
                a_t = a1
            # 사용자 요구사항: "구간 내부 매 점마다 적분으로 구해라"
            # → 각 시점마다 [t0, t] 를 새로 적분
            d_raan = integrate_raan_thrust_leg(
                a0, a_t, delta_t_sec, e, i_rad, params
            )
        else:
            d_raan = raan_drift_rate(a0, e, i_rad, params) * delta_t_sec

        raan_acc += d_raan

    return raan_init_rad + raan_acc


def sc_arrival_time_at_debris(debris_idx, phases, debris_dict):
    """
    SC 가 debris{debris_idx} 에 도달한 시각 [day] 을 반환.

    판별 기준: phase 가 'thrust' 이고 t1 에서의 고도(=h1_km) 가
    해당 debris 의 alt0_km 와 일치하는 첫 phase 의 t1.
    debris_idx == 1 이면 출발 시점이므로 0.0 을 반환.

    Returns
    -------
    t_day : float
    """
    if debris_idx == 1:
        return 0.0
    target_alt = debris_dict[f'debris{debris_idx:04d}']['alt0_km']
    for ph in phases:
        if ph['kind'] == 'thrust' and abs(ph['h1_km'] - target_alt) < 0.5:
            return ph['t1_day']
    raise ValueError(
        f"debris{debris_idx:04d} (alt={target_alt}km) 도달 phase 를 "
        f"찾을 수 없습니다."
    )


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequence A SC 의 고도 / RAAN 시간변화 플롯"
    )
    parser.add_argument(
        '--debris', type=int, default=None, choices=list(range(1, 14)),
        help="함께 그릴 debris 번호 1~13. SC 가 해당 debris 에 도달한 시각까지 "
             "초록색으로 debris RAAN 을 plot 한다. 생략하면 SC 만 그림."
    )
    parser.add_argument(
        '--debris_yaml', type=str,
        default='configs/debris_sequenceA.yaml',
        help="debris YAML 경로 (기본: configs/debris_sequenceA.yaml)"
    )
    parser.add_argument(
        '--constants', type=str,
        default='configs/constants.yaml',
        help="물리 상수 YAML 경로"
    )
    parser.add_argument(
        '--simulation', type=str,
        default='configs/simulation.yaml',
        help="시뮬레이션 설정 YAML 경로"
    )
    parser.add_argument(
        '--n_samples', type=int, default=2000,
        help="시간축 샘플 수 (기본 2000). 클수록 그래프 매끄러움."
    )
    args = parser.parse_args()

    # 로드
    params, _ = load_params(args.constants, args.simulation)
    debris = load_debris_list(args.debris_yaml)

    i_rad = np.deg2rad(params['inclination_deg'])
    # 모든 debris 가 같은 inclination 이라고 가정 (constants.yaml 의 값 사용)
    # debris e 도 동일하게 처리. 일반적으로 SC 와 debris 의 e 는 0 으로 가정.
    e_SC = debris['debris0001']['e']

    # SC 초기 RAAN = debris0001 RAAN
    raan_init_deg = debris['debris0001']['RAAN']
    raan_init_rad = np.deg2rad(raan_init_deg)

    # 시나리오 구성
    phases = build_scenario(debris)

    print(f"[info] inclination     : {params['inclination_deg']:.4f} deg")
    print(f"[info] e (SC)          : {e_SC}")
    print(f"[info] SC 초기 RAAN    : {raan_init_deg:.4f} deg "
          f"(debris0001 기준)")
    print(f"[info] n_phases        : {len(phases)}")
    print(f"[info] t_max           : {phases[-1]['t1_day']:.2f} day")
    print()

    # phase 시작 시점 누적 RAAN 사전 계산
    raan_starts = compute_phase_endpoint_raan(phases, params, i_rad, e_SC)

    # 시간축 샘플링
    t_max_day = phases[-1]['t1_day']
    t_days = np.linspace(0.0, t_max_day, args.n_samples)

    altitudes_km = np.empty_like(t_days)
    raan_deg     = np.empty_like(t_days)

    # 시점별 계산 (tqdm 진행상황 표시)
    for k, t in enumerate(tqdm(t_days, desc="SC 시점별 계산", unit="pt")):
        altitudes_km[k] = altitude_at(t, phases)
        raan_rad = raan_at(t, phases, raan_starts,
                           params, i_rad, e_SC, raan_init_rad)
        # [0, 360) wrap
        raan_deg[k] = np.rad2deg(raan_rad) % 360.0

    # ── 선택된 debris 의 RAAN drift 계산 (옵션) ──
    # --debris 가 None 또는 1 이면 debris RAAN 그래프를 그리지 않는다.
    #   - None : 인자 미지정
    #   - 1    : 출발 debris (도착 시각 = 0 → 그릴 구간 없음)
    plot_debris = args.debris is not None and args.debris != 1
    t_d_days = raan_d_deg = None
    if plot_debris:
        d_name = f'debris{args.debris:04d}'
        d_info = debris[d_name]
        t_arrive = sc_arrival_time_at_debris(args.debris, phases, debris)
        print(f"[info] 선택 debris   : {d_name} (alt={d_info['alt0_km']} km, "
              f"RAAN0={d_info['RAAN']:.4f}°)")
        print(f"[info] SC 도착 시각  : {t_arrive:.2f} day")

        # debris 는 추력 X → Ω̇ 가 시간에 무관한 상수 → 해석식
        a_d   = params['Re'] + d_info['alt0_km'] * 1e3
        e_d   = d_info['e']
        i_d   = np.deg2rad(d_info['i'])
        omega_dot_d = raan_drift_rate(a_d, e_d, i_d, params)   # [rad/s]
        raan_d_init = np.deg2rad(d_info['RAAN'])

        # 0 ~ 도착시각 구간만 샘플링
        t_d_days = np.linspace(0.0, t_arrive, args.n_samples)
        raan_d_rad = raan_d_init + omega_dot_d * (t_d_days * 86400.0)
        raan_d_deg = (np.rad2deg(raan_d_rad)) % 360.0

    # ── 플롯 ──
    print("[info] 플롯 생성 중...")
    fig, ax1 = plt.subplots(figsize=(11, 5))

    # 왼쪽 축: altitude (파랑)
    color_alt = 'tab:blue'
    ax1.plot(t_days, altitudes_km, color=color_alt, lw=1.6)
    ax1.set_xlabel('Time, day', fontsize=12)
    ax1.set_ylabel('Altitude, km', color=color_alt, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_alt)
    ax1.set_xlim(0, t_max_day)
    ax1.grid(alpha=0.3)

    # 오른쪽 축: RAAN
    color_raan = 'tab:red'
    color_dbr  = 'tab:green'
    ax2 = ax1.twinx()
    # wrap 에 의한 360↔0 점프를 그래프에서 직선으로 잇지 않도록 NaN 삽입
    raan_plot_masked = raan_deg.astype(float).copy()
    jump_idx = np.where(np.abs(np.diff(raan_plot_masked)) > 180.0)[0]
    for j in jump_idx:
        raan_plot_masked[j] = np.nan
    sc_label = r'$\Omega_{SC}$' if plot_debris else None
    ax2.plot(t_days, raan_plot_masked, color=color_raan, lw=1.6,
             label=sc_label)

    if plot_debris:
        # debris RAAN 도 wrap 점프 처리
        raan_d_plot = raan_d_deg.astype(float).copy()
        jumps_d = np.where(np.abs(np.diff(raan_d_plot)) > 180.0)[0]
        for j in jumps_d:
            raan_d_plot[j] = np.nan
        ax2.plot(t_d_days, raan_d_plot, color=color_dbr, lw=1.6,
                 label=rf'$\Omega_{{debris{args.debris:04d}}}$')
        # debris RAAN 만 표시할 때는 ylabel 을 단순히 'Ω, deg' 로
        ax2.set_ylabel(r'$\Omega$, deg', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='upper right', fontsize=10)
    else:
        ax2.set_ylabel(r'$\Omega_{SC}$, deg', color=color_raan, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_raan)

    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 60, 120, 180, 240, 300, 360])

    plt.title('Sequence A', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()