"""
scripts/propagate_raan.py
=========================
주어진 일수(days) 동안 각 debris의 RAAN이 J2 섭동에 의해 자연 drift된 값을
계산하여 출력한다.

배경 :
  debris 는 추력이 없는 상태로 자유 비행하므로 반장축 a, 이심률 e, 경사각 i
  는 변하지 않는다고 가정한다 (J2 섭동의 secular 효과는 RAAN, AOP, M 에만
  작용). 따라서 RAAN 변화율 Ω̇ 가 일정하므로 :
        Ω(t) = Ω₀ + Ω̇ · t
  로 단순 선형 외삽이 가능하다.

  Ω̇ 는 src/transfer_solver.py 의 raan_drift_rate() 함수 (논문 식 22) 를
  그대로 사용한다.

사용법 :
  python scripts/propagate_raan.py --days 30
  python scripts/propagate_raan.py --days 30 --debris configs/debris_sequenceA.yaml
  python scripts/propagate_raan.py --days -15.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path 에 추가하여 src.* 임포트 가능하게 함
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import raan_drift_rate


def propagate_raan(debris_info, days, params):
    """
    한 debris 의 RAAN 을 days 일만큼 J2 섭동으로 외삽한다.

    Parameters
    ----------
    debris_info : dict
        {'alt0_km', 'e', 'i', 'RAAN', ...} — RAAN, i 는 [deg]
    days        : float
        외삽 기간 [days] (음수 = 과거)
    params      : dict
        load_params() 로 로드한 물리 파라미터

    Returns
    -------
    RAAN_new_deg : float [deg]   ([0, 360) 으로 wrapping)
    omega_dot    : float [rad/s] (참고용)
    """
    Re = params['Re']

    # debris 궤도 요소 (원궤도 가정이 아니므로 e 그대로 사용)
    a = Re + debris_info['alt0_km'] * 1e3   # 반장축 [m]
    e = debris_info['e']
    i = np.deg2rad(debris_info['i'])        # 경사각 [rad]
    RAAN0 = np.deg2rad(debris_info['RAAN']) # 초기 RAAN [rad]

    # J2 RAAN drift rate (식 22)
    omega_dot = raan_drift_rate(a, e, i, params)   # [rad/s]

    # 시간 [s]
    t_sec = days * 86400.0

    # 선형 외삽
    RAAN_new = RAAN0 + omega_dot * t_sec

    # [0, 360) deg 로 wrapping
    RAAN_new_deg = np.rad2deg(RAAN_new) % 360.0

    return RAAN_new_deg, omega_dot


def main():
    parser = argparse.ArgumentParser(
        description="J2 섭동 기반 debris RAAN 외삽 스크립트"
    )
    parser.add_argument(
        '--days', type=float, required=True,
        help="외삽 기간 [days]. 음수 가능 (과거)."
    )
    parser.add_argument(
        '--debris', type=str,
        default='configs/debris_sequenceA.yaml',
        help="debris 목록 YAML 파일 경로 "
             "(기본: configs/debris_sequenceA.yaml)"
    )
    parser.add_argument(
        '--constants', type=str,
        default='configs/constants.yaml',
        help="물리 상수 YAML 파일 경로 (기본: configs/constants.yaml)"
    )
    parser.add_argument(
        '--simulation', type=str,
        default='configs/simulation.yaml',
        help="시뮬레이션 설정 YAML 파일 경로 (기본: configs/simulation.yaml)"
    )
    args = parser.parse_args()

    # 파라미터 / debris 목록 로드
    params, _ = load_params(
        constants_path=args.constants,
        simulation_path=args.simulation,
    )
    debris_dict = load_debris_list(args.debris)

    # 헤더 출력
    print(f"# RAAN propagation by J2 secular drift")
    print(f"# elapsed time : {args.days} days  ({args.days * 86400.0:.1f} s)")
    print(f"# debris file  : {args.debris}")
    print(f"# n_debris     : {len(debris_dict)}")
    print()
    print(f"{'name':<12} {'RAAN0 [deg]':>12} {'RAAN(t) [deg]':>15} "
          f"{'dRAAN [deg]':>12} {'Omega_dot [deg/day]':>22}")
    print("-" * 78)

    # 각 debris 외삽
    for name in sorted(debris_dict.keys()):
        info = debris_dict[name]
        raan0_deg = info['RAAN']
        raan_new_deg, omega_dot = propagate_raan(info, args.days, params)

        # ΔRAAN = RAAN(t) - RAAN0 을 [-180, 180) 로 wrapping (시각 비교용)
        dRAAN_deg = ((raan_new_deg - raan0_deg) + 180.0) % 360.0 - 180.0

        # 참고용 omega_dot [deg/day]
        omega_dot_dpd = np.rad2deg(omega_dot) * 86400.0

        print(f"{name:<12} {raan0_deg:>12.4f} {raan_new_deg:>15.4f} "
              f"{dRAAN_deg:>+12.4f} {omega_dot_dpd:>+22.6f}")


if __name__ == "__main__":
    main()