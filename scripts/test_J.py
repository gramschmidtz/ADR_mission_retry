"""
test_J.py
=========
phasing orbit 고도 h_P 를 390 km 부터 1500 km 까지 100 km 단위로 변화시키며
목적함수 J 값과 그 구성 성분(ΔV_PT, T_PT)을 직접 계산하여 표와 그래프로 출력한다.

논문 식 (28):
    J = α · ΔV_PT + (1-α) · T_PT
        ΔV_PT = ΔV_{T2,a} + ΔV_{T2,b} + ΔV_drag         (식 26)
        T_PT  = T_{T2,a} + T_p + T_{T2,b}                (식 27)

α 는 configs/simulation.yaml 의 training_data.alpha 를 사용한다.

transfer_solver.optimize_phasing_orbit 도 동일한 grid search 로 동작하지만,
이 스크립트는 evaluate_phasing_orbit 함수를 dense grid 위에서 호출하여
J(h_P) 곡선의 형태(특히 h_P = h_D2 의 극과 양 가지)를 직접 시각화하는
것이 목적이다.

사용 debris : configs/debris.yaml 의 debris0001 → debris0002

실행 방법:
    python scripts/test_J.py
"""

import sys
import os
import numpy as np

# 프로젝트 루트를 path 에 추가 (다른 scripts 와 동일 패턴)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_solver import (
    evaluate_phasing_orbit,
    tof_tsiolkovsky,
    propellant_consumed_eq10,
    delta_v_from_mass,
)


# ──────────────────────────────────────────────────────────────
# 사용자 설정
# ──────────────────────────────────────────────────────────────
DEBRIS_PAIR = ("debris0001", "debris0002")
# h_P 탐색 범위와 step 은 모두 simulation.yaml 의 search 섹션을 사용
#   search.h_P_min_km, search.h_P_max_km, search.h_step_km


def compute_m_SC_after_T1(debris1, m_SC_start, params):
    """
    solve_transfer 의 T1 단계를 그대로 재현하여
    D1 방출 후 chaser 단독 질량을 계산한다 (Tsiolkovsky 해석해 + 식 10).

    T1: D1 orbit → disposal orbit (chaser+D1 합산 질량으로 연소)
    """
    h_D1   = debris1["alt0_km"]
    m_D1   = debris1["mass"]
    h_disp = params["disposal_alt_km"]

    m_total_T1       = m_SC_start + m_D1
    tof_T1           = tof_tsiolkovsky(h_D1, h_disp, m_total_T1, params)
    m_prop_T1        = propellant_consumed_eq10(tof_T1, params)
    m_total_after_T1 = m_total_T1 - m_prop_T1
    dv_T1            = delta_v_from_mass(m_total_T1, m_prop_T1, params)
    m_SC_after_T1    = m_SC_start - m_prop_T1
    return m_SC_after_T1, dv_T1, tof_T1


def main():
    # ── 설정 로드 ──
    params, sim_cfg = load_params(
        constants_path="configs/constants.yaml",
        simulation_path="configs/simulation.yaml",
    )
    alpha = sim_cfg["training_data"]["alpha"]
    debris_yaml = sim_cfg["visualization"]["debris_file"]
    debris_list = load_debris_list(debris_yaml)

    name1, name2 = DEBRIS_PAIR
    debris1 = debris_list[name1]
    debris2 = debris_list[name2]

    h_D1, RAAN_D1_deg, m_D1 = debris1["alt0_km"], debris1["RAAN"], debris1["mass"]
    h_D2, RAAN_D2_deg = debris2["alt0_km"], debris2["RAAN"]
    h_disp = params["disposal_alt_km"]
    i_rad = np.deg2rad(params["inclination_deg"])
    m_SC_start = params["m0"]

    # ── T1 단계 진행하여 m_SC_after_T1 계산 ──
    m_SC_after_T1, dv_T1, tof_T1 = compute_m_SC_after_T1(
        debris1, m_SC_start, params
    )

    # ── 헤더 출력 ──
    print("=" * 88)
    print(f"  J(h_P) 스윕 : {name1} → {name2}")
    print("=" * 88)
    print(f"  α (simulation.yaml) : {alpha}")
    print(f"  D1  : h={h_D1:7.2f} km, RAAN={RAAN_D1_deg:7.2f} deg, m={m_D1:6.2f} kg")
    print(f"  D2  : h={h_D2:7.2f} km, RAAN={RAAN_D2_deg:7.2f} deg")
    print(f"  disposal alt        : {h_disp:.2f} km")
    print(f"  m_SC start          : {m_SC_start:.2f} kg")
    print(f"  m_SC after T1       : {m_SC_after_T1:.2f} kg")
    print(f"  T1 ΔV               : {dv_T1:.2f} m/s")
    print(f"  T1 TOF              : {tof_T1/86400:.2f} days")
    print("-" * 88)

    # ── grid 스윕 (탐색 범위와 step 은 모두 simulation.yaml 의 search 섹션) ──
    h_P_min  = max(h_disp, params['h_P_min_km'])
    h_P_max  = params['h_P_max_km']
    h_step   = params['h_step_km']
    h_P_grid = np.arange(h_P_min, h_P_max + h_step / 2, h_step)
    print(f"  h_P 탐색 범위        : [{h_P_min:.1f}, {h_P_max:.1f}] km, "
          f"step {h_step} km ({len(h_P_grid)} 점)")
    print("-" * 88)

    rows = []
    print(f"{'h_P[km]':>9} | {'J':>10} | {'ΔV_PT[m/s]':>11} | "
          f"{'T_PT[day]':>10} | {'T2a[day]':>9} | {'Tp[day]':>9} | "
          f"{'T2b[day]':>9} | {'ΔV_drag':>8}")
    print("-" * 88)

    for h_P in h_P_grid:
        J, res = evaluate_phasing_orbit(
            h_P_km     = float(h_P),
            h_D1_km    = h_D1,
            RAAN_D1_0  = np.deg2rad(RAAN_D1_deg),
            m_D1       = m_D1,
            h_disp_km  = h_disp,
            h_D2_km    = h_D2,
            RAAN_D2_0  = np.deg2rad(RAAN_D2_deg),
            m_SC_start = m_SC_start,
            params     = params,
            alpha      = alpha,
            i_rad      = i_rad,
        )

        if not np.isfinite(J) or not res:
            print(f"{h_P:9.1f} | {'inf':>10} | (phasing 불가, ΔΩ 부호로 인해 skip)")
            rows.append({
                "h_P_km": h_P, "J": np.inf,
                "dv_PT": np.nan, "tof_PT_day": np.nan,
                "T2a_day": np.nan, "Tp_day": np.nan,
                "T2b_day": np.nan, "dv_drag": np.nan,
            })
            continue

        tof_PT_day = res["tof_PT"] / 86400.0
        T2a_day    = res["tof_T2a"] / 86400.0
        Tp_day     = res["Tp"] / 86400.0
        T2b_day    = res["tof_T2b"] / 86400.0

        print(f"{h_P:9.1f} | {J:10.4f} | {res['dv_PT']:11.2f} | "
              f"{tof_PT_day:10.2f} | {T2a_day:9.2f} | {Tp_day:9.2f} | "
              f"{T2b_day:9.2f} | {res['dv_drag_P']:8.2f}")

        rows.append({
            "h_P_km" : h_P,
            "J"      : J,
            "dv_PT"  : res["dv_PT"],
            "tof_PT_day": tof_PT_day,
            "T2a_day": T2a_day,
            "Tp_day" : Tp_day,
            "T2b_day": T2b_day,
            "dv_drag": res["dv_drag_P"],
        })

    print("-" * 88)
    # ── 최소 J 위치 ──
    finite_rows = [r for r in rows if np.isfinite(r["J"])]
    if finite_rows:
        best = min(finite_rows, key=lambda r: r["J"])
        print(f"\n  grid 최소 J : h_P = {best['h_P_km']:.1f} km, J = {best['J']:.4f}")
        print(f"               ΔV_PT = {best['dv_PT']:.2f} m/s, "
              f"T_PT = {best['tof_PT_day']:.2f} days")
    print("=" * 88)

    # ── 그래프 ──
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[알림] matplotlib 미설치 → 그래프 생략")
        return

    h_arr     = np.array([r["h_P_km"]     for r in rows])
    J_arr     = np.array([r["J"]          for r in rows])
    dv_arr    = np.array([r["dv_PT"]      for r in rows])
    tof_arr   = np.array([r["tof_PT_day"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    ax.plot(h_arr, J_arr, "o-", color="tab:blue")
    if finite_rows:
        ax.axvline(best["h_P_km"], color="red", ls="--", lw=1,
                   label=f"min J @ {best['h_P_km']:.0f} km")
        ax.legend(loc="best")
    ax.set_ylabel(f"J  (α = {alpha})")
    ax.set_title(f"J(h_P) sweep : {name1} → {name2}  "
                 f"(h_D1={h_D1:.0f}, h_D2={h_D2:.0f} km)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(h_arr, dv_arr, "s-", color="tab:orange")
    ax.set_ylabel("ΔV_PT  [m/s]")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(h_arr, tof_arr, "^-", color="tab:green")
    ax.set_ylabel("T_PT  [days]")
    ax.set_yscale("log")
    ax.set_xlabel("phasing orbit altitude  h_P  [km]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    out_dir = sim_cfg.get("visualization", {}).get("output_dir", "results/figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"J_sweep_{name1}_to_{name2}_alpha{alpha}.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  그래프 저장 : {out_path}")
    plt.show()


if __name__ == "__main__":
    main()