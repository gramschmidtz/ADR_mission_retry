"""
check_samples.py
================
data/training_data.csv (또는 다른 경로) 를 읽어 sample 통계와 히스토그램을
화면에 띄운다. **파일은 읽기만 하고 수정하거나 저장하지 않는다.**

기능
----
  1) 텍스트 통계 (콘솔 출력):
       - 총 row 수 / success / fail 개수 + 비율
       - fail 행의 입력 분포 통계 (h, RAAN, mass 의 min/max/mean)
       - fail 행의 m_prop / TOF / h_P 가 모두 NaN 인지 검증
       - d1==d2 같은 이상 케이스 경고
  2) 히스토그램 (matplotlib 창으로 표시, success=1 행 기준):
       - m_prop_kg 분포 (linear bins)
       - TOF_days 분포 (linear bins)
       - TOF_days 분포 (log-spaced bins) — long-tail 가시화

사용 :
  python scripts/check_samples.py
  python scripts/check_samples.py --csv data/training_data.csv
  python scripts/check_samples.py --bins 60
"""

import os
import sys
import argparse
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ──────────────────────────────────────────────────────────────
# csv → list of dict (가벼움, pandas 의존성 없음)
# ──────────────────────────────────────────────────────────────
def load_csv_rows(csv_path):
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _to_float_or_nan(s):
    if s is None or s == '' or s.lower() == 'nan':
        return float('nan')
    try:
        return float(s)
    except ValueError:
        return float('nan')


# ──────────────────────────────────────────────────────────────
# 통계 도우미
# ──────────────────────────────────────────────────────────────
def _basic_stats(values):
    """nan-safe min/max/mean. 빈 리스트 → (None, None, None)."""
    import math
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return (None, None, None)
    return (min(finite), max(finite), sum(finite) / len(finite))


def _fmt(x, width=8, prec=2):
    if x is None:
        return f"{'(n/a)':>{width}}"
    return f"{x:>{width}.{prec}f}"


# ──────────────────────────────────────────────────────────────
# 히스토그램 (matplotlib show)
# ──────────────────────────────────────────────────────────────
def _show_histograms(m_prop_vals, tof_vals, n_bins):
    """m_prop / TOF (linear, log) 히스토그램을 화면에 띄운다 (저장 X)."""
    import matplotlib.pyplot as plt
    import math
    import numpy as np

    # ── m_prop : linear ──
    finite_mp = [v for v in m_prop_vals if not math.isnan(v)]
    if finite_mp:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(finite_mp, bins=n_bins, color='tab:blue',
                edgecolor='black', linewidth=0.3)
        ax.set_xlabel('m_prop [kg]')
        ax.set_ylabel('sample count')
        ax.set_title(f'm_prop distribution (n={len(finite_mp)})')
        ax.grid(alpha=0.3)
        fig.tight_layout()

    # ── TOF : linear ──
    finite_tof = [v for v in tof_vals if not math.isnan(v)]
    if finite_tof:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(finite_tof, bins=n_bins, color='tab:orange',
                edgecolor='black', linewidth=0.3)
        ax.set_xlabel('TOF [days]')
        ax.set_ylabel('sample count')
        ax.set_title(f'TOF distribution — linear (n={len(finite_tof)})')
        ax.grid(alpha=0.3)
        fig.tight_layout()

        # ── TOF : log-spaced bins ──
        pos = [v for v in finite_tof if v > 0]
        if pos:
            log_edges = np.logspace(
                math.log10(min(pos)), math.log10(max(pos)), n_bins + 1
            )
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(pos, bins=log_edges, color='tab:green',
                    edgecolor='black', linewidth=0.3)
            ax.set_xscale('log')
            ax.set_xlabel('TOF [days]  (log-spaced bins)')
            ax.set_ylabel('sample count')
            ax.set_title(f'TOF distribution — log (n={len(pos)})')
            ax.grid(alpha=0.3, which='both')
            fig.tight_layout()

    plt.show()


# ──────────────────────────────────────────────────────────────
# 메인 분석
# ──────────────────────────────────────────────────────────────
def analyze(csv_path, n_bins=40):
    if not os.path.exists(csv_path):
        print(f"[ERROR] csv not found: {csv_path}")
        return 1

    print("=" * 70)
    print(f"  Inspecting : {csv_path}")
    print("=" * 70)

    rows = load_csv_rows(csv_path)
    n_total = len(rows)
    if n_total == 0:
        print("  (csv 가 비어 있음)")
        return 0

    n_success = sum(1 for r in rows if r.get('success', '0') == '1')
    n_fail    = n_total - n_success

    print(f"  total rows   : {n_total}")
    print(f"  success (1)  : {n_success}  ({100 * n_success / n_total:.3f}%)")
    print(f"  fail    (0)  : {n_fail}     ({100 * n_fail    / n_total:.3f}%)")
    print()

    # 실패 분석은 fail 이 있을 때만
    fail_rows = [r for r in rows if r.get('success', '0') != '1']
    if fail_rows:
        # 1) m_prop / TOF / h_P 가 모두 NaN/empty 인가 검증
        n_with_value_in_fail = 0
        for r in fail_rows:
            for key in ('m_prop_kg', 'TOF_days', 'h_P_km'):
                val = r.get(key, '')
                if val not in ('', 'nan', 'NaN'):
                    try:
                        float(val)
                        n_with_value_in_fail += 1
                        break
                    except ValueError:
                        pass
        print(f"  fail 중 m_prop/TOF/h_P 에 finite 값이 남은 행 : "
              f"{n_with_value_in_fail}")
        if n_with_value_in_fail > 0:
            print("    → 이론상 0 이어야 정상. 0 이 아니면 데이터 일관성 문제.")
        print()

        # 2) 입력 feature 분포 (success vs fail)
        print("  입력 feature 분포 비교 (success 행 vs fail 행) :")
        print("                       success                  |"
              "                  fail")
        print("    feature    |    min       max      mean    "
              " |    min       max      mean")
        print("    " + "-" * 80)
        feature_cols = ['h_D1_km', 'RAAN_D1_deg', 'm_D1_kg',
                        'h_D2_km', 'RAAN_D2_deg', 'm_SC_kg']
        success_rows_for_stat = [r for r in rows if r.get('success', '0') == '1']
        for col in feature_cols:
            s_vals = [_to_float_or_nan(r.get(col, '')) for r in success_rows_for_stat]
            f_vals = [_to_float_or_nan(r.get(col, '')) for r in fail_rows]
            s_min, s_max, s_mean = _basic_stats(s_vals)
            f_min, f_max, f_mean = _basic_stats(f_vals)
            print(f"    {col:11s}|{_fmt(s_min)}  {_fmt(s_max)}  {_fmt(s_mean)}"
                  f"  |{_fmt(f_min)}  {_fmt(f_max)}  {_fmt(f_mean)}")
        print()

        # 3) D1 == D2 가 fail 에 섞여 있는지 확인 (있으면 안 됨)
        n_same_debris = sum(
            1 for r in fail_rows
            if r.get('d1_name') and r.get('d1_name') == r.get('d2_name')
        )
        if n_same_debris > 0:
            print(f"  [WARN] d1_name == d2_name 인 fail 행 : {n_same_debris}")
            print("         (정상 생성 코드에서는 0 이어야 함)")
        else:
            print("  d1 == d2 인 fail 행 : 0  (정상)")
        print()

        # 4) 처음 10 개 실패 행 예시
        print("  실패 sample 처음 10 개 :")
        sample_keys = ['sample_id', 'd1_name', 'd2_name',
                       'h_D1_km', 'h_D2_km', 'RAAN_D1_deg', 'RAAN_D2_deg']
        available = [k for k in sample_keys if k in fail_rows[0]]
        header = '    ' + '  '.join(f"{k:>12s}" for k in available)
        print(header)
        for r in fail_rows[:10]:
            line = '    ' + '  '.join(
                f"{str(r.get(k, '')):>12s}" for k in available)
            print(line)
        print()

    # ────────────────────────────────────────────
    # 히스토그램 (success=1 행 기준 — 출력 m_prop, TOF 분포)
    # matplotlib 창 3개 (m_prop / TOF linear / TOF log) 를 띄운다.
    # ────────────────────────────────────────────
    success_rows = [r for r in rows if r.get('success', '0') == '1']
    m_prop_vals = [_to_float_or_nan(r.get('m_prop_kg', '')) for r in success_rows]
    tof_vals    = [_to_float_or_nan(r.get('TOF_days',  '')) for r in success_rows]

    print("=" * 70)
    print(f"  Histograms (success=1 만, n_bins={n_bins}) — 창을 닫으면 종료")
    print("=" * 70)
    _show_histograms(m_prop_vals, tof_vals, n_bins)

    print("=" * 70)
    if n_fail > 0:
        print(f"  요약 : {n_fail} / {n_total} ({100 * n_fail / n_total:.3f}%) "
              f"행이 실패. 학습 전에 drop_failed_samples.py 로 제거 권장.")
    else:
        print("  요약 : 실패 sample 없음.")
    print("=" * 70)
    return 0


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Inspect samples + show histograms (m_prop, TOF)"
    )
    ap.add_argument(
        '--csv', default='data/training_data.csv',
        help='입력 csv 경로 (기본: data/training_data.csv)',
    )
    ap.add_argument(
        '--bins', type=int, default=40,
        help='히스토그램 bin 수 (기본 40)',
    )
    args = ap.parse_args()
    return analyze(args.csv, n_bins=args.bins)


if __name__ == '__main__':
    sys.exit(main() or 0)