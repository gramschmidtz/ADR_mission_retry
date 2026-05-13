"""
drop_failed_samples.py
======================
data/training_data.csv 에서 success=0 인 행을 제거한 새 csv 를 만든다.
기본 동작은 **원본을 안전하게 백업** 한 뒤 같은 파일명에 덮어쓴다.

사용 :
  # 기본 : data/training_data.csv 를 in-place 정리, .bak 백업 자동 생성
  python scripts/drop_failed_samples.py

  # 다른 csv 지정
  python scripts/drop_failed_samples.py --csv data/foo.csv

  # 별도 출력 파일에 쓰고 원본 보존 (in-place 아님)
  python scripts/drop_failed_samples.py --out data/training_data.clean.csv

  # 백업 만들지 않기
  python scripts/drop_failed_samples.py --no-backup

  # 무엇이 일어날지 보기만 하기 (실제 쓰지 않음)
  python scripts/drop_failed_samples.py --dry-run
"""

import os
import sys
import csv
import shutil
import argparse


# ──────────────────────────────────────────────────────────────
def count_rows(csv_path):
    """헤더 제외한 행 수 반환."""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)


def filter_csv(in_path, out_path, dry_run=False):
    """
    in_path csv 를 읽어 success=1 행만 out_path 에 쓴다.
    out_path 가 in_path 와 같으면 같은 자리에 임시파일 → rename 패턴 사용.

    Returns
    -------
    (n_total, n_success, n_fail)
    """
    n_total = 0
    n_success = 0

    if dry_run:
        # 행 개수만 세기
        with open(in_path, 'r', newline='', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                n_total += 1
                if row.get('success', '0') == '1':
                    n_success += 1
        return (n_total, n_success, n_total - n_success)

    # in-place 인지 판단
    inplace = os.path.abspath(in_path) == os.path.abspath(out_path)
    tmp_path = out_path + '.tmp' if inplace else out_path

    os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)

    with open(in_path, 'r', newline='', encoding='utf-8') as f_in, \
         open(tmp_path, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"빈 csv 또는 헤더 없음: {in_path}")
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            n_total += 1
            if row.get('success', '0') == '1':
                writer.writerow(row)
                n_success += 1

    if inplace:
        # 임시 파일을 원본 자리로 이동 (POSIX 에서 atomic)
        os.replace(tmp_path, out_path)

    return (n_total, n_success, n_total - n_success)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Drop rows with success=0 from training_data.csv"
    )
    ap.add_argument(
        '--csv', default='data/training_data.csv',
        help='입력 csv 경로 (기본: data/training_data.csv)',
    )
    ap.add_argument(
        '--out', default=None,
        help='출력 csv 경로 (기본: 입력과 동일하게 in-place 갱신)',
    )
    ap.add_argument(
        '--no-backup', action='store_true',
        help='in-place 모드에서 .bak 백업을 만들지 않음',
    )
    ap.add_argument(
        '--dry-run', action='store_true',
        help='실제 파일 변경 없이 영향만 출력',
    )
    args = ap.parse_args()

    in_path  = args.csv
    out_path = args.out or in_path
    inplace  = os.path.abspath(in_path) == os.path.abspath(out_path)

    if not os.path.exists(in_path):
        print(f"[ERROR] csv not found: {in_path}")
        return 1

    print("=" * 70)
    print(f"  Drop failed (success=0) samples")
    print("=" * 70)
    print(f"  in  : {in_path}")
    print(f"  out : {out_path}   {'(in-place)' if inplace else ''}")
    print(f"  dry-run : {args.dry_run}")

    # 백업 생성 (in-place 일 때만 의미 있음)
    if inplace and not args.no_backup and not args.dry_run:
        bak_path = in_path + '.bak'
        # 백업 이미 있으면 번호 붙임
        i = 1
        while os.path.exists(bak_path):
            bak_path = f"{in_path}.bak{i}"
            i += 1
        shutil.copy2(in_path, bak_path)
        print(f"  backup : {bak_path}")
    print("-" * 70)

    n_total, n_success, n_fail = filter_csv(
        in_path, out_path, dry_run=args.dry_run)

    print(f"  read   : {n_total} rows")
    print(f"  keep   : {n_success} rows  ({100 * n_success / max(n_total,1):.3f}%)")
    print(f"  drop   : {n_fail} rows     ({100 * n_fail    / max(n_total,1):.3f}%)")

    if args.dry_run:
        print(f"  [dry-run] 파일 변경 없음.")
    else:
        new_total = count_rows(out_path)
        print(f"  output : {new_total} rows in {out_path}")
        if new_total != n_success:
            print(f"  [WARN] 기록 행 수가 일치하지 않음 — 확인 필요.")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)