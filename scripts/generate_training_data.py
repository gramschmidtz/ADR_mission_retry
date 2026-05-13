"""
generate_training_data.py
=========================
configs/random500debris.yaml (또는 simulation.yaml 의 training_data.debris_yaml)
의 N 개 debris 에 대해 **모든 순서쌍 (D_i, D_j), i ≠ j** 의 transfer 데이터
를 생성한다 (N = 500 이면 250,000 - 500 = 249,500 sample).

각 sample 은 src.transfer_grid.compute_transfer_grid 를 호출해서
(입력) h_D1, Ω_D1, m_D1, h_D2, Ω_D2, m_SC
(출력) m_prop [kg], TOF [s]
를 얻고, **CSV 한 파일** 로 저장한다. 학습용 표준 tabular 형식.

부가적으로 같은 디렉토리에 `*.meta.yaml` 을 만들어 생성 파라미터
(α, m_SC, h_P grid, source debris yaml, success/fail 수, debris 분포) 를
기록한다. 데이터 + metadata 분리는 ML 실무 표준 패턴.

CSV 컬럼
--------
  sample_id            : 1..N×(N-1) 번호
  d1_name, d2_name     : debris yaml 의 key (디버깅용)
  h_D1_km, RAAN_D1_deg, m_D1_kg
  h_D2_km, RAAN_D2_deg, m_SC_kg          ← ANN 입력 6개 (논문 식 30)
  m_prop_kg, TOF_days                    ← ANN 출력 2개 (논문 식 31)
  h_P_km                                 ← 부수 정보 (최적 phasing 고도)
  success                                ← bool (feasibility)

α 와 h_P grid 는 simulation.yaml 의 search 섹션을 사용.

병렬 처리
---------
환경변수 ADR_NUM_WORKERS 또는 스크립트 상단 NUM_WORKERS 로 worker 수 지정.
0 또는 1 이면 single-thread, 2 이상이면 multiprocessing.Pool 사용.

실행 :
  python scripts/generate_training_data.py
  ADR_NUM_WORKERS=8 python scripts/generate_training_data.py
"""

import sys
import os
import time
import csv
import logging
import yaml
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import load_params, load_debris_list
from src.transfer_grid import compute_transfer_grid


# ──────────────────────────────────────────────────────────────
# 사용자 설정 — 기본값은 simulation.yaml 의 training_data 섹션에서 가져옴
# ──────────────────────────────────────────────────────────────
# 진행 상황 출력 주기 (sample 수)
PROGRESS_INTERVAL = 1000

# 병렬 worker 수 : 환경변수 우선, 없으면 CPU 코어 수 - 1
NUM_WORKERS = int(os.environ.get(
    'ADR_NUM_WORKERS',
    max(1, (os.cpu_count() or 2) - 1),
))


# ──────────────────────────────────────────────────────────────
# 로깅 — 콘솔만 우선 설정. file handler 는 main() 에서 alpha 를 알게 된
#       후 _attach_log_file 로 동적으로 부착한다.
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('gen_training')


def _attach_log_file(alpha):
    """alpha-stamped log 파일 핸들러를 부착한다.
    예 : generate_training_data_alpha0.0.log
    """
    log_path = f"generate_training_data_alpha{alpha}.log"
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(fh)
    return log_path


# ──────────────────────────────────────────────────────────────
# Worker (multiprocessing 용 — 반드시 top-level)
# ──────────────────────────────────────────────────────────────
_WORKER_PARAMS = None
_WORKER_ALPHA  = None
_WORKER_H_GRID = None


def _init_worker(params, alpha, h_P_grid):
    """multiprocessing pool initializer — worker 별 1회 호출.

    m_SC 는 task tuple 에 sample 별로 넣어서 전달 (학습 데이터의 m_SC 분포를
    [m_dry, m0] uniform 으로 만들기 위함).
    """
    global _WORKER_PARAMS, _WORKER_ALPHA, _WORKER_H_GRID
    _WORKER_PARAMS = params
    _WORKER_ALPHA  = alpha
    _WORKER_H_GRID = h_P_grid


def _process_pair(args):
    """
    args = (idx, d1_name, d1_dict, d2_name, d2_dict, m_sc_kg)
    return : dict — CSV row 한 줄
    """
    idx, d1_name, d1, d2_name, d2, m_sc = args
    try:
        m_prop, TOF, info = compute_transfer_grid(
            h_D1=d1['alt0_km'], RAAN_D1_deg=d1['RAAN'], m_D1=d1['mass'],
            h_D2=d2['alt0_km'], RAAN_D2_deg=d2['RAAN'],
            m_SC=m_sc,
            params=_WORKER_PARAMS, alpha=_WORKER_ALPHA,
            h_P_grid_km=_WORKER_H_GRID,
        )
        success = bool(info.get('feasible', False)
                       and np.isfinite(m_prop)
                       and np.isfinite(TOF))
        h_P_km = float(info['h_P_km']) if success else float('nan')
    except Exception:
        m_prop = float('nan')
        TOF    = float('nan')
        h_P_km = float('nan')
        success = False

    return {
        'sample_id'  : idx,
        'd1_name'    : d1_name,
        'd2_name'    : d2_name,
        'h_D1_km'    : float(d1['alt0_km']),
        'RAAN_D1_deg': float(d1['RAAN']),
        'm_D1_kg'    : float(d1['mass']),
        'h_D2_km'    : float(d2['alt0_km']),
        'RAAN_D2_deg': float(d2['RAAN']),
        'm_SC_kg'    : float(m_sc),
        'm_prop_kg'  : float(m_prop) if success else float('nan'),
        'TOF_days'   : (float(TOF) / 86400.0) if success else float('nan'),
        'h_P_km'     : float(h_P_km) if success else float('nan'),
        'success'    : int(success),   # 0 / 1 (csv 호환)
    }


CSV_COLUMNS = [
    'sample_id',
    'd1_name', 'd2_name',
    'h_D1_km', 'RAAN_D1_deg', 'm_D1_kg',
    'h_D2_km', 'RAAN_D2_deg', 'm_SC_kg',
    'm_prop_kg', 'TOF_days',
    'h_P_km', 'success',
]


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    # 설정 로드
    params, sim_cfg = load_params(
        constants_path='configs/constants.yaml',
        simulation_path='configs/simulation.yaml',
    )
    td_cfg = sim_cfg['training_data']
    alpha       = float(td_cfg['alpha'])
    debris_yaml = str(td_cfg['debris_yaml'])
    output_csv  = str(td_cfg['output_path'])

    # alpha 알게 된 시점에 log 파일 핸들러 부착
    log_path = _attach_log_file(alpha)

    # yaml 의 output_path 에 _alpha{값} 이 이미 있으면 그대로 사용.
    # 없으면 자동 부착 (예 : training_data.csv → training_data_alpha0.0.csv)
    import re
    if not re.search(r'_alpha[0-9]+(?:\.[0-9]+)?', output_csv):
        base, ext = os.path.splitext(output_csv)
        output_csv = f"{base}_alpha{alpha}{ext}"

    # m_SC 분포 : [m_dry, m0] uniform — 논문 §3.1
    #   m_dry = m0 - m_prop_max  (= 400 - 100 = 300 kg)
    m_sc_max = float(params['m0'])
    m_sc_min = float(params['m0']) - float(params['m_prop_max'])
    seed_m_sc = int(td_cfg.get('seed_m_sc', 20240601))

    # h_P grid 미리 1회 생성
    h_min  = max(params['disposal_alt_km'], params['h_P_min_km'])
    h_max  = params['h_P_max_km']
    h_step = params['h_step_km']
    h_P_grid = np.arange(h_min, h_max + h_step / 2, h_step)

    # debris set 로드
    debris_all = load_debris_list(debris_yaml)
    debris_keys = list(debris_all.keys())
    N = len(debris_keys)
    total_pairs = N * (N - 1)

    logger.info("=" * 70)
    logger.info(f"  Training data generation (논문 §3.1 input/output mapping)")
    logger.info("=" * 70)
    logger.info(f"  debris yaml      : {debris_yaml}")
    logger.info(f"  N (debris count) : {N}")
    logger.info(f"  total pairs      : {total_pairs}  (= N × (N-1))")
    logger.info(f"  α                : {alpha}")
    logger.info(f"  m_SC             : uniform [{m_sc_min:.1f}, {m_sc_max:.1f}] kg "
                f"(seed = {seed_m_sc})")
    logger.info(f"  h_P grid         : [{h_min:.1f}, {h_max:.1f}] km, "
                f"step {h_step} km ({len(h_P_grid)} 점)")
    logger.info(f"  output csv       : {output_csv}")
    logger.info(f"  log file         : {log_path}")
    logger.info(f"  workers          : {NUM_WORKERS}")
    logger.info("-" * 70)

    # 작업 목록 — (idx, d1_name, d1, d2_name, d2, m_sc)
    #   m_sc 는 task 생성 시점에 deterministic 하게 sampling
    rng = np.random.default_rng(seed_m_sc)
    m_sc_arr = rng.uniform(m_sc_min, m_sc_max, size=total_pairs)
    tasks = []
    idx = 0
    for ki in debris_keys:
        d1 = debris_all[ki]
        for kj in debris_keys:
            if ki == kj:
                continue
            d2 = debris_all[kj]
            tasks.append(
                (idx + 1, ki, d1, kj, d2, float(m_sc_arr[idx]))
            )
            idx += 1
    assert len(tasks) == total_pairs

    # 출력 디렉토리 + CSV 라이터 준비 (스트리밍 저장으로 메모리 절약)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    success_count = 0
    t_start = time.time()

    with open(output_csv, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        if NUM_WORKERS > 1:
            logger.info(f"  multiprocessing.Pool({NUM_WORKERS}) 시작 ...")
            pool = mp.Pool(
                processes=NUM_WORKERS,
                initializer=_init_worker,
                initargs=(params, alpha, h_P_grid),
            )
            try:
                iter_results = pool.imap_unordered(
                    _process_pair, tasks, chunksize=256)
                for n_done, row in enumerate(iter_results, start=1):
                    writer.writerow(row)
                    if row['success']:
                        success_count += 1
                    if (n_done % PROGRESS_INTERVAL == 0) or (n_done == total_pairs):
                        elapsed = time.time() - t_start
                        rate    = n_done / max(elapsed, 1e-9)
                        eta     = (total_pairs - n_done) / max(rate, 1e-9)
                        logger.info(
                            f"  [{n_done:>7d}/{total_pairs}] "
                            f"({100 * n_done / total_pairs:5.1f}%)  "
                            f"success {success_count}/{n_done}  "
                            f"rate {rate:6.1f} /s  "
                            f"ETA {eta/60:6.1f} min"
                        )
            finally:
                pool.close()
                pool.join()
        else:
            logger.info(f"  single-thread 모드 (NUM_WORKERS={NUM_WORKERS})")
            _init_worker(params, alpha, h_P_grid)
            for n_done, task in enumerate(tasks, start=1):
                row = _process_pair(task)
                writer.writerow(row)
                if row['success']:
                    success_count += 1
                if (n_done % PROGRESS_INTERVAL == 0) or (n_done == total_pairs):
                    elapsed = time.time() - t_start
                    rate    = n_done / max(elapsed, 1e-9)
                    eta     = (total_pairs - n_done) / max(rate, 1e-9)
                    logger.info(
                        f"  [{n_done:>7d}/{total_pairs}] "
                        f"({100 * n_done / total_pairs:5.1f}%)  "
                        f"success {success_count}/{n_done}  "
                        f"rate {rate:6.1f} /s  "
                        f"ETA {eta/60:6.1f} min"
                    )

    elapsed = time.time() - t_start
    logger.info("-" * 70)
    logger.info(f"  완료 : {total_pairs} pair / {success_count} success "
                f"({100 * success_count / total_pairs:.1f}%)  "
                f"총 {elapsed / 60:.2f} 분  "
                f"({total_pairs / max(elapsed, 1e-9):.1f} pair/s)")

    # 메타데이터 yaml 저장 (생성 파라미터 보존)
    meta_yaml = output_csv.replace('.csv', '.meta.yaml')
    if meta_yaml == output_csv:
        meta_yaml = output_csv + '.meta.yaml'

    meta = {
        'source_debris_yaml' : debris_yaml,
        'num_debris'         : N,
        'num_pairs'          : total_pairs,
        'num_success'        : success_count,
        'success_rate'       : success_count / max(total_pairs, 1),
        'alpha'              : alpha,
        'm_SC_distribution'  : f"uniform[{m_sc_min:.1f}, {m_sc_max:.1f}] kg",
        'm_SC_min_kg'        : float(m_sc_min),
        'm_SC_max_kg'        : float(m_sc_max),
        'm_SC_seed'          : seed_m_sc,
        'h_P_min_km'         : float(h_min),
        'h_P_max_km'         : float(h_max),
        'h_P_step_km'        : float(h_step),
        'h_P_grid_points'    : int(len(h_P_grid)),
        'csv_columns'        : CSV_COLUMNS,
        'output_csv'         : output_csv,
        'elapsed_minutes'    : round(elapsed / 60.0, 3),
        'num_workers'        : NUM_WORKERS,
    }
    with open(meta_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  csv  저장 : {output_csv}")
    logger.info(f"  meta 저장 : {meta_yaml}")
    logger.info("=" * 70)


if __name__ == '__main__':
    mp.freeze_support()
    main()