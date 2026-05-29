"""
sequence_search.py
==================
논문 §4 Fig. 10 의 SS-ANN 알고리즘 구현.

알고리즘 :
  Breadth-first beam search.
    - 매 depth d 에서, beam 안의 모든 sequence × 모든 후보 debris 쌍을
      평가하여 (m_prop, TOF) 를 얻는다.
    - feasibility (남은 추진제, 남은 시간) 통과한 자손 후보들 중
      total TOF 가 가장 짧은 N_S 개만 다음 depth 의 beam 으로 넘김.
    - beam 이 비거나, 모든 자손이 feasibility 실패하면 종료.

가정 (논문 §2.2 / §4 / §3.1) :
  - chaser 가 처음부터 D_1 위치에 있다. 첫 edge = (D_1 → D_2).
    Sequence 의 첫 원소 D_1 은 chaser 가 처음 잡는 disposal 대상.
  - 한 transfer 한 호출에 T1 (D_prev → disposal) + phasing + T2b (D_next 도달)
    이 모두 포함됨 → compute_transfer_grid 와 동일.
  - capture (rendezvous + grab) 시간 t_stay = 30 일이 각 disposal 후 추가.
  - mission 종료 조건 : 누적시간 > T_max (10 년) 또는 추진제 잔량 ≤ 0.
  - RAAN propagation: Ω(t) = Ω_0 + Ω̇ × t  (식 22).
  - Beam pruning 기준 : sequence 의 **누적 total TOF**.

API
---
SequenceSearch.run() → List[Sequence]  (살아남은 모든 sequence, 깊이 내림차순)

Cost evaluation 은 두 모드 :
  mode='solver' : compute_transfer_grid 직접 호출 (정확, 한 호출 ~3ms).
  mode='ann'    : 학습된 MLP batch forward (GPU 권장, batch 한 호출당 us 수준).
"""

from __future__ import annotations

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from src.config_loader import load_debris_list
from src.transfer_solver import raan_drift_rate
from src.transfer_grid    import compute_transfer_grid


# 상수
DAY        = 86400.0          # [s]
T_STAY_SEC = 30.0 * DAY        # capture time per debris (§2.2)


# ──────────────────────────────────────────────────────────────
# Sequence dataclass
# ──────────────────────────────────────────────────────────────
@dataclass
class StepLog:
    """한 transfer 의 상세 기록 (Fig. 11 시각화용)."""
    src_idx     : int           # 출발 debris index (0-based)
    dst_idx     : int           # 도착 debris index
    src_alt_km  : float
    dst_alt_km  : float
    t_start_s   : float         # transfer 시작 시각 (이전 capture 끝난 시점)
    t_T1_end_s  : float         # T1 종료 시각 (disposal 도달)
    t_T2a_end_s : float         # T2a 종료 시각 (phasing 도달)
    t_Tp_end_s  : float         # phasing 종료 시각
    t_end_s     : float         # T2b 종료 시각 (D_next 도착) = capture 시작
    t_capture_end_s : float     # capture 끝 = 다음 step 시작
    disposal_alt_km : float
    h_P_km      : float
    m_prop_kg   : float         # 이 transfer 총 추진제
    TOF_s       : float         # 전체 mission 시간 (T1+T2a+Tp+T2b+Ts).
                                #   ANN 학습 타깃이 compute_transfer_grid 의
                                #   'TOF' (Ts 30일 capture 포함) 이므로 ANN 예측,
                                #   solver 반환값, 그리고 본 필드 모두 Ts 포함 기준.


@dataclass
class Sequence:
    """beam 안의 한 sequence."""
    visited_idx     : tuple              # 방문한 debris index 순서 (tuple of int)
    visited_set     : frozenset          # 방문 set (O(1) 조회)
    total_TOF_s     : float = 0.0        # 누적 transfer 시간 (capture 제외)
    total_time_s    : float = 0.0        # 누적 mission 시간 (capture 포함)
    m_prop_used_kg  : float = 0.0        # 누적 추진제 사용
    m_SC_now_kg     : float = 0.0        # 현재 chaser 질량
    steps           : list  = field(default_factory=list)   # StepLog 들

    @property
    def depth(self) -> int:
        return len(self.visited_idx)

    @property
    def last_idx(self) -> int:
        return self.visited_idx[-1]


# ──────────────────────────────────────────────────────────────
# Debris 정보 → numpy array 로 (벡터 연산용)
# ──────────────────────────────────────────────────────────────
class DebrisArray:
    """5000 debris 의 정적 정보를 numpy array 로 보관."""

    def __init__(self, debris_dict, params):
        self.names = list(debris_dict.keys())
        N = len(self.names)
        self.N = N

        # 정적 array
        self.alt_km    = np.array(
            [debris_dict[n]['alt0_km'] for n in self.names], dtype=np.float64)
        self.RAAN0_deg = np.array(
            [debris_dict[n]['RAAN']    for n in self.names], dtype=np.float64)
        self.mass_kg   = np.array(
            [debris_dict[n]['mass']    for n in self.names], dtype=np.float64)
        self.inc_deg   = np.array(
            [debris_dict[n]['i']       for n in self.names], dtype=np.float64)
        self.ecc       = np.array(
            [debris_dict[n]['e']       for n in self.names], dtype=np.float64)

        # 파생 정보
        Re_m  = params['Re']
        self.a_m       = Re_m + self.alt_km * 1000.0
        self.inc_rad   = np.deg2rad(self.inc_deg)
        self.RAAN0_rad = np.deg2rad(self.RAAN0_deg)

        # RAAN drift rate Ω̇ (s^-1) — 벡터로 한 번에 계산
        self.omega_dot = raan_drift_rate(
            self.a_m, self.ecc, self.inc_rad, params)

    def RAAN_at(self, t_s):
        """
        시각 t_s 에서 모든 debris 의 RAAN [rad]. shape (N,).
        t_s 는 scalar 또는 (M,) array 가능 — broadcasting.
        """
        return self.RAAN0_rad + self.omega_dot * t_s


# ──────────────────────────────────────────────────────────────
# Edge evaluator — 두 모드
# ──────────────────────────────────────────────────────────────
class _SolverEvaluator:
    """compute_transfer_grid 직접 호출. 한 호출 ~3ms.

    Batched call : Python loop 으로 1개씩 — 솔버는 본래 scalar API.
    """

    def __init__(self, params, alpha):
        self.params = params
        self.alpha  = alpha
        # h_P grid 미리 1회 생성 (vectorize 못 해도 호출 안 줄임은 가능)
        h_min  = max(params['disposal_alt_km'], params['h_P_min_km'])
        h_max  = params['h_P_max_km']
        h_step = params['h_step_km']
        self.h_P_grid_km = np.arange(h_min, h_max + h_step / 2, h_step)

    def evaluate(self, h_D1_km, RAAN_D1_rad, m_D1_kg,
                       h_D2_km, RAAN_D2_rad,
                       m_SC_kg):
        """
        모든 인자는 같은 길이 numpy array (M,).
        반환 : m_prop (M,), TOF_s (M,), feasible (M,) bool, h_P (M,)
        """
        M = h_D1_km.shape[0]
        m_prop = np.full(M, np.nan)
        TOF_s  = np.full(M, np.nan)
        h_P    = np.full(M, np.nan)
        ok     = np.zeros(M, dtype=bool)

        for k in range(M):
            try:
                mp, tof, info = compute_transfer_grid(
                    h_D1        = float(h_D1_km[k]),
                    RAAN_D1_deg = float(np.rad2deg(RAAN_D1_rad[k])),
                    m_D1        = float(m_D1_kg[k]),
                    h_D2        = float(h_D2_km[k]),
                    RAAN_D2_deg = float(np.rad2deg(RAAN_D2_rad[k])),
                    m_SC        = float(m_SC_kg[k]),
                    params      = self.params,
                    alpha       = self.alpha,
                    h_P_grid_km = self.h_P_grid_km,
                )
                if info.get('feasible', False) and np.isfinite(mp) and np.isfinite(tof):
                    m_prop[k] = mp
                    TOF_s[k]  = tof
                    h_P[k]    = info['h_P_km']
                    ok[k]     = True
            except Exception:
                pass
        return m_prop, TOF_s, ok, h_P


class _ANNEvaluator:
    """학습된 MLP 로 batch forward.

    csv 학습 데이터의 입력 6차원, 출력 2차원과 동일한 표현.
    inverse_y 로 원 단위 복원.
    h_P 는 ANN 학습 입력/출력에 없으므로 NaN.
    """

    def __init__(self, ckpt_dir, params, network_yaml='configs/network.yaml',
                 batch_size=65536, device=None):
        import torch
        import yaml
        from src.ann_dataset import Scaler
        from src.ann_model   import build_model_from_config

        self._torch = torch
        self.params = params
        self.batch_size = int(batch_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # network config (모델 구조)
        with open(network_yaml, 'r', encoding='utf-8') as f:
            ncfg = yaml.safe_load(f)
        self.network_cfg = ncfg

        # 모델 weight
        ckpt_path = os.path.join(ckpt_dir, 'best.pt')
        state = torch.load(ckpt_path, map_location=self.device)
        self.model = build_model_from_config(ncfg['model']).to(self.device)
        self.model.load_state_dict(state['model_state'])
        self.model.eval()

        # 정규화
        self.scaler = Scaler.load_npz(os.path.join(ckpt_dir, 'scaler.npz'))

    def evaluate(self, h_D1_km, RAAN_D1_rad, m_D1_kg,
                       h_D2_km, RAAN_D2_rad,
                       m_SC_kg):
        torch = self._torch
        M = h_D1_km.shape[0]
        # 입력 6 차원 배열 구성 — 학습 데이터와 같은 순서
        # INPUT_COLS = ['h_D1_km','RAAN_D1_deg','m_D1_kg',
        #               'h_D2_km','RAAN_D2_deg','m_SC_kg']
        X = np.stack([
            h_D1_km.astype(np.float32),
            np.rad2deg(RAAN_D1_rad).astype(np.float32),
            m_D1_kg.astype(np.float32),
            h_D2_km.astype(np.float32),
            np.rad2deg(RAAN_D2_rad).astype(np.float32),
            m_SC_kg.astype(np.float32),
        ], axis=1)

        # RAAN 을 [0, 360) 로 정규화 (학습 데이터 범위와 일치시키기 위해)
        X[:, 1] = np.mod(X[:, 1], 360.0)
        X[:, 4] = np.mod(X[:, 4], 360.0)

        # 정규화
        Xn = self.scaler.transform_x(X)

        # batch forward
        outs = []
        with torch.no_grad():
            for s in range(0, M, self.batch_size):
                e = min(s + self.batch_size, M)
                xb = torch.from_numpy(Xn[s:e]).to(self.device)
                yb = self.model(xb).cpu().numpy()
                outs.append(yb)
        Yn = np.concatenate(outs, axis=0)
        Y = self.scaler.inverse_y(Yn)   # (M, 2): m_prop_kg, TOF_days

        m_prop = Y[:, 0].astype(np.float64)
        TOF_s  = (Y[:, 1].astype(np.float64)) * DAY

        # feasibility : 모델 출력이 음수면 invalid 로 표시 (드물 것)
        ok = (m_prop > 0) & (TOF_s > 0) & np.isfinite(m_prop) & np.isfinite(TOF_s)
        h_P = np.full(M, np.nan)        # ANN 은 h_P 모름
        return m_prop, TOF_s, ok, h_P


def build_evaluator(mode, params, alpha=None, ckpt_dir=None,
                    network_yaml='configs/network.yaml',
                    ann_batch_size=65536):
    if mode == 'solver':
        if alpha is None:
            raise ValueError("mode='solver' 면 alpha 가 필요합니다.")
        return _SolverEvaluator(params, alpha)
    if mode == 'ann':
        if not ckpt_dir:
            raise ValueError("mode='ann' 이면 ckpt_dir 가 필요합니다.")
        return _ANNEvaluator(ckpt_dir, params,
                             network_yaml=network_yaml,
                             batch_size=ann_batch_size)
    raise ValueError(f"unsupported mode: {mode}")


# ──────────────────────────────────────────────────────────────
# Main sequence search
# ──────────────────────────────────────────────────────────────
class SequenceSearch:
    """
    Parameters
    ----------
    debris_yaml_path : str
    params           : dict   (load_params 결과)
    evaluator        : SolverEvaluator or ANNEvaluator
    beam_width       : int    N_S (논문 = 100)
    mission_max_year : float  논문 = 10
    t_stay_s         : float  논문 = 30 일
    starting_subsample : int or None
        첫 depth 의 시작점 sub-sample 크기. None 이면 모든 N (=5000) 사용.
        solver 모드에서 디버깅용으로 작은 값 사용 권장.
    starting_seed : int
        sub-sample 셔플 seed.
    verbose       : bool
    """

    def __init__(self, debris_yaml_path, params, evaluator,
                 beam_width=100,
                 mission_max_year=10.0,
                 t_stay_s=T_STAY_SEC,
                 starting_subsample=None,
                 starting_seed=42,
                 verbose=True):
        self.params = params
        self.evaluator = evaluator
        self.beam_width = int(beam_width)
        self.T_max_s = float(mission_max_year) * 365.25 * DAY
        self.t_stay_s = float(t_stay_s)
        self.starting_subsample = starting_subsample
        self.starting_seed = int(starting_seed)
        self.verbose = bool(verbose)

        # debris 로드
        debris = load_debris_list(debris_yaml_path)
        self.debris_dict = debris
        self.darr = DebrisArray(debris, params)

        # mass 제한
        self.m_dry = float(params['m0']) - float(params['m_prop_max'])
        self.m_initial = float(params['m0'])

    # ── 첫 depth 의 시작점 (D_1) 후보 ────────────────────
    def _initial_starts(self):
        N = self.darr.N
        all_idx = np.arange(N)
        if self.starting_subsample and self.starting_subsample < N:
            rng = np.random.default_rng(self.starting_seed)
            idx = rng.choice(N, size=self.starting_subsample, replace=False)
            idx = np.sort(idx)
            return idx
        return all_idx

    # ── 한 depth 의 확장 ────────────────────────────────
    def _expand_one_depth(self, beam: List[Sequence]) -> List[Sequence]:
        """
        현재 beam 의 모든 sequence 를 한 단계 확장.
        모든 자손 후보를 한 번에 batch evaluate 한 뒤,
        total TOF 짧은 상위 N_S 개만 다음 beam 으로 keep.
        """
        if not beam:
            return []

        # ─ 자손 후보 (parent_idx, child_idx) 페어 수집 (벡터로 build) ─
        # parent 마다 미방문 debris 만 후보로
        parent_ids   = []
        src_idx      = []   # = parent.last_idx
        dst_idx      = []
        parent_visit_t = []   # parent 의 누적 시간 (transfer 시작 시각)
        parent_m_D1    = []   # = parent.last_idx 의 mass
        parent_m_SC    = []   # = parent.m_SC_now_kg

        N = self.darr.N
        all_indices = np.arange(N)
        for p_i, parent in enumerate(beam):
            visited = parent.visited_set
            unvisited_mask = np.ones(N, dtype=bool)
            for v in visited:
                unvisited_mask[v] = False
            candidate = all_indices[unvisited_mask]
            if candidate.size == 0:
                continue
            src = parent.last_idx
            M = candidate.size
            parent_ids.append(np.full(M, p_i, dtype=np.int64))
            src_idx   .append(np.full(M, src, dtype=np.int64))
            dst_idx   .append(candidate)
            parent_visit_t.append(np.full(M, parent.total_time_s, dtype=np.float64))
            parent_m_D1   .append(np.full(M, self.darr.mass_kg[src], dtype=np.float64))
            parent_m_SC   .append(np.full(M, parent.m_SC_now_kg,   dtype=np.float64))

        if not parent_ids:
            return []

        parent_ids = np.concatenate(parent_ids)
        src_idx    = np.concatenate(src_idx)
        dst_idx    = np.concatenate(dst_idx)
        t_eval     = np.concatenate(parent_visit_t)
        m_D1_arr   = np.concatenate(parent_m_D1)
        m_SC_arr   = np.concatenate(parent_m_SC)

        # ─ RAAN propagation 한 번에 ─
        # parent 마다 다른 t_eval 이지만 같은 row 는 동일 t.
        RAAN_src = (self.darr.RAAN0_rad[src_idx]
                    + self.darr.omega_dot[src_idx] * t_eval)
        RAAN_dst = (self.darr.RAAN0_rad[dst_idx]
                    + self.darr.omega_dot[dst_idx] * t_eval)

        h_src = self.darr.alt_km[src_idx]
        h_dst = self.darr.alt_km[dst_idx]

        # ─ Edge cost 평가 (batch) ─
        m_prop_e, TOF_e_s, ok, h_P_e = self.evaluator.evaluate(
            h_src, RAAN_src, m_D1_arr,
            h_dst, RAAN_dst,
            m_SC_arr,
        )

        # ─ 자손 후보의 누적 metric 계산 + feasibility ─
        # total TOF (transfer 시간만 누적) — pruning 기준
        # total time (transfer + capture) — feasibility check 용
        parent_total_TOF  = np.array([beam[p].total_TOF_s   for p in parent_ids])
        parent_total_time = np.array([beam[p].total_time_s for p in parent_ids])
        parent_m_used     = np.array([beam[p].m_prop_used_kg for p in parent_ids])

        cand_total_TOF  = parent_total_TOF  + TOF_e_s
        cand_total_time = parent_total_time + TOF_e_s + self.t_stay_s
        cand_m_used     = parent_m_used + m_prop_e
        cand_m_SC_after = self.m_initial - cand_m_used

        feasible = (
            ok
            & (cand_total_time <= self.T_max_s)
            & (cand_m_SC_after >= self.m_dry)
        )

        if not feasible.any():
            return []

        # 살아남은 자손에 대해 total TOF 가 짧은 상위 N_S 선택
        fi = np.flatnonzero(feasible)
        kept_sort_key = cand_total_TOF[fi]
        # argsort 후 상위 beam_width
        order = np.argsort(kept_sort_key)
        keep  = fi[order[:self.beam_width]]

        # ─ 새 beam 의 Sequence 객체 만들기 ─
        new_beam = []
        for k in keep:
            p_i = int(parent_ids[k])
            parent = beam[p_i]
            src    = int(src_idx[k])
            dst    = int(dst_idx[k])

            # StepLog 구성 — phase 별 시간은 ANN 모드에서는 알 수 없음 → solver
            # 모드에서만 정확. ANN 모드는 transfer 전체를 한 segment 로 단순화.
            t_start  = parent.total_time_s
            tof_s    = float(TOF_e_s[k])
            # ANN 모드는 phase 분해 없음 → 0 으로 채우고 시각화에서 단순 linear
            step = StepLog(
                src_idx    = src,
                dst_idx    = dst,
                src_alt_km = float(self.darr.alt_km[src]),
                dst_alt_km = float(self.darr.alt_km[dst]),
                t_start_s  = t_start,
                t_T1_end_s = t_start,           # 모름 → 0 length
                t_T2a_end_s= t_start,
                t_Tp_end_s = t_start,
                t_end_s    = t_start + tof_s,
                t_capture_end_s = t_start + tof_s + self.t_stay_s,
                disposal_alt_km = float(self.params['disposal_alt_km']),
                h_P_km     = float(h_P_e[k]) if np.isfinite(h_P_e[k]) else float('nan'),
                m_prop_kg  = float(m_prop_e[k]),
                TOF_s      = tof_s,
            )
            new_seq = Sequence(
                visited_idx    = parent.visited_idx + (dst,),
                visited_set    = parent.visited_set | {dst},
                total_TOF_s    = float(cand_total_TOF[k]),
                total_time_s   = float(cand_total_time[k]),
                m_prop_used_kg = float(cand_m_used[k]),
                m_SC_now_kg    = float(cand_m_SC_after[k]),
                steps          = list(parent.steps) + [step],
            )
            new_beam.append(new_seq)
        return new_beam

    # ── 첫 depth (D_1 후보들로 sequence 1개씩 시드) ─────
    def _initial_beam(self) -> List[Sequence]:
        starts = self._initial_starts()
        beam = []
        for s in starts:
            seq = Sequence(
                visited_idx    = (int(s),),
                visited_set    = frozenset({int(s)}),
                total_TOF_s    = 0.0,
                total_time_s   = 0.0,                # chaser 는 D_1 위치에 이미 있다
                m_prop_used_kg = 0.0,
                m_SC_now_kg    = self.m_initial,
                steps          = [],
            )
            beam.append(seq)
        return beam

    # ── 메인 run ─────────────────────────────────────────
    def run(self):
        t0 = time.time()
        beam = self._initial_beam()
        if self.verbose:
            print(f"[seq-search] starting beam size = {len(beam)} "
                  f"(starts = {'all' if not self.starting_subsample else self.starting_subsample})")
            print(f"[seq-search] beam_width = {self.beam_width}, "
                  f"T_max = {self.T_max_s/DAY/365.25:.2f} y, "
                  f"m_dry = {self.m_dry} kg")

        # 모든 sequence 의 완료된 (= 더 이상 자랄 수 없는) 결과를 보관
        completed = []

        depth = 1
        while beam:
            t_d = time.time()
            new_beam = self._expand_one_depth(beam)
            elapsed_d = time.time() - t_d

            if not new_beam:
                # 모든 sequence 가 더 이상 못 자람 → 현재 beam 이 최종 후보
                completed.extend(beam)
                if self.verbose:
                    print(f"[seq-search] depth {depth}: 모든 자손 infeasible. "
                          f"종료. ({elapsed_d:.1f}s)")
                break

            if self.verbose:
                # 통계
                tofs = np.array([s.total_TOF_s   for s in new_beam])
                times= np.array([s.total_time_s for s in new_beam])
                mus  = np.array([s.m_prop_used_kg for s in new_beam])
                print(f"[seq-search] depth {depth+1:>2d}: beam={len(new_beam):>3d}  "
                      f"TOF[d] min/med/max = "
                      f"{tofs.min()/DAY:.1f}/{np.median(tofs)/DAY:.1f}/{tofs.max()/DAY:.1f}  "
                      f"time[y] = "
                      f"{times.min()/DAY/365.25:.2f}/{np.median(times)/DAY/365.25:.2f}/"
                      f"{times.max()/DAY/365.25:.2f}  "
                      f"m_prop[kg] min/med/max = "
                      f"{mus.min():.2f}/{np.median(mus):.2f}/{mus.max():.2f}  "
                      f"({elapsed_d:.1f}s)")

            beam = new_beam
            depth += 1

        # 깊이 내림차순 → 동률이면 m_prop 작은 순
        completed.sort(key=lambda s: (-s.depth, s.m_prop_used_kg))
        total = time.time() - t0
        if self.verbose:
            print(f"[seq-search] 총 {len(completed)} sequences, "
                  f"max depth = {completed[0].depth if completed else 0}, "
                  f"elapsed {total/60:.2f} min")
        return completed