# ADR Mission Design with Neural Networks

논문 **Viavattene et al. (2022) "Design of multiple space debris removal
missions using machine learning"** (*Acta Astronautica* 193, 277–286) 의 재구현.

저궤도 우주 쓰레기 제거 (Active Debris Removal, ADR) 임무 설계를 위한
SS-ANN (Sequence Search + Artificial Neural Network) 파이프라인을 PyTorch
로 구현한다.

```
debris database  →  transfer cost dataset  →  MLP 학습  →  sequence search  →  Fig.11 검증
   (yaml)            (csv, 약 25만 샘플)        (PyTorch)     (beam search)        (matplotlib)
```

---

## 핵심 아이디어

1. **저추력 transfer model** (논문 §2.2) : chaser 가 한 debris (D₁) 를
   처분궤도까지 끌고 가서 release → phasing orbit 에서 J₂ 섭동으로 RAAN
   변화를 자연스럽게 누적 → 다음 debris (D₂) 로 transfer.
   한 transfer 의 비용은 *m_prop* (소비된 추진제) 와 *TOF* (비행 시간).
2. **ANN 으로 cost 함수 근사** (논문 §3) : (h_D1, Ω_D1, m_D1, h_D2, Ω_D2,
   m_SC) → (m_prop, TOF) 매핑을 MLP 로 학습. 한 transfer 평가가 ms → μs.
3. **Sequence search (논문 §4)** : 5000 debris pool 에서 beam search 로
   최대한 많은 debris 를 10 년 / 100 kg 추진제 안에 dispose 하는 순서 결정.

---

## 디렉토리 구조

```
ADR_mission_retry/
├── configs/
│   ├── constants.yaml          # 물리 상수, spacecraft 사양
│   ├── simulation.yaml         # 데이터 생성 / debris 분포 / 탐색 범위
│   ├── network.yaml            # MLP 구조 / 학습 hyperparameter
│   └── (random500debris.yaml, random5000debris.yaml — 스크립트로 자동 생성)
│
├── src/                        # 재사용 모듈
│   ├── config_loader.py        # yaml 로드 + 파라미터 통합
│   ├── dynamics.py             # MEE, Gauss matrix, J2, drag
│   ├── transfer_solver.py      # transfer model 핵심 (Tsiolkovsky, RAAN drift)
│   ├── transfer_grid.py        # (in/out) → m_prop, TOF.  h_P 는 grid search
│   ├── ann_dataset.py          # CSV → Dataset + normalization (Scaler)
│   ├── ann_model.py            # MLP (PyTorch nn.Module)
│   ├── ann_trainer.py          # train/val loop + TensorBoard
│   └── sequence_search.py      # beam search (ANN / solver 두 모드)
│
├── scripts/                    # 실행 스크립트
│   ├── make_random_debris.py        # 1. debris yaml 생성
│   ├── generate_training_data.py    # 2. 학습 데이터 CSV 생성 (병렬)
│   ├── check_samples.py             # 3a. 데이터 분포 점검 (히스토그램)
│   ├── drop_failed_samples.py       # 3b. 실패 sample 제거
│   ├── train_ann.py                 # 4. MLP 학습 (PyTorch + TensorBoard)
│   ├── run_sequence_search.py       # 5. SS-ANN 실행 → sequence csv
│   └── evaluate_sequence.py         # 6. solver 재평가 + Fig.11 + 오차 분석
│
└── results/                    # 산출물 (gitignored)
    ├── checkpoints/<ts>_alpha{α}/   # best.pt, scaler.npz, network.yaml, history.json
    ├── runs/<ts>_alpha{α}/          # TensorBoard event 파일
    └── sequences/                   # sequence csv + Fig.11 png
```

---

## 의존성

- Python ≥ 3.10
- numpy, scipy, pyyaml, matplotlib
- pandas (선택 — 분석용)
- **PyTorch** (학습 시 GPU 권장)
- tensorboard

```bash
conda create -n adrmission python=3.11
conda activate adrmission
pip install torch torchvision tensorboard
pip install numpy scipy pyyaml matplotlib
```

> **참고** : `setuptools 82+` 이면 tensorboard 가 `pkg_resources` 누락
> 으로 깨진다. `pip install "setuptools<81"` 로 다운그레이드.

---

## 사용 흐름 (end-to-end)

### 1. Debris database 생성

```bash
python scripts/make_random_debris.py
```

`configs/simulation.yaml` 의 `debris_generation` 섹션 (n / output / seed +
분포) 을 읽어 `configs/random500debris.yaml` (학습용) 과
`configs/random5000debris.yaml` (검증용) 을 생성한다. 서로 다른 seed.

### 2. 학습 데이터 생성

```bash
python scripts/generate_training_data.py
ADR_NUM_WORKERS=8 python scripts/generate_training_data.py   # 병렬
```

`configs/random500debris.yaml` 의 모든 순서쌍 (500×499 = 249,500) 에 대해
`src/transfer_grid.compute_transfer_grid` 를 호출해 m_prop, TOF 계산.
출력은 `data/training_data_alpha{α}.csv` (yaml 의 `alpha` 값이 자동
파일명에 부착됨).

`m_SC` 는 sample 마다 `uniform[m_dry=300, m0=400]` 으로 sampling (논문
§3.1 의 chaser mass 변동을 반영). 이렇게 안 하면 sequence search 의 후속
step 에서 chaser 가 가벼워졌을 때 ANN 이 out-of-distribution 영역에 빠짐.

병렬 처리 : multiprocessing.Pool + imap_unordered. 1 코어 약 12 분, 8
코어 약 1.5 분.

### 3. 데이터 점검 (선택)

```bash
python scripts/check_samples.py
python scripts/drop_failed_samples.py
```

- `check_samples` : 성공/실패 rate, 입력 분포 비교, m_prop / TOF 히스토그램
  (linear / log-spaced) 표시.
- `drop_failed_samples` : `success=0` 행 제거 (자동 .bak 백업).

### 4. MLP 학습

```bash
python scripts/train_ann.py
```

`configs/network.yaml` 의 모든 설정을 따른다 :

| 항목 | 기본값 |
|---|---|
| 입력 | 6 (h_D1, Ω_D1, m_D1, h_D2, Ω_D2, m_SC) |
| 출력 | 2 (m_prop, TOF) |
| Hidden | [64, 64, 64] |
| Activation | tanh |
| Optimizer | Adam, lr=1e-3 |
| Loss | MSE |
| Scheduler | ReduceLROnPlateau |
| Early stop | patience 20 |
| Split | 80/10/10 (random, seed 고정) |
| Normalization | 입력 minmax, 출력 m_prop minmax / TOF log1p+minmax |

출력 :

- `results/checkpoints/<timestamp>_alpha{α}/best.pt` — val loss 최소 모델
- `results/checkpoints/<timestamp>_alpha{α}/scaler.npz` — 정규화 파라미터
- `results/checkpoints/<timestamp>_alpha{α}/history.json` — epoch 별 metric
- `results/runs/<timestamp>_alpha{α}/` — TensorBoard logs

```bash
tensorboard --logdir results/runs
```

기록되는 metric : loss (train/val 의 total/m_prop/TOF 분리), R² (val/test),
MAE (kg, day), 산점도 figure, 오차 histogram, learning rate, epoch time.

### 5. Sequence search

```bash
python scripts/run_sequence_search.py \
    --alpha 0.0 \
    --mode ann \
    --ckpt-dir results/checkpoints/<timestamp>_alpha0.0
```

5000 debris 의 beam search (N_S=100, T_max=10y, t_stay=30day) 를 실행.
첫 depth 에서 5000 시작점 × 4999 후보를 ANN batch forward (GPU 권장) 로
한꺼번에 평가 후 total TOF 짧은 100 만 keep. 이후 매 depth 마다 동일.

출력 : `results/sequences/seq_alpha{α}_ann_<ts>.csv`
(step, src/dst, t_start_day, ann_TOF_day, ann_m_prop_kg, cum_…).

`--mode solver` 옵션으로 ANN 없이 솔버 직접 호출도 가능 (디버깅용,
5000 풀스케일은 약 20 시간 소요 — `--starting-subsample N` 으로 줄임).

### 6. Solver 재평가 + Fig. 11 시각화

```bash
python scripts/evaluate_sequence.py \
    --seq-csv results/sequences/seq_alpha0.0_ann_<ts>.csv \
    --alpha 0.0
```

각 step 을 `compute_transfer_grid` 로 재평가해서:

- **시각화** (논문 Fig. 11 재현) : 좌측 y = altitude (파란, phase 별 5점
  T1→T2a→Tp→T2b→capture), 우측 y = cumulative m_prop (주황, 굵은 = solver
  phase 별 누적, 점선 = ANN 누적). 시간 누적은 solver TOF 기준 (ANN 의
  부정확한 시각 정보 무시).
- **오차 분석** : step 별 ANN vs solver 의 m_prop 오차%, TOF 오차%, 통계
  (mean / std / |max|). Phase 별 추진제 분해 (T1 / T2a / drag / T2b).

출력 : `..._eval.csv` (전 step 의 solver 값 + 오차) + `..._eval.png`
(Fig. 11 한 panel).

---

## 핵심 설계 결정

### Transfer model

- **h_P (phasing orbit 고도) 는 grid search** 로 결정 (Brent 미사용).
  J(h_P) 는 h_P = h_D2 에서 식 (25) 의 분모 (Ω̇_P − Ω̇_D2) 가 0 이 되는
  극을 가지며 양 가지가 단절된 양봉 구조라, Brent
  (`scipy.minimize_scalar(method='bounded')`) 가 한 가지에 갇혀
  엉뚱한 h_P (예: 상한 2000 km) 를 반환하는 사례가 확인됨.
  50 km 간격, 33점 grid 가 `transfer_solver.optimize_phasing_orbit` 과
  `transfer_grid.compute_transfer_grid` 양쪽에서 모두 동일하게 사용된다.
- ΔV 산출 : Hohmann two-burn, 부호 무관 절대값 합. TOF 산출 :
  Tsiolkovsky 해석해 + low-thrust 적분 (수치적분 회피로 1200× 가속).
- RAAN drift 는 식 (22) 의 J₂ 평균 변화율. 적분은 phase 별 사다리꼴.
- Drag 보상은 phasing orbit 에서만 모델링 (논문 식 26 의 ΔV_P).

### Sequence search

- **Pruning 기준은 누적 total TOF** (논문 §4 본문 "shorter transfer time").
  α 값에 따라 transfer cost 자체가 달라지는 것을 통해 sequence 품질에
  간접 영향. α 값마다 별도 학습 데이터 / 모델 필요.
- 첫 depth 에서 5000 시작점 모두 시드 (chaser 가 첫 debris 위치에 이미
  도착했다고 가정 — 논문 §4 의 j ∈ [1, N] 해석).
- ANN batch evaluation : 65536 chunk 로 GPU forward.
- RAAN propagation 벡터화 : 5000 debris × 매 depth = 1 ms.

### MLP architecture (논문 §3 의 모호함)

논문이 hidden / activation / optimizer 를 명시하지 않으므로 표준값 사용 :
- 3 hidden × 64 unit, tanh, Adam (논문이 시사하는 MATLAB nntool 의 기본
  Levenberg-Marquardt + tanh 와 유사).
- 출력의 long-tail TOF 는 `log1p + minmax` 로 정규화 (필수 — 안 하면 큰
  TOF 영역에서 회귀 실패).

### α 별 산출물 추적

yaml 의 `training_data.alpha` 값을 모든 파일명 / 디렉토리명에 자동 부착 :

- `data/training_data_alpha0.0.csv`
- `data/training_data_alpha0.0.meta.yaml`
- `generate_training_data_alpha0.0.log`
- `results/checkpoints/<ts>_alpha0.0/`
- `results/runs/<ts>_alpha0.0/`
- `results/sequences/seq_alpha0.0_ann_<ts>.{csv,png}`

α 만 yaml 에서 바꾸면 모든 후속 산출물이 자동으로 다른 namespace 에
저장된다. 동일 α 의 dataset → model → sequence 추적이 쉬움.

---

## 진단 / 분석 스크립트 (선택)

| 스크립트 | 용도 |
|---|---|
| `scripts/test_J.py` | J(h_P) 다봉 형태 1D 진단 |
| `scripts/sweep_raan_d2.py` | RAAN_D2 sweep, 4-panel |
| `scripts/sweep_raan_hp.py` | (RAAN_D2 × h_P) 2D heatmap |
| `scripts/sweep_alpha_raan_hp.py` | (α × RAAN_D2) 의 h_P\* 분포 |
| `scripts/visualize_transfer.py` | 한 transfer 의 상세 시각화 |

---

## 알려진 한계

1. **논문의 절벽 / stiff dataset 문제** : RAAN drift 분모가 0 에 가까운
   영역 (Ω̇_h_P ≈ Ω̇_h_D2 또는 ΔΩ ≈ 0) 에서 J(h_P) 가 cliff 모양 → 데이터에
   long-tail 분포. ANN 이 이 영역에서 평균값으로 회귀해 worst-case 오차
   50-90% 발생 가능. 그러나 |오차| > 5% sample 은 전체의 2% 미만이라
   sequence search 의 beam pruning 으로 흡수됨.
2. **ANN 모드 시각화의 한계** : ANN 은 phase 시간 / h_P 를 예측하지 않으므로
   sequence search 단계에서는 transfer 전체를 한 segment 로만 다룸.
   `evaluate_sequence.py` 가 sequence 결정 후 solver 로 재평가해서 phase
   detail 을 얻음.
3. **α 별 별도 학습 필요** : 한 ANN 모델은 한 α 값에 종속. Sequence A/B/C
   (α = 0 / 0.95 / 1) 를 모두 만들려면 데이터 + 학습 × 3.
4. **RAAN circular encoding 미적용** : 0°/360° 경계 sample 에서 ANN 오차
   증가. sin/cos encoding 추가로 추가 개선 가능 (`input_dim 6 → 8`).

---

## 라이선스 / 참고

- 본 구현은 학술 재현 목적.
- 원 논문 :
  Viavattene, G., et al. (2022). *Design of multiple space debris removal
  missions using machine learning.* Acta Astronautica, 193, 277–286.