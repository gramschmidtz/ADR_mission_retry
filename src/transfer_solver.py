"""
transfer_solver.py
==================
논문 Section 2.2의 transfer model을 정확히 구현한다.

전체 전이 구조 (Fig. 2):
  T1   : D1 orbit → disposal orbit          (thrust ON,  고도 하강)
  T2,a : disposal orbit → phasing orbit     (thrust ON,  고도 변경, 0일 수도 있음)
  Tp   : phasing orbit 대기                  (thrust OFF, J2 섭동으로 RAAN 자연 변화)
  T2,b : phasing orbit → D2 orbit           (thrust ON,  고도 상승)
  Ts   : D2 orbit 대기                       (thrust OFF, 30일 고정, rendezvous)

핵심 아이디어:
  - 고도 변화(T1, T2)는 thrust를 사용하며 ΔV로 추진제 계산
  - RAAN 변화는 thrust를 쓰지 않고 J2 섭동에 의한 자연 drift 이용
  - phasing orbit 고도 h_P 를 grid search 로 최적화하여 RAAN을 맞춤 (식 28)
  - eclipse 는 무시 (추력 항상 ON 가정)
"""

import numpy as np


# ────────────────────────────────────────────────
# 1. 기본 물리 계산
# ────────────────────────────────────────────────

def raan_drift_rate(a, e, i, params):
    """
    J2 섭동에 의한 궤도 평균 RAAN 변화율을 계산한다 (논문 식 22).

    Ω̇ = -3/2 * J2 * sqrt(μ) * Re² / [a^(7/2) * (1-e²)²] * cos(i)

    Parameters
    ----------
    a      : 반장축 [m]
    e      : 이심률
    i      : 경사각 [rad]
    params : dict

    Returns
    -------
    omega_dot : float [rad/s]  (음수 = 서쪽으로 회전)
    """
    mu = params['mu']
    Re = params['Re']
    J2 = params['J2']

    omega_dot = (
        -1.5 * J2 * np.sqrt(mu) * Re**2
        / (a**(3.5) * (1 - e**2)**2)
        * np.cos(i)
    )
    return omega_dot


# ────────────────────────────────────────────────
# 1-b. RAAN drift 시간 적분 헬퍼
#      (raan_drift_integration.py 와 동일 알고리즘:
#       a(t) 는 시간 선형 보간, Ω̇(a(t)) 를 사다리꼴 적분)
# ────────────────────────────────────────────────

def integrate_raan_thrust_leg(a_start, a_end, T_leg, e, i_rad, params,
                              n_sample=400):
    """
    추력 구간(T1, T2a, T2b) 의 누적 RAAN 변화량 ΔΩ 를 식 (22) 시간 적분으로 계산.
    a(t) 는 0 → T_leg 동안 a_start → a_end 로 시간 선형 보간 (저추력 근사).

    --- 왜 시간 적분인가 ---
    식 (22) Ω̇ ∝ a^(-7/2) 는 a 가 변하면 비선형적으로 변하므로, 추력 구간에서
    "평균 a 한 점에서의 Ω̇ × T_leg" 1차 근사보다 시간에 대한 사다리꼴 적분이
    훨씬 정확하다. raan_drift_integration.py / build_timeline 등 다른 모듈과
    동일한 적분 결과를 내야 phasing 일관성이 유지된다.

    --- 벡터화 최적화 원리 ---
    이전 구현은 list comprehension 으로 raan_drift_rate 를 n_sample (=400) 번
    파이썬 함수 호출했다. 한 번의 evaluate_phasing_orbit 이 3 leg × 400 = 1,200 회
    호출, optimize_phasing_orbit 의 h_P grid (~33 점) 가 그 만큼 evaluate 를
    돌리면 한 solve_transfer 당 ~40,000 함수 호출 → 파이썬 함수 호출 오버헤드가
    지배적.

    벡터화 :
      식 (22) Ω̇(a) = -1.5 · J2 · √μ · Re² / [a^(7/2) · (1-e²)²] · cos(i)
      에서 a 외의 항은 모두 상수이므로
        coef = -1.5 · J2 · √μ · Re² · cos(i) / (1-e²)²
        Ω̇(a) = coef / a^(3.5)
      로 분리 가능. a_arr (shape: n_sample,) 에 numpy 의 ** 연산을 한 번
      적용하면 n_sample 개 점의 Ω̇ 가 한 ufunc 호출로 산출된다.
      파이썬 인터프리터를 거치지 않고 C 레벨에서 처리되므로 함수 호출
      오버헤드가 사라진다.

    측정 결과 : solve_transfer 1 회당 44 ms → 2.3 ms (≈ 19× 향상),
                결과는 부동소수점 오차 범위 내에서 동일 (T2b 끝 RAAN 잔차
                -0.000017° 그대로 유지).

    Parameters
    ----------
    a_start, a_end : float [m]    구간 시작/끝 반장축
    T_leg          : float [s]    구간 시간
    e              : float        이심률
    i_rad          : float [rad]  경사각
    params         : dict
    n_sample       : int          적분 샘플 수 (기본 400)

    Returns
    -------
    delta_RAAN : float [rad]   T_leg 동안 누적된 RAAN 변화 (스칼라)
    """
    if T_leg <= 0:
        return 0.0

    # raan_drift_rate 인라인 (식 22) :
    #   Ω̇(a) = -1.5 · J2 · sqrt(μ) · Re² / (a^(7/2) · (1-e²)²) · cos(i)
    mu = params['mu']
    Re = params['Re']
    J2 = params['J2']
    coef = -1.5 * J2 * np.sqrt(mu) * Re**2 / (1.0 - e**2)**2 * np.cos(i_rad)

    t_arr = np.linspace(0.0, T_leg, n_sample)
    a_arr = a_start + (a_end - a_start) * (t_arr / T_leg)
    # 벡터 연산: 한 번에 n_sample 개 점을 처리
    omega_dot = coef / a_arr**3.5

    # 사다리꼴 적분 ∫_0^{T_leg} Ω̇(a(t)) dt
    dt = t_arr[1] - t_arr[0]   # 등간격이므로 단일값
    delta_RAAN = 0.5 * (omega_dot[0] + omega_dot[-1]) * dt + np.sum(omega_dot[1:-1]) * dt
    return float(delta_RAAN)


def integrate_raan_constant_leg(a_const, T_leg, e, i_rad, params):
    """
    추력 OFF 구간 (Tp, Ts) 의 누적 RAAN 변화량.
    a 가 일정하므로 해석적 형태 Ω̇·T_leg 로 충분 — 인터페이스 일관성을 위해 함수화.
    """
    if T_leg <= 0:
        return 0.0
    return raan_drift_rate(a_const, e, i_rad, params) * T_leg


def circular_velocity(a, mu):
    """원궤도 속도 [m/s]"""
    return np.sqrt(mu / a)


def delta_v_hohmann_one_way(a1, a2, mu):
    """
    단방향 Hohmann 전이의 ΔV를 계산한다.
    a1 → a2 방향 한 번의 번인 기동 ΔV.

    원궤도 → 천이궤도 진입 또는
    천이궤도 → 원궤도 도착 중 하나의 기동.

    여기서는 두 번의 burn을 각각 따로 계산하지 않고,
    논문의 단순화된 모델대로 원궤도 간 속도 차이로 근사한다.

    실제로는 논문이 구체적인 ΔV 계산식을 명시하지 않으므로
    단순 원궤도 속도 차이를 사용한다.
    (논문은 "thrust to obtain change in altitude" 만 언급)
    """
    v1 = circular_velocity(a1, mu)
    v2 = circular_velocity(a2, mu)
    return abs(v2 - v1)


def delta_v_altitude_change(h1_km, h2_km, params):
    """
    두 원형 궤도 사이 고도 변경에 필요한 총 ΔV를 계산한다.
    (지수대기모델은 사용하지 않음.)

    a1 → (전이나선) → a2

    Parameters
    ----------
    h1_km, h2_km : 고도 [km]
    params       : dict

    Returns
    -------
    dv : float [m/s]
    """
    mu = params['mu']
    Re = params['Re']

    a1 = Re + h1_km * 1e3
    a2 = Re + h2_km * 1e3
    a_transfer = (a1 + a2) / 2.0    # 전이 타원 반장축

    # 출발 원궤도 속도
    v_circ1 = np.sqrt(mu / a1)
    # # 전이 타원에서 출발 지점 속도 (원지점 또는 근지점)
    # v_trans1 = np.sqrt(mu * (2.0 / a1 - 1.0 / a_transfer))

    # 도착 원궤도 속도
    v_circ2 = np.sqrt(mu / a2)
    # # 전이 타원에서 도착 지점 속도
    # v_trans2 = np.sqrt(mu * (2.0 / a2 - 1.0 / a_transfer))

    # dv1 = abs(v_trans1 - v_circ1)
    # dv2 = abs(v_circ2 - v_trans2)

    # return dv1 + dv2

    return abs(v_circ2 - v_circ1)


def tof_tsiolkovsky(h1_km, h2_km, m_before_kg, params):
    """
    저추력(low-thrust) 연속 연소로 고도 h1 → h2 로 이동하는 데 걸리는 시간 [s].

    저추력 ΔV 누적 식 (질량이 시간에 따라 감소함을 정확히 반영) :

        ΔV = ∫_0^{T_leg} T_max / m(t) dt,    where m(t) = m_before - ṁ·t

    적분을 풀면 Tsiolkovsky 방정식 :

        ΔV = Isp·ge·ln(m_before / (m_before - ṁ·T_leg))

    T_leg 에 대해 풀면 해석해 :

        T_leg = (m_before · Isp · ge / T_max) · (1 - exp(-ΔV/(Isp·ge)))

    여기서 ΔV 는 고도 변경에 필요한 양으로, 두-burn Hohmann ΔV 를 사용한다.
    (논문은 ΔV 산출 방식을 명시하지 않으나 작은 고도 차이에서 Hohmann ΔV 는
    Edelbaum 등 다른 분석식과 거의 동일한 값을 준다.)

    --- 왜 운동방정식 적분이 아닌가 ---
    원칙적으로 가장 정확한 TOF 산출은 논문 식 (1)~(10) 의 운동방정식을
    수치적분(scipy.integrate.solve_ivp)하여 chaser 가 목표 고도에 도달하는
    시점을 직접 측정하는 방법이다. 하지만 다음 이유로 채택하지 않는다.

      1. 속도 :
         저추력 SEP 의 가속도가 ~5×10⁻⁵ m/s² 로 매우 작아 ODE step 수가
         폭증한다 (T1 한 leg 만 16,620 step, 4 초 소요).
         학습데이터 89,700 transfer 생성 시 :
           - rtol=1e-6 : 약 233 일
           - rtol=1e-3 : 약 56 일
         해석식 (현재 방식) 으로는 약 3 분 → 약 1,200×~5,100× 차이.
         scripts/benchmark_tof.py 참고.

      2. 정확도 트레이드오프 :
         해석식은 J2/drag 섭동을 무시하지만, 저추력 leg 동안 누적되는
         섭동 효과는 ANN 학습용 추진제/시간 추정에는 충분히 작다.
         RAAN drift 만 별도로 식 (22) 시간 적분으로 정확히 계산한다
         (integrate_raan_thrust_leg 참고).

      3. 학습데이터 검증 시 :
         일부 샘플에 대해 운동방정식 적분 도구 (scripts/benchmark_tof.py 의
         tof_numerical) 로 비교 검증하는 것이 실용적이다.

    Parameters
    ----------
    h1_km, h2_km : 고도 [km]
    m_before_kg  : 연소 시작 시 시스템 질량 [kg]
                   (T1 : chaser+D1 합산, T2a/T2b : chaser 단독)
    params       : dict (T_max, Isp, ge 포함)

    Returns
    -------
    tof : float [s]
    """
    dv      = delta_v_altitude_change(h1_km, h2_km, params)
    T_max   = params['T_max']
    Isp     = params['Isp']
    ge      = params['ge']
    # Tsiolkovsky 해석해
    return (m_before_kg * Isp * ge / T_max) * (1.0 - np.exp(-dv / (Isp * ge)))


# ────────────────────────────────────────────────
# 1-c. 식 (10) 기반 추진제 / ΔV 계산
# ────────────────────────────────────────────────
#
# 논문 식 (10) :   ṁ = -T_max · |N| / (Isp · ge)
#
# 추력 항상 ON (eclipse 무시) + |N| = 1 가정하면 ṁ 가 일정하므로
# 시간 적분이 단순 곱:
#     Δm = T_max / (Isp · ge) · T_leg
#
# Δm 으로부터 실제 ΔV 는 Tsiolkovsky 역으로 :
#     ΔV = Isp · ge · ln(m_before / m_after)

def propellant_consumed_eq10(T_leg, params):
    """
    식 (10) 적분으로 추력 구간 동안 소비되는 추진제 질량 Δm 을 계산.

    Δm = T_max / (Isp · ge) · T_leg     (eclipse 무시, 추력 항상 ON)

    Parameters
    ----------
    T_leg  : float [s]    구간 시간 (추력 ON 시간)
    params : dict         T_max, Isp, ge 포함

    Returns
    -------
    dm : float [kg]   소비된 추진제 질량 (양수)
    """
    if T_leg <= 0:
        return 0.0
    T_max     = params['T_max']
    Isp       = params['Isp']
    ge        = params['ge']
    m_dot_abs = T_max / (Isp * ge)   # |ṁ| [kg/s]
    return m_dot_abs * T_leg


def delta_v_from_mass(m_before, dm, params):
    """
    Tsiolkovsky 역으로 ΔV 를 질량 변화량에서 계산.
        ΔV = Isp · ge · ln(m_before / (m_before - dm))

    Parameters
    ----------
    m_before : float [kg]    번 직전 질량
    dm       : float [kg]    소비된 추진제 (양수)
    params   : dict

    Returns
    -------
    dv : float [m/s]
    """
    if dm <= 0:
        return 0.0
    Isp = params['Isp']
    ge  = params['ge']
    m_after = m_before - dm
    if m_after <= 0:
        # 물리적으로 불가능 — 추진제 부족
        return np.inf
    return Isp * ge * np.log(m_before / m_after)


def mass_after_burn(m_before, dv, params):
    """
    Tsiolkovsky 로켓 방정식으로 번 후 질량 계산.

    m_after = m_before * exp(-ΔV / (Isp * ge))

    Parameters
    ----------
    m_before : float [kg]
    dv       : float [m/s]
    params   : dict

    Returns
    -------
    m_after  : float [kg]
    """
    Isp = params['Isp']
    ge  = params['ge']
    return m_before * np.exp(-dv / (Isp * ge))


def propellant_mass(m_before, dv, params):
    """소비 추진제 질량 [kg]"""
    return m_before - mass_after_burn(m_before, dv, params)


# ────────────────────────────────────────────────
# 2. RAAN phasing 계산
# ────────────────────────────────────────────────

def compute_phasing_time(delta_RAAN_rad, omega_dot_P, omega_dot_D2):
    """
    RAAN phasing 에 필요한 시간을 계산한다 (논문 식 25).

        Tp = ΔΩ_P / (Ω̇_P - Ω̇_D2)

    --- 분모 부호로 추격/지연 전략을 자동 결정 ---
    chaser 와 D2 는 둘 다 동일 경사각 (i = 87.9°) 에서 RAAN 이 음의 방향으로
    drift 하지만 속도가 다르다. |Ω̇| ∝ a^(-7/2) 이므로 :

      h_P < h_D2  →  |Ω̇_P| > |Ω̇_D2|  →  분모 < 0  →  "추격" 전략 (catch-up)
      h_P > h_D2  →  |Ω̇_P| < |Ω̇_D2|  →  분모 > 0  →  "지연" 전략 (wait-up)
      h_P = h_D2  →  분모 ≈ 0          →  phasing 불가 (특이점)

    만남 조건은 Ω_D2(tf) ≡ Ω_SC(tf) (mod 2π) 이므로 분자 ΔΩ_P 는 임의의
    정수 회전 k 만큼 이동시켜도 같은 만남을 표현한다 :

        Tp(k) = (ΔΩ_P + 2π·k) / (Ω̇_P - Ω̇_D2),   k ∈ ℤ

    이 중 자연스러운 (가장 짧은) 양의 Tp 를 얻으려면 **분자를 분모와 같은
    부호로 정규화** 하면 된다 :

      분모 > 0 (지연) →  ΔΩ_P 를 [0, 2π) 로 정규화
      분모 < 0 (추격) →  ΔΩ_P 를 (-2π, 0] 로 정규화

    이 정규화는 추격이든 지연이든 **항상 양수 Tp** 를 보장한다.
    (이전 ±2π 분기 로직 / wrap [-π, π] 로직을 모두 대체.)

    Parameters
    ----------
    delta_RAAN_rad : float [rad]   필요한 RAAN 변화량 ΔΩ_P (wrap 되지 않은 원시값)
    omega_dot_P    : float [rad/s] phasing orbit RAAN drift rate
    omega_dot_D2   : float [rad/s] D2 orbit RAAN drift rate

    Returns
    -------
    Tp : float [s]  (분모 0 인 특이점에서만 np.inf)
    """
    denom = omega_dot_P - omega_dot_D2
    if abs(denom) < 1e-20:
        return np.inf
    # 분자를 분모 부호와 같게 정규화 → 항상 양수 Tp
    if denom > 0:
        # 지연 전략: 분자 ∈ [0, 2π)
        num = delta_RAAN_rad % (2 * np.pi)
    else:
        # 추격 전략: 분자 ∈ (-2π, 0]
        num = delta_RAAN_rad % (2 * np.pi) - 2 * np.pi
    return num / denom


def compute_delta_RAAN_needed(
    RAAN_D1_0, RAAN_D2_0,
    dRAAN_T1, dRAAN_T2a, dRAAN_T2b,
    omega_dot_D2,
    T1, T2a, T2b
):
    """
    phasing orbit 에서 메워야 할 RAAN 변화량 ΔΩ_P 를 계산한다.
    논문 식 (23), (24), (25) 기반.

    chaser 가 D2 에 도달하는 시각 tf 에서 Ω_SC,f = Ω_D2,f 가 되어야 한다.

    각 추력 구간(T1, T2a, T2b) 의 RAAN drift 는 a(t) 가 변하므로
    **시간 적분으로 미리 계산**한 ΔΩ 값을 그대로 받는다
    (raan_drift_integration.py / integrate_raan_thrust_leg 와 동일 알고리즘).

        Ω_SC,f = Ω_D1,0 + ΔΩ_T1 + ΔΩ_T2a + Ω̇_P·Tp + ΔΩ_T2b           (식 24, 적분형)
        Ω_D2,f = Ω_D2,0 + Ω̇_D2 · (T1 + T2a + Tp + T2b)                (식 23)

    Ω_SC,f = Ω_D2,f 조건으로 Tp 를 풀면 :
        ΔΩ_P = [Ω_D2,0 + Ω̇_D2·(T1 + T2a + T2b)]
              - [Ω_D1,0 + ΔΩ_T1 + ΔΩ_T2a + ΔΩ_T2b]
        Tp   = ΔΩ_P / (Ω̇_P - Ω̇_D2)                                    (식 25)

    Parameters
    ----------
    RAAN_D1_0, RAAN_D2_0 : 초기 RAAN [rad]
    dRAAN_T1, dRAAN_T2a, dRAAN_T2b : 각 추력 구간의 적분 ΔΩ [rad]
        (integrate_raan_thrust_leg 출력)
    omega_dot_D2 : D2 의 RAAN drift rate [rad/s]
    T1, T2a, T2b : 각 추력 구간 시간 [s]

    Returns
    -------
    delta_RAAN : float [rad]   원시 누적값 (wrap 없음).
                 정규화는 호출자(compute_phasing_time) 가 분모 부호에 맞춰 수행.
    """
    # T2b 끝(= phasing 미포함) 시점에서의 SC 와 D2 누적 RAAN
    RAAN_SC_excl_Tp = RAAN_D1_0 + dRAAN_T1 + dRAAN_T2a + dRAAN_T2b
    RAAN_D2_full    = RAAN_D2_0 + omega_dot_D2 * (T1 + T2a + T2b)

    # phasing 에서 메워야 할 차이 : "D2 가 SC 보다 얼마나 앞서있는지"
    # wrap 하지 않은 원시 누적값을 반환 — compute_phasing_time 이 분모 부호에 따라
    # [0, 2π) 또는 (-2π, 0] 로 정규화하여 항상 양수 Tp 를 보장한다.
    delta_RAAN = RAAN_D2_full - RAAN_SC_excl_Tp
    return delta_RAAN


# ────────────────────────────────────────────────
# 3. phasing orbit 고도 최적화 (논문 Section 2.2.2)
# ────────────────────────────────────────────────

def evaluate_phasing_orbit(
    h_P_km,
    h_D1_km, RAAN_D1_0, m_D1,
    h_disp_km,
    h_D2_km, RAAN_D2_0,
    m_SC_start,      # chaser 초기 질량 (T1 시작 전, D1 도킹 전)
    params, alpha,
    i_rad            # 경사각 [rad]
):
    """
    주어진 phasing orbit 고도 h_P에서 목적함수 J를 계산한다 (논문 식 28).

    J = α * ΔV_PT + (1-α) * T_PT

    ΔV_PT = ΔV_P + ΔV_{d→p} + ΔV_{p→D2}   (식 26)
    T_PT  = T_P  + T_{d→p} + T_{p→D2}      (식 27)

    phasing orbit 최소 고도 = disposal 고도 (논문 명시)

    Parameters
    ----------
    h_P_km    : phasing orbit 고도 [km] (최적화 변수)
    h_D1_km   : D1 고도 [km]
    RAAN_D1_0 : D1 초기 RAAN [rad]
    m_D1      : D1 질량 [kg]
    h_disp_km : disposal 고도 [km]
    h_D2_km   : D2 고도 [km]
    RAAN_D2_0 : D2 초기 RAAN [rad]
    m_SC_start: chaser 초기 질량 [kg] (T1 시작 시점, D1 docked 전)
    params    : dict
    alpha     : float ∈ [0,1]
    i_rad     : 경사각 [rad]

    Returns
    -------
    J         : float, 목적함수 값
    result    : dict, 상세 결과
    """
    mu = params['mu']
    Re = params['Re']

    # phasing orbit은 disposal 고도 이상이어야 함
    h_P_km = max(h_P_km, h_disp_km)

    # 반장축
    a_D1   = Re + h_D1_km   * 1e3
    a_disp = Re + h_disp_km * 1e3
    a_P    = Re + h_P_km    * 1e3
    a_D2   = Re + h_D2_km   * 1e3

    # --- T1: D1 → disposal (고도 하강) ---
    # T_leg 는 Tsiolkovsky 해석해 (chaser+D1 합산 질량 기준)
    m_total_T1 = m_SC_start + m_D1
    tof_T1     = tof_tsiolkovsky(h_D1_km, h_disp_km, m_total_T1, params)
    # 추진제 소비는 식 (10) 적분
    dm_T1      = propellant_consumed_eq10(tof_T1, params)
    # T1 후 chaser 단독 질량 (D1 방출, chaser 가 소비한 추진제만 차감)
    m_after_T1 = m_SC_start - dm_T1

    # T1 동안 RAAN drift (시간 적분: a(t) 선형 보간 + Ω̇(a(t)) 사다리꼴 적분)
    dRAAN_T1 = integrate_raan_thrust_leg(
        a_D1, a_disp, tof_T1, 0.0, i_rad, params
    )

    # --- T2,a: disposal → phasing orbit (고도 변경) ---
    if abs(h_P_km - h_disp_km) < 0.1:
        # phasing orbit = disposal orbit → T2,a = 0
        tof_T2a = 0.0
        dm_T2a  = 0.0
    else:
        tof_T2a = tof_tsiolkovsky(h_disp_km, h_P_km, m_after_T1, params)
        dm_T2a  = propellant_consumed_eq10(tof_T2a, params)
    m_after_T2a = m_after_T1 - dm_T2a   # T2a 추진제 소비 후 chaser 질량
    # dv_T2a 는 추진제 계산 후(질량을 안 뒤) Tsiolkovsky 역으로 계산

    # T2,a 동안 RAAN drift (시간 적분)
    dRAAN_T2a = integrate_raan_thrust_leg(
        a_disp, a_P, tof_T2a, 0.0, i_rad, params
    )

    # --- T2,b: phasing orbit → D2 orbit ---
    # (RAAN phasing 식이 T2b 항을 포함하므로 Tp 보다 먼저 계산해야 함)
    # m_before 는 T2a 끝 + drag 보상으로 줄어든 질량인데, drag 보상이 Tp 에
    # 의존하므로 미리 모름. 근사로 m_after_T2a (drag 보상 전) 사용 — drag 보상은
    # 보통 추진제 소비량의 일부에 불과해 영향이 작다.
    tof_T2b = tof_tsiolkovsky(h_P_km, h_D2_km, m_after_T2a, params)
    dm_T2b  = propellant_consumed_eq10(tof_T2b, params)

    # T2,b 동안 RAAN drift (시간 적분)
    dRAAN_T2b = integrate_raan_thrust_leg(
        a_P, a_D2, tof_T2b, 0.0, i_rad, params
    )

    # --- 추력 OFF 구간(phasing, D2)의 일정한 Ω̇ ---
    omega_dot_P    = raan_drift_rate(a_P,    0.0, i_rad, params)
    omega_dot_D2   = raan_drift_rate(a_D2,   0.0, i_rad, params)
    omega_dot_disp = raan_drift_rate(a_disp, 0.0, i_rad, params)

    # phasing 에서 메워야 할 RAAN 차이 계산
    # (각 추력 구간 ΔΩ 는 적분으로 미리 계산했으므로 그대로 넘김)
    delta_RAAN = compute_delta_RAAN_needed(
        RAAN_D1_0, RAAN_D2_0,
        dRAAN_T1, dRAAN_T2a, dRAAN_T2b,
        omega_dot_D2,
        tof_T1, tof_T2a, tof_T2b
    )

    # --- Tp: phasing 시간 (식 25) ---
    # compute_phasing_time 이 분모 부호에 따라 분자를 정규화하여 항상 양수 Tp 를 반환.
    # 분모 ≈ 0 (h_P = h_D2 특이점) 인 경우에만 np.inf.
    Tp = compute_phasing_time(delta_RAAN, omega_dot_P, omega_dot_D2)

    if not np.isfinite(Tp):
        return np.inf, {}

    # --- 추진제 / ΔV / 질량 계산 (식 10 기반, chronological 순서) ---
    #
    # 시간 순서 : T1 → T2a → Tp(drag) → T2b → Ts
    #   - T2a : m_after_T1 → m_after_T2a   (이미 위에서 계산)
    #   - Tp  : m_after_T2a → m_after_drag (대기항력 보상)
    #   - T2b : m_after_drag → m_final     (chaser 추력 상승)
    #
    # 주의 : tof_T2b 는 위에서 m_after_T2a 를 m_before 로 가정해 미리 산출됐다.
    # 엄밀하게는 T2b 의 m_before 가 m_after_drag (drag 소비 후 질량) 이어야
    # 하지만, drag 가 T2b 의 RAAN 적분에 영향을 주지 않고 Tp 와 tof_T2b 가
    # 결합돼 있어 미리 dm_drag 를 모른다. m_after_drag ≈ m_after_T2a 의
    # 1차 근사를 사용 — 두 질량의 차이는 보통 dm_drag (≪ 1 kg) 정도이므로
    # tof_T2b / dm_T2b 의 변동은 0.1% 미만이다.

    dv_T2a       = delta_v_from_mass(m_after_T1, dm_T2a, params)

    # Tp 중 drag 보상 (chronological 첫 번째) — 식 (26) 의 ΔV_P
    # drag_dv_circular 는 가속도 |a_D|(m_after_T2a) · Tp 를 그대로 반환하므로
    # mass 의존성이 이미 m_after_T2a (Tp 시작 시점 질량) 에 묶여 있다.
    dv_drag_P    = drag_dv_circular(h_P_km, Tp, m_after_T2a, params)
    m_after_drag = mass_after_burn(m_after_T2a, dv_drag_P, params)

    # T2b 추력 (chronological 두 번째) — m_after_drag 를 m_before 로 사용
    m_final      = m_after_drag - dm_T2b
    dv_T2b       = delta_v_from_mass(m_after_drag, dm_T2b, params)

    # --- 목적함수 값 계산 (식 26~28) ---
    dv_PT  = dv_T2a + dv_T2b + dv_drag_P           # 식 (26)
    tof_PT = tof_T2a + Tp + tof_T2b                 # 식 (27)

    # 정규화 (스케일 맞추기: ΔV [m/s], TOF [days])
    # 논문 식 (28): J = α * ΔV_PT + (1-α) * T_PT
    # 단위가 다르므로 정규화 필요 → 각 기준값으로 나눔
    dv_ref  = 1000.0          # [m/s] 기준값
    tof_ref = 365.0 * 86400.0 # [s] 기준값 (1년)

    J = alpha * (dv_PT / dv_ref) + (1 - alpha) * (tof_PT / tof_ref)

    result = {
        'h_P_km'    : h_P_km,
        'dm_T1'     : dm_T1,         # 식 (10) 적분 추진제 소비 [kg], dv_T1 은 호출자가 계산
        'tof_T1'    : tof_T1,
        'dv_T2a'    : dv_T2a,
        'tof_T2a'   : tof_T2a,
        'Tp'        : Tp,
        'dv_T2b'    : dv_T2b,
        'tof_T2b'   : tof_T2b,
        'dv_drag_P' : dv_drag_P,
        'dv_PT'     : dv_PT,
        'tof_PT'    : tof_PT,
        'delta_RAAN': delta_RAAN,
        # chronological 시점별 chaser 질량 [kg] :
        #   m_after_T2a  = T2a 끝 = Tp 시작
        #   m_after_drag = Tp(drag) 끝 = T2b 시작
        #   m_final      = T2b 끝 = Ts 시작
        'm_after_T2a' : m_after_T2a,
        'm_after_drag': m_after_drag,
        'm_final'     : m_final,
        'J'           : J,
    }
    return J, result


def drag_dv_circular(h_km, T_sec, m, params):
    """
    원궤도에서 일정 시간 동안 누적되는 대기항력 ΔV를 계산한다.
    논문 식 (17)~(21) 의 modified equinoctial 형식 그대로.

    Tp (phasing 대기) 와 Ts (D2 stay) 양쪽 모두에서 동일한 수학으로 호출된다.
    chaser 가 추력으로 항력을 상쇄해 고도를 유지한다고 가정하고, 그에
    필요한 임펄스 ΔV 를 a_drag · T 로 산출한다.

    --- 64점 격자 → 단일점 단축 원리 ---
    이전 구현은 L ∈ [0, 2π] 를 64 점 격자로 스윕하여 각 점마다
    keplerian_to_mee + drag_acceleration_lvlh 를 호출 (= 64 회) 하고
    |a_D| 의 한 바퀴 평균을 취했다.

    그러나 원궤도 (e = 0) 에서 modified equinoctial 요소가 :
        f = e · cos(AOP+RAAN) = 0,   g = e · sin(AOP+RAAN) = 0
    이 되어 식 (20), (21) 이 :
        v_r     = √(μ/p) · (f sin L − g cos L) = 0
        v_θ     = √(μ/p) · (1 + f cos L + g sin L) = √(μ/p)
    로 단순화된다. 즉 v_r, v_θ, |v| 모두 L 에 무관한 상수.
    또한 r = p/q 에서 q = 1 → r = p (일정), 따라서 ρ(r-Re) 도 일정.
    그러므로 식 (18), (19) 의 |a_D| 는 한 바퀴 동안 완전히 일정 →
    64 점 평균 = 1 점 평가가 수학적으로 정확히 동일하다.

    측정상 |a_D| 의 64 점 분산 ≈ 0 (부동소수점 오차 한계 내),
    1 점 평가 결과는 이전 64 점 평균과 비율 1.000000 으로 일치.

    --- 향후 일반화 시 ---
    e ≠ 0 궤도에 적용하게 되면 이 단축 가정이 깨지므로 L ∈ [0, 2π]
    격자 적분 (이전 구현) 으로 되돌려야 한다.

    Parameters
    ----------
    h_km    : 원궤도 고도 [km]
    T_sec   : 누적 시간 [s]
    m       : chaser 질량 [kg]    (a_drag ∝ 1/m 이므로 시점별 질량 사용)
    params  : dict   (mu, Re, CD, S, inclination_deg 포함)

    Returns
    -------
    dv_drag : float [m/s]
    """
    from src.dynamics import keplerian_to_mee, drag_acceleration_lvlh

    if T_sec <= 0:
        return 0.0

    mu    = params['mu']
    Re    = params['Re']
    CD    = params['CD']
    S     = params['S']
    i_rad = np.deg2rad(params['inclination_deg'])

    a_circ = Re + h_km * 1e3   # 반장축 [m]

    # e=0 원궤도 → L 에 무관하게 |a_D| 일정 → 단일 점 (L=0) 에서 계산
    mee = keplerian_to_mee(a=a_circ, e=0.0, i=i_rad,
                           RAAN=0.0, AOP=0.0, nu=0.0)
    a_D = drag_acceleration_lvlh(mee, mu, CD, S, m, Re)   # [a_Dr, a_Dθ, 0]
    a_drag_mag = float(np.linalg.norm(a_D))

    # 시간 T 동안 누적 (drag 가속도가 거의 일정)
    return a_drag_mag * T_sec


def optimize_phasing_orbit(
    h_D1_km, RAAN_D1_0, m_D1,
    h_disp_km,
    h_D2_km, RAAN_D2_0,
    m_SC_start,
    params, alpha, i_rad
):
    """
    phasing orbit 고도 h_P를 grid search 로 최적화한다 (논문 Section 2.2.2).

    목적함수: J = α * ΔV_PT + (1-α) * T_PT  (식 28)
    탐색 범위: h_P ∈ [params['h_P_min_km'], params['h_P_max_km']],
              h_step_km 간격 grid (simulation.yaml 의 search 섹션).
              단 하한은 disposal 고도로 clamp — phasing orbit 은 disposal
              보다 낮을 수 없음.

    --- 왜 Brent 가 아니라 grid search 인가 ---
    J(h_P) 는 h_P = h_D2 에서 식 (25) 의 분모 Ω̇_P − Ω̇_D2 가 0 이 되어
    극(pole) 을 가지며, 그 양쪽으로 단절된 양봉 구조를 보인다 :
      - 추격 가지 (h_P < h_D2) : h_min 근방에 진짜 최소가 자주 위치
      - 대기 가지 (h_P > h_D2) : 상한 근방에 가짜 국소 최소가 자주 발생
    Brent (`scipy.minimize_scalar(method='bounded')`) 는 단봉 가정을 깔고
    동작하므로, 황금분할 초기 샘플이 우연히 한 가지 안에 모두 떨어지면
    반대 가지의 진짜 최소를 영영 못 본다. 실제로 debris0001(500km, Ω=100°)
    → debris0002(800km, Ω=95°) 케이스에서 Brent 가 h_P=2000km(상한)을
    반환해 TOF ~3700 일이 나오는 버그가 확인됐다 (진짜 최적은
    h_P=390km, TOF 149 일).

    grid search 는 양 가지를 모두 평가하므로 이 함정에 빠지지 않는다.
    transfer_grid.compute_transfer_grid 와 동일한 grid 를 사용해 두 경로의
    결과 일관성을 보장한다.

    Parameters
    ----------
    h_D1_km    : D1 고도 [km]
    RAAN_D1_0  : D1 초기 RAAN [rad]
    m_D1       : D1 질량 [kg]
    h_disp_km  : disposal 고도 [km]
    h_D2_km    : D2 고도 [km]
    RAAN_D2_0  : D2 초기 RAAN [rad]
    m_SC_start : chaser 초기 질량 [kg]
    params     : dict  (h_P_min_km, h_P_max_km, h_step_km 포함)
    alpha      : float ∈ [0,1]
    i_rad      : 경사각 [rad]

    Returns
    -------
    best_result : dict  최적 phasing orbit 의 evaluate_phasing_orbit 결과
    """
    # 탐색 범위 (simulation.yaml 의 search 섹션)
    # 하한은 disposal 고도와 yaml 설정 중 더 큰 값으로 clamp.
    h_min  = max(h_disp_km, params['h_P_min_km'])
    h_max  = params['h_P_max_km']
    h_step = params['h_step_km']
    if h_max <= h_min:
        raise ValueError(
            f"h_P_max_km ({h_max:.1f}) 이 h_P_min_km/disposal_alt "
            f"({h_min:.1f}) 이하입니다. simulation.yaml 의 search 섹션을 확인하세요."
        )

    # transfer_grid.compute_transfer_grid 와 동일한 grid 생성
    # (np.arange 의 우개구간 보정을 위해 step/2 여유)
    h_P_grid = np.arange(h_min, h_max + h_step / 2, h_step)

    best_J   = np.inf
    best_res = None
    for h_P in h_P_grid:
        J, res = evaluate_phasing_orbit(
            h_P_km     = float(h_P),
            h_D1_km    = h_D1_km,
            RAAN_D1_0  = RAAN_D1_0,
            m_D1       = m_D1,
            h_disp_km  = h_disp_km,
            h_D2_km    = h_D2_km,
            RAAN_D2_0  = RAAN_D2_0,
            m_SC_start = m_SC_start,
            params     = params,
            alpha      = alpha,
            i_rad      = i_rad,
        )
        if (res is None) or (not np.isfinite(J)):
            continue
        if J < best_J:
            best_J   = J
            best_res = res

    if best_res is None:
        raise RuntimeError(
            f"phasing orbit 을 찾을 수 없습니다 (h_P_grid 전체에서 J=inf). "
            f"D1/D2 조합이 phasing 불가능한지, h_P 탐색 범위가 충분히 넓은지 "
            f"(simulation.yaml 의 search 섹션) 확인하세요."
        )

    return best_res


# ────────────────────────────────────────────────
# 4. 전체 transfer 계산 (논문 Section 2.2)
# ────────────────────────────────────────────────

def solve_transfer(debris1, debris2, m_SC, params, alpha):
    """
    D1 → disposal → phasing → D2 전체 전이를 계산한다.

    Parameters
    ----------
    debris1 : dict  {'alt0_km', 'RAAN', 'mass', 'i', ...}
    debris2 : dict
    m_SC    : float [kg]  chaser 초기 질량 (추진제 포함)
    params  : dict
    alpha   : float ∈ [0,1]  목적함수 가중치

    Returns
    -------
    result : dict
        {
          'T1'       : T1 시간 [s]
          'T2a'      : T2,a 시간 [s]
          'Tp'       : Tp 시간 [s]
          'T2b'      : T2,b 시간 [s]
          'Ts'       : Ts 시간 [s]  (30일 고정)
          'TOF'      : 총 비행 시간 [s]
          'dv_T1'    : T1 ΔV [m/s]
          'dv_T2'    : T2 ΔV [m/s]
          'dv_drag'  : phasing drag ΔV [m/s]
          'm_prop'   : 총 추진제 소비 [kg]
          'h_P_km'   : 최적 phasing orbit 고도 [km]
          'phase'    : 각 구간 상세 dict
        }
    """
    Re      = params['Re']
    i_deg   = params['inclination_deg']
    i_rad   = np.deg2rad(i_deg)
    h_disp  = params['disposal_alt_km']
    Ts      = 30.0 * 86400.0   # 30일 고정 (논문)

    h_D1   = debris1['alt0_km']
    RAAN_D1 = np.deg2rad(debris1['RAAN'])
    m_D1   = debris1['mass']

    h_D2   = debris2['alt0_km']
    RAAN_D2 = np.deg2rad(debris2['RAAN'])

    # ── Phase T1: D1 orbit → disposal orbit ──
    # chaser + D1 합산 질량으로 고도 하강.
    # T_leg 는 Tsiolkovsky 해석해, 추진제는 식 (10) 적분으로 산출.
    m_total_T1       = m_SC + m_D1
    tof_T1           = tof_tsiolkovsky(h_D1, h_disp, m_total_T1, params)
    m_prop_T1        = propellant_consumed_eq10(tof_T1, params)   # 식 (10) 적분
    m_total_after_T1 = m_total_T1 - m_prop_T1
    dv_T1            = delta_v_from_mass(m_total_T1, m_prop_T1, params)
    # D1 방출 → chaser 단독 질량 (chaser 가 소비한 추진제만큼만 차감)
    m_SC_after_T1    = m_SC - m_prop_T1

    # ── Phase T2,a + Tp + T2,b 최적화 ──
    phasing = optimize_phasing_orbit(
        h_D1_km    = h_D1,
        RAAN_D1_0  = RAAN_D1,
        m_D1       = m_D1,
        h_disp_km  = h_disp,
        h_D2_km    = h_D2,
        RAAN_D2_0  = RAAN_D2,
        m_SC_start = m_SC,
        params     = params,
        alpha      = alpha,
        i_rad      = i_rad
    )

    # 추진제 소비 합산 (chronological 순서: T2a → Tp(drag) → T2b → Ts(drag))
    m_prop_T2a   = m_SC_after_T1            - phasing['m_after_T2a']
    m_prop_drag  = phasing['m_after_T2a']   - phasing['m_after_drag']
    m_prop_T2b   = phasing['m_after_drag']  - phasing['m_final']

    # Ts (D2 stay) 동안 대기항력 보상 — D2 고도에서 Ts 동안 누적 ΔV 를
    # chaser 추력으로 상쇄하여 고도 유지. 고도 800 km 같은 높은 곳은 거의
    # 무시할 수준 (~0.001 kg), 500 km 근방에서는 ~0.05 kg 정도.
    dv_Ts_drag    = drag_dv_circular(h_D2, Ts, phasing['m_final'], params)
    m_SC_end      = mass_after_burn(phasing['m_final'], dv_Ts_drag, params)
    m_prop_Ts     = phasing['m_final'] - m_SC_end

    m_prop_total = (m_prop_T1 + m_prop_T2a + m_prop_drag
                    + m_prop_T2b + m_prop_Ts)

    # 총 비행 시간
    TOF = tof_T1 + phasing['tof_T2a'] + phasing['Tp'] + phasing['tof_T2b'] + Ts

    return {
        # 구간별 시간 [s]
        'T1'      : tof_T1,
        'T2a'     : phasing['tof_T2a'],
        'Tp'      : phasing['Tp'],
        'T2b'     : phasing['tof_T2b'],
        'Ts'      : Ts,
        'TOF'     : TOF,
        # ΔV [m/s]
        'dv_T1'      : dv_T1,
        'dv_T2a'     : phasing['dv_T2a'],
        'dv_T2b'     : phasing['dv_T2b'],
        'dv_drag'    : phasing['dv_drag_P'],   # Tp drag 보상
        'dv_Ts_drag' : dv_Ts_drag,              # Ts drag 보상
        # 질량 [kg]
        'm_prop'      : m_prop_total,
        'm_SC_start'  : m_SC,
        'm_SC_end'    : m_SC_end,               # Ts drag 후 chaser 최종 질량
        # phasing 결과
        'h_P_km'  : phasing['h_P_km'],
        'delta_RAAN_rad': phasing['delta_RAAN'],
        # 구간별 상세
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