"""
dynamics.py
===========
논문 Section 2에 기술된 궤도 역학을 구현한다.
Modified Equinoctial Elements (MEE)를 상태 벡터로 사용하며,
J2 섭동과 대기항력을 고려한다.

MEE 정의:
  p  : semi-latus rectum [m]
  f  : eccentricity vector x-component (= e*cos(ω+Ω))
  g  : eccentricity vector y-component (= e*sin(ω+Ω))
  h  : inclination vector x-component  (= tan(i/2)*cos(Ω))
  k  : inclination vector y-component  (= tan(i/2)*sin(Ω))
  L  : true longitude [rad] (= Ω + ω + ν)
"""

import numpy as np


# ──────────────────────────────────────────────
# 1. 좌표 변환 유틸리티
# ──────────────────────────────────────────────

def keplerian_to_mee(a, e, i, RAAN, AOP, nu):
    """
    Classical Keplerian elements → Modified Equinoctial Elements 변환.

    Parameters
    ----------
    a    : 반장축 [m]
    e    : 이심률 [-]
    i    : 경사각 [rad]
    RAAN : 승교점 적경 [rad]
    AOP  : 근지점 인수 [rad]
    nu   : 진근점 이각 [rad]

    Returns
    -------
    mee : ndarray, shape (6,)  [p, f, g, h, k, L]
    """
    p = a * (1.0 - e**2)                    # semi-latus rectum [m]
    f = e * np.cos(AOP + RAAN)              # eccentricity vector x
    g = e * np.sin(AOP + RAAN)              # eccentricity vector y
    h = np.tan(i / 2.0) * np.cos(RAAN)     # inclination vector x
    k = np.tan(i / 2.0) * np.sin(RAAN)     # inclination vector y
    L = RAAN + AOP + nu                     # true longitude [rad]
    return np.array([p, f, g, h, k, L])


def mee_to_keplerian(mee):
    """
    Modified Equinoctial Elements → Classical Keplerian elements 변환.

    Parameters
    ----------
    mee : ndarray, shape (6,)  [p, f, g, h, k, L]

    Returns
    -------
    tuple : (a, e, i, RAAN, AOP, nu)  단위: [m, -, rad, rad, rad, rad]
    """
    p, f, g, h, k, L = mee

    e    = np.sqrt(f**2 + g**2)             # 이심률
    a    = p / (1.0 - e**2)                 # 반장축 [m]
    chi  = np.sqrt(h**2 + k**2)
    i    = 2.0 * np.arctan(chi)             # 경사각 [rad]
    RAAN = np.arctan2(k, h)                 # 승교점 적경 [rad]
    AOP  = np.arctan2(g * h - f * k, f * h + g * k)   # 근지점 인수 [rad]
    nu   = L - RAAN - AOP                   # 진근점 이각 [rad]

    return a, e, i, RAAN, AOP, nu


def mee_to_rv(mee, mu):
    """
    Modified Equinoctial Elements → 위치/속도 벡터 (ECI) 변환.

    Parameters
    ----------
    mee : ndarray, shape (6,)  [p, f, g, h, k, L]
    mu  : float, 중력 상수 [m^3/s^2]

    Returns
    -------
    r : ndarray, shape (3,)  위치 벡터 [m]
    v : ndarray, shape (3,)  속도 벡터 [m/s]
    """
    p, f, g, h, k, L = mee

    alpha2    = h**2 - k**2
    s2        = 1.0 + h**2 + k**2
    q         = 1.0 + f * np.cos(L) + g * np.sin(L)
    r_mag     = p / q
    sqrt_mu_p = np.sqrt(mu / p)

    # 위치 벡터 (ECI)
    r = (r_mag / s2) * np.array([
        np.cos(L) + alpha2 * np.cos(L) + 2.0 * h * k * np.sin(L),
        np.sin(L) - alpha2 * np.sin(L) + 2.0 * h * k * np.cos(L),
        2.0 * (h * np.sin(L) - k * np.cos(L))
    ])

    # 속도 벡터 (ECI)
    v = (sqrt_mu_p / s2) * np.array([
        -(np.sin(L) + alpha2 * np.sin(L) - 2.0 * h * k * np.cos(L)
          + g - 2.0 * f * h * k + alpha2 * g),
        -(-np.cos(L) + alpha2 * np.cos(L) + 2.0 * h * k * np.sin(L)
          - f + 2.0 * g * h * k + alpha2 * f),
        2.0 * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
    ])

    return r, v


# ──────────────────────────────────────────────
# 2. 가우스 변분 방정식 행렬 A(x), 벡터 b(x)
# ──────────────────────────────────────────────

def gauss_matrix_A(mee, mu):
    """
    논문 식 (2)~(3)의 가우스 변분 방정식 행렬 A(x)를 계산한다.
    LVLH 섭동 가속도 [a_r, a_theta, a_h] → MEE 변화율 매핑.

    Parameters
    ----------
    mee : ndarray, shape (6,)  [p, f, g, h, k, L]
    mu  : float

    Returns
    -------
    A : ndarray, shape (6, 3)
        행: [p_dot, f_dot, g_dot, h_dot, k_dot, L_dot] 기여분
        열: [radial, transversal, out-of-plane]
    """
    p, f, g, h, k, L = mee

    # 보조 변수 (논문 식 (5)~(7))
    q         = 1.0 + f * np.cos(L) + g * np.sin(L)   # 식 (5)
    s2        = 1.0 + h**2 + k**2                      # 식 (6)
    sqrt_p_mu = np.sqrt(p / mu)

    # 행렬 원소 (논문 식 (3a)~(3j))
    a12 = 2.0 * p / q * sqrt_p_mu                                          # 식 (3a)
    a21 = sqrt_p_mu * np.sin(L)                                             # 식 (3b)
    a22 = sqrt_p_mu / q * ((q + 1.0) * np.cos(L) + f)                      # 식 (3c)
    a23 = -sqrt_p_mu * g / q * (h * np.sin(L) - k * np.cos(L))            # 식 (3d)
    a31 = -sqrt_p_mu * np.cos(L)                                            # 식 (3e)
    a32 = sqrt_p_mu / q * ((q + 1.0) * np.sin(L) + g)                      # 식 (3f)
    a33 = sqrt_p_mu * f / q * (h * np.sin(L) - k * np.cos(L))             # 식 (3g)
    a43 = sqrt_p_mu * s2 / (2.0 * q) * np.cos(L)                          # 식 (3h)
    a53 = sqrt_p_mu * s2 / (2.0 * q) * np.sin(L)                          # 식 (3i)
    a63 = sqrt_p_mu / q * (h * np.sin(L) - k * np.cos(L))                 # 식 (3j)

    # 6×3 행렬 조립 (열 순서: radial, transversal, out-of-plane)
    A = np.array([
        [0.0, a12, 0.0],   # p
        [a21, a22, a23],   # f
        [a31, a32, a33],   # g
        [0.0, 0.0, a43],   # h
        [0.0, 0.0, a53],   # k
        [0.0, 0.0, a63],   # L (추력에 의한 추가 항)
    ])
    return A


def gauss_vector_b(mee, mu):
    """
    논문 식 (4)의 벡터 b(x)를 계산한다.
    추력이 없을 때 true longitude L의 자연 변화율(케플러 운동)을 나타낸다.

    Parameters
    ----------
    mee : ndarray, shape (6,)
    mu  : float

    Returns
    -------
    b : ndarray, shape (6,)
    """
    p, f, g, h, k, L = mee
    q = 1.0 + f * np.cos(L) + g * np.sin(L)

    b = np.zeros(6)
    # L 성분만 비영 (논문 식 (4))
    b[5] = np.sqrt(mu * p) * (q / p)**2
    return b


# ──────────────────────────────────────────────
# 3. 섭동 가속도
# ──────────────────────────────────────────────

def j2_acceleration_lvlh(mee, mu, Re, J2):
    """
    J2 섭동 가속도를 LVLH 좌표계 [a_r, a_theta, a_h]로 계산한다.
    논문 식 (11)~(16)에 해당.

    Parameters
    ----------
    mee : ndarray, shape (6,)
    mu  : float
    Re  : float, 지구 적도 반경 [m]
    J2  : float

    Returns
    -------
    a_J2 : ndarray, shape (3,)  [a_r, a_theta, a_h]
    """
    p, f, g, h, k, L = mee
    q     = 1.0 + f * np.cos(L) + g * np.sin(L)
    r_mag = p / q   # 궤도 반경 [m]

    # ECI 위치/속도 벡터
    r_vec, v_vec = mee_to_rv(mee, mu)

    # 지심 위도 φ
    sin_phi = r_vec[2] / r_mag

    # J2 ECI 가속도
    coeff    = 1.5 * J2 * mu * Re**2 / r_mag**4
    a_J2_eci = coeff * np.array([
        (r_vec[0] / r_mag) * (5.0 * sin_phi**2 - 1.0),
        (r_vec[1] / r_mag) * (5.0 * sin_phi**2 - 1.0),
        (r_vec[2] / r_mag) * (5.0 * sin_phi**2 - 3.0),
    ])

    # LVLH 기저 벡터
    ir  = r_vec / np.linalg.norm(r_vec)                                       # radial
    h_  = np.cross(r_vec, v_vec)
    ih  = h_ / np.linalg.norm(h_)                                             # out-of-plane
    it  = np.cross(ih, ir)                                                    # transversal

    # ECI → LVLH 투영
    return np.array([
        np.dot(a_J2_eci, ir),
        np.dot(a_J2_eci, it),
        np.dot(a_J2_eci, ih),
    ])


def drag_acceleration_lvlh(mee, mu, CD, S, m, Re):
    """
    대기항력 가속도를 LVLH 좌표계 [a_Dr, a_Dtheta, 0]로 계산한다.
    논문 식 (17)~(21)에 해당.
    지구 자전은 무시한다 (상대 속도 ≈ 관성 속도).

    Parameters
    ----------
    mee : ndarray, shape (6,)
    mu  : float
    CD  : float, 항력 계수
    S   : float, 공력 기준 면적 [m^2]
    m   : float, 현재 질량 [kg]
    Re  : float, 지구 반경 [m]

    Returns
    -------
    a_D : ndarray, shape (3,)  [a_Dr, a_Dtheta, 0]
    """
    p, f, g, h, k, L = mee
    q     = 1.0 + f * np.cos(L) + g * np.sin(L)
    r_mag = p / q

    # 고도 [m]
    alt = r_mag - Re

    # 지수 대기 밀도
    rho = exponential_atmosphere(alt)

    # LVLH 속도 성분 (논문 식 (20)~(21))
    sqrt_mu_p = np.sqrt(mu / p)
    v_r     = sqrt_mu_p * (f * np.sin(L) - g * np.cos(L))           # 식 (20)
    v_theta = sqrt_mu_p * (1.0 + f * np.cos(L) + g * np.sin(L))     # 식 (21)
    v_mag   = np.sqrt(v_r**2 + v_theta**2)

    # 단위 질량당 항력 계수
    D_coeff = 0.5 * rho * CD * S / m

    # LVLH 성분 (논문 식 (18)~(19))
    a_Dr     = -D_coeff * v_mag * v_r
    a_Dtheta = -D_coeff * v_mag * v_theta

    return np.array([a_Dr, a_Dtheta, 0.0])


def exponential_atmosphere(alt_m):
    """
    분층 지수 대기 모델: 고도에 따른 밀도 [kg/m^3] 반환.

    Parameters
    ----------
    alt_m : float, 고도 [m]

    Returns
    -------
    rho : float [kg/m^3]
    """
    alt_km = alt_m / 1000.0

    # (기준고도 [km], 기준밀도 [kg/m^3], 스케일 높이 [km])
    # 출처: Vallado, "Fundamentals of Astrodynamics and Applications" 4th ed.
    table = [
        (0,    1.225,       8.44),
        (25,   3.899e-2,    6.49),
        (30,   1.774e-2,    6.75),
        (40,   3.972e-3,    7.07),
        (50,   1.057e-3,    7.47),
        (60,   3.206e-4,    7.83),
        (70,   8.770e-5,    7.10),
        (80,   1.905e-5,    6.00),
        (90,   3.396e-6,    5.95),
        (100,  5.297e-7,    5.84),
        (110,  9.661e-8,    5.84),
        (120,  2.438e-8,    7.31),
        (130,  8.484e-9,    8.20),
        (140,  3.845e-9,    8.66),
        (150,  2.070e-9,    9.19),
        (180,  5.464e-10,   9.81),
        (200,  2.789e-10,  11.19),
        (250,  7.248e-11,  15.13),
        (300,  2.418e-11,  22.97),
        (350,  9.518e-12,  29.74),
        (400,  3.725e-12,  37.20),
        (450,  1.585e-12,  45.68),
        (500,  6.967e-13,  53.15),
        (600,  1.454e-13,  58.54),
        (700,  3.614e-14,  65.80),
        (800,  1.170e-14,  68.75),
        (900,  5.245e-15,  73.90),
        (1000, 3.019e-15,  79.94),
        (1500, 2.007e-16, 124.64),
    ]

    for idx in range(len(table) - 1):
        h0, rho0, H = table[idx]
        h1 = table[idx + 1][0]
        if h0 <= alt_km < h1:
            return rho0 * np.exp(-(alt_km - h0) / H)

    # 범위 초과 시 마지막 층 외삽
    h0, rho0, H = table[-1]
    return rho0 * np.exp(-(alt_km - h0) / H)


# ──────────────────────────────────────────────
# 4. 전체 상태 방정식
# ──────────────────────────────────────────────

def eom_chaser(t, state, thrust_func, params):
    """
    chaser 운동 방정식 (논문 식 (1), (8)~(10)).
    상태 벡터: [p, f, g, h, k, L, m] (MEE 6개 + 질량 1개 = 7개)

    추력 함수 thrust_func(t, state) → ndarray(3,)
    LVLH 방향의 추력 방향 벡터 N = [Nr, Ntheta, Nh]
    크기 |N| ∈ [0, 1] (throttle 포함)

    Parameters
    ----------
    t           : float, 현재 시각 [s]
    state       : ndarray, shape (7,)
    thrust_func : callable, thrust_func(t, state) → ndarray(3,)
    params      : dict, 물리 파라미터

    Returns
    -------
    dstate_dt : ndarray, shape (7,)
    """
    mee = state[:6]
    m   = state[6]

    mu    = params['mu']
    Re    = params['Re']
    J2    = params['J2']
    CD    = params['CD']
    S     = params['S']
    T_max = params['T_max']
    Isp   = params['Isp']
    ge    = params['ge']

    # 추력 방향 벡터 (throttle 포함)
    N     = thrust_func(t, state)        # shape (3,)
    N_mag = np.linalg.norm(N)            # throttle 크기

    # 추력 가속도 (논문 식 (9))
    a_T = (T_max / m) * N

    # J2 섭동 가속도 (LVLH)
    a_J2 = j2_acceleration_lvlh(mee, mu, Re, J2)

    # 대기항력 가속도 (LVLH)
    a_D = drag_acceleration_lvlh(mee, mu, CD, S, m, Re)

    # 전체 섭동 가속도 (논문 식 (8))
    a_total = a_T + a_J2 + a_D

    # 가우스 변분 방정식: MEE_dot = A(x)*a + b(x)
    A       = gauss_matrix_A(mee, mu)
    b       = gauss_vector_b(mee, mu)
    mee_dot = A @ a_total + b

    # 질량 변화율 (논문 식 (10))
    m_dot = -(T_max * N_mag) / (Isp * ge)

    return np.concatenate([mee_dot, [m_dot]])


def eom_debris(t, state, params):
    """
    debris 운동 방정식 (추력 없음).
    J2 섭동과 대기항력만 적용.

    Parameters
    ----------
    t      : float
    state  : ndarray, shape (6,)  MEE
    params : dict

    Returns
    -------
    mee_dot : ndarray, shape (6,)
    """
    mee = state
    mu  = params['mu']
    Re  = params['Re']
    J2  = params['J2']
    CD  = params['CD']
    S   = params['S']
    m   = params.get('m_debris', 200.0)   # 대표 질량 [kg]

    a_J2 = j2_acceleration_lvlh(mee, mu, Re, J2)
    a_D  = drag_acceleration_lvlh(mee, mu, CD, S, m, Re)

    A = gauss_matrix_A(mee, mu)
    b = gauss_vector_b(mee, mu)

    return A @ (a_J2 + a_D) + b
