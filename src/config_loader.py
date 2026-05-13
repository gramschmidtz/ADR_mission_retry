"""
config_loader.py
================
YAML 설정 파일을 로드하고 파라미터를 통합 관리한다.
"""

import yaml
import numpy as np
from pathlib import Path


def load_yaml(path):
    """YAML 파일을 dict로 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_params(constants_path='configs/constants.yaml',
                simulation_path='configs/simulation.yaml'):
    """
    constants.yaml과 simulation.yaml을 로드하여
    물리 파라미터 dict를 반환한다.

    Returns
    -------
    params : dict
        dynamics.py, transfer_solver.py에서 사용하는 물리 파라미터
    sim_cfg : dict
        시뮬레이션 설정
    """
    const = load_yaml(constants_path)
    sim   = load_yaml(simulation_path)

    # search 섹션 (선택 — yaml 에 없으면 기본값 사용)
    search_cfg = sim.get('search', {})

    # 물리 파라미터 통합
    params = {
        # 물리 상수
        'mu'   : const['physical']['mu'],
        'Re'   : const['physical']['Re'],
        'ge'   : const['physical']['ge'],
        'J2'   : const['physical']['J2'],
        'CD'   : const['physical']['CD'],
        'S'    : const['physical']['S'],
        # 우주선 파라미터
        'T_max'      : const['spacecraft']['T_max'],
        'Isp'        : const['spacecraft']['Isp'],
        'm0'         : const['spacecraft']['m0'],
        'm_prop_max' : const['spacecraft']['m_prop_max'],
        # 처분 궤도
        'disposal_alt_km' : const['orbit']['disposal_alt_km'],
        'inclination_deg' : const['orbit']['inclination_deg'],
        # 탐색 범위 (simulation.yaml 의 search 섹션 ; 없으면 기본값)
        'h_P_min_km'    : float(search_cfg.get('h_P_min_km',    300.0)),
        'h_P_max_km'    : float(search_cfg.get('h_P_max_km',   2000.0)),
        'h_step_km'     : float(search_cfg.get('h_step_km',     100.0)),
        'raan_min_deg'  : float(search_cfg.get('raan_min_deg',    0.0)),
        'raan_max_deg'  : float(search_cfg.get('raan_max_deg',  360.0)),
        'raan_step_deg' : float(search_cfg.get('raan_step_deg',   2.0)),
        'alpha_min'     : float(search_cfg.get('alpha_min',       0.0)),
        'alpha_max'     : float(search_cfg.get('alpha_max',       1.0)),
        'alpha_step'    : float(search_cfg.get('alpha_step',      0.05)),
    }

    return params, sim


def load_debris_list(debris_yaml_path):
    """
    debris.yaml을 로드하여 debris 목록을 반환한다.

    Returns
    -------
    debris_dict : dict
        { 'debris0001': {'alt0_km':..., 'RAAN':..., 'mass':..., ...}, ... }
    """
    return load_yaml(debris_yaml_path)


def debris_to_mee(debris_info, params):
    """
    debris.yaml의 한 항목을 MEE 상태 벡터로 변환한다.

    Parameters
    ----------
    debris_info : dict
        {'alt0_km': float, 'e': float, 'i': float,
         'RAAN': float, 'AOP': float, 'nu': float, 'mass': float}
    params : dict

    Returns
    -------
    mee  : ndarray, shape (6,)
    mass : float [kg]
    """
    from src.dynamics import keplerian_to_mee

    Re  = params['Re']
    mu  = params['mu']

    alt_m = debris_info['alt0_km'] * 1000.0   # km → m
    a     = Re + alt_m                         # 반장축 (원궤도) [m]
    e     = debris_info['e']
    i     = np.deg2rad(debris_info['i'])
    RAAN  = np.deg2rad(debris_info['RAAN'])
    AOP   = np.deg2rad(debris_info['AOP'])
    nu    = np.deg2rad(debris_info['nu'])
    mass  = debris_info['mass']

    mee = keplerian_to_mee(a, e, i, RAAN, AOP, nu)
    return mee, mass