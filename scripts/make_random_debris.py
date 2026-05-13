"""
make_random_debris.py
=====================
simulation.yaml 의 debris_generation 섹션을 읽어, 거기 정의된 set 들을
모두 생성한다.

debris 분포는 코드에 하드코딩되지 않고 simulation.yaml 에서 가져온다:
  debris_generation.altitude_km.{min, max}
  debris_generation.raan_deg.{min, max}
  debris_generation.mass_kg.{min, max}
  debris_generation.inclination_deg
  debris_generation.eccentricity
  debris_generation.arg_perigee_deg

생성할 set 목록은 simulation.yaml 의 debris_generation.sets :
  - n / output / seed  의 list 로 정의.

실행 :
  python scripts/make_random_debris.py
"""

import os
import sys
import yaml
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _load_debris_generation_cfg(simulation_yaml='configs/simulation.yaml'):
    """simulation.yaml 의 debris_generation 섹션을 로드."""
    with open(simulation_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if 'debris_generation' not in cfg:
        raise KeyError(
            f"'{simulation_yaml}' 에 'debris_generation' 섹션이 없습니다. "
            "분포 / set 목록을 yaml 에 정의하세요."
        )
    return cfg['debris_generation']


def generate_one_set(num_debris, output_path, seed, dist):
    """단일 debris set 생성. dist 는 debris_generation 섹션 dict."""
    rng = random.Random(seed)

    alt_lo,  alt_hi  = dist['altitude_km']['min'], dist['altitude_km']['max']
    raan_lo, raan_hi = dist['raan_deg']['min'],    dist['raan_deg']['max']
    m_lo,    m_hi    = dist['mass_kg']['min'],     dist['mass_kg']['max']
    inc_deg = dist['inclination_deg']
    ecc     = dist['eccentricity']
    aop_deg = dist['arg_perigee_deg']

    debris_data = {}
    for i in range(1, num_debris + 1):
        name = f"debris{i:04d}"
        debris_data[name] = {
            'alt0_km': round(rng.uniform(alt_lo,  alt_hi),  2),
            'e'      : ecc,
            'i'      : inc_deg,
            'RAAN'   : round(rng.uniform(raan_lo, raan_hi), 2),
            'AOP'    : aop_deg,
            'nu'     : round(rng.uniform(0.0,     360.0),   2),
            'mass'   : round(rng.uniform(m_lo,    m_hi),    2),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(debris_data, f, default_flow_style=False, sort_keys=False)
    print(f"  {num_debris:5d} debris  →  {output_path}  (seed = {seed})")


def main():
    dist = _load_debris_generation_cfg('configs/simulation.yaml')
    sets = dist.get('sets', [])
    if not sets:
        raise ValueError(
            "simulation.yaml 의 debris_generation.sets 가 비어 있습니다."
        )

    print("=" * 60)
    print("  Generating debris yaml files (config from simulation.yaml)")
    print("=" * 60)
    print(f"  altitude    : [{dist['altitude_km']['min']:.1f}, "
          f"{dist['altitude_km']['max']:.1f}] km")
    print(f"  RAAN        : [{dist['raan_deg']['min']:.1f}, "
          f"{dist['raan_deg']['max']:.1f}] deg")
    print(f"  mass        : [{dist['mass_kg']['min']:.1f}, "
          f"{dist['mass_kg']['max']:.1f}] kg")
    print(f"  inclination : {dist['inclination_deg']:.2f} deg")
    print(f"  eccentricity: {dist['eccentricity']}")
    print("-" * 60)
    for item in sets:
        generate_one_set(
            num_debris  = int(item['n']),
            output_path = str(item['output']),
            seed        = int(item['seed']),
            dist        = dist,
        )
    print("=" * 60)


if __name__ == "__main__":
    main()