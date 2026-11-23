"""Property test: whatever a controller proposes, the safety filter must keep
the battery inside its SOC window, and must hold the grid limit whenever
that is physically possible.

Usage:  python tests/test_safety_filter.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from microgrid import Config, physical_limits, step_soc  # noqa: E402
from safety import safe_power_bounds, safety_filter  # noqa: E402


def main(n=100_000, seed=0):
    cfg = Config()
    rng = np.random.default_rng(seed)
    soc_viol = grid_viol_avoidable = unreachable = 0
    for _ in range(n):
        soc = rng.uniform(cfg.soc_min, cfg.soc_max)
        load = rng.uniform(0, 450)
        pv = rng.uniform(0, 260)
        p_prop = rng.uniform(-400, 400)          # deliberately wild proposals
        fr = safety_filter(p_prop, soc, load, pv, cfg)
        p = physical_limits(fr.p_safe, soc, cfg)
        new_soc = step_soc(soc, p, cfg)
        if new_soc < cfg.soc_min - 1e-9 or new_soc > cfg.soc_max + 1e-9:
            soc_viol += 1
        grid = load - pv - p
        _, hi = safe_power_bounds(soc, cfg)
        possible = (load - pv - hi) <= cfg.grid_max_kw + 1e-9
        if grid > cfg.grid_max_kw + 1e-6:
            if possible:
                grid_viol_avoidable += 1
            else:
                unreachable += 1
                assert fr.reason == "grid_unreachable"
    print(f"{n} random cases | SOC breaches: {soc_viol} | avoidable grid breaches: {grid_viol_avoidable} "
          f"| grid limit physically unreachable (reported): {unreachable}")
    assert soc_viol == 0 and grid_viol_avoidable == 0
    print("PASS")


if __name__ == "__main__":
    main()
