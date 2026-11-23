"""Independent safety layer (a 'safety filter' or 'shield').

It does not know which controller is running or how it decided. It only sees
the measured state (state of charge, load, PV) and the operating limits, and
it changes the proposed battery power as little as possible so that:
  1. state of charge stays inside [soc_min, soc_max] after this hour, and
  2. grid import stays at or below the contracted maximum.

This mirrors how a safety instrumented system sits beside a basic process
control system: the controller can be clever, but the limits are enforced
by a separate, simple, verifiable layer.
"""
from dataclasses import dataclass

import numpy as np

from microgrid import Config


@dataclass
class FilterResult:
    p_safe: float
    intervened: bool
    reason: str          # "", "soc", "grid", "grid_unreachable"


def safe_power_bounds(soc: float, cfg: Config):
    """Battery power range that keeps SOC inside the operating window for one hour."""
    p_dis_max = max(0.0, (soc - cfg.soc_min) * cfg.bat_kwh * cfg.eta)
    p_ch_max = max(0.0, (cfg.soc_max - soc) * cfg.bat_kwh / cfg.eta)
    lo = -min(cfg.bat_kw, p_ch_max)   # most charging allowed
    hi = min(cfg.bat_kw, p_dis_max)   # most discharging allowed
    return lo, hi


def safety_filter(p_proposed: float, soc: float, load_kw: float, pv_kw: float,
                  cfg: Config) -> FilterResult:
    lo, hi = safe_power_bounds(soc, cfg)
    reason = ""

    # Grid import limit: load - pv - p <= grid_max  ->  p >= load - pv - grid_max
    p_grid_min = load_kw - pv_kw - cfg.grid_max_kw
    if p_grid_min > lo:
        if p_grid_min <= hi:
            lo = p_grid_min
        else:
            # Even full safe discharge cannot hold the grid limit. SOC limit has
            # priority (protects the asset); the overload is reported, not hidden.
            lo = hi
            reason = "grid_unreachable"

    p_safe = float(np.clip(p_proposed, lo, hi))
    intervened = abs(p_safe - p_proposed) > 1e-6
    if intervened and not reason:
        # Raised to lo because of the grid limit, or clipped by the SOC window?
        reason = "grid" if (p_proposed < lo and abs(lo - p_grid_min) < 1e-9) else "soc"
    return FilterResult(p_safe, intervened, reason)
