"""Microgrid model: solar PV + building load + battery + grid connection.

One step = one hour. Sign convention for battery power p (kW):
    p > 0  -> battery discharges into the building
    p < 0  -> battery charges
Grid power g = load - pv - p   (g > 0 import, g < 0 export)

Data: hourly profiles shipped with pymgrid (DOE reference buildings,
TMY solar data, CAISO grid carbon intensity).
"""
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd


@dataclass
class Config:
    # Solar
    pv_kwp: float = 300.0        # installed PV peak power (kWp)
    pr: float = 0.85             # performance ratio (inverter, temperature, soiling losses)
    # Battery
    bat_kwh: float = 500.0       # usable energy capacity
    bat_kw: float = 125.0        # inverter power limit (charge and discharge)
    eta: float = 0.95            # one-way efficiency (round trip ~0.90)
    soc0: float = 0.50
    # Operating limits that the SAFETY LAYER enforces
    soc_min: float = 0.10        # below this: deep discharge, faster ageing, BMS trip risk
    soc_max: float = 0.90        # above this: overcharge stress
    grid_max_kw: float = 250.0   # contracted maximum import from the grid
    # Economics
    export_price: float = 0.04   # $/kWh paid for exported energy
    deg_cost: float = 0.02       # $/kWh of battery throughput (wear)


def tou_price(hours: np.ndarray) -> np.ndarray:
    """Time-of-use import tariff in $/kWh (illustrative)."""
    h = hours % 24
    price = np.full(h.shape, 0.08)                      # night, off-peak
    price[(h >= 8) & (h < 14)] = 0.15                   # morning shoulder
    price[(h >= 20) & (h < 22)] = 0.15                  # evening shoulder
    price[(h >= 14) & (h < 20)] = 0.30                  # afternoon peak
    return price


def load_data(cfg: Config) -> pd.DataFrame:
    import pymgrid
    root = os.path.join(os.path.dirname(pymgrid.__file__), "data")
    ghi = pd.read_csv(os.path.join(root, "pv", "Houston_722430TYA.csv")).iloc[:, 0].to_numpy(float)
    load = pd.read_csv(os.path.join(root, "load",
                       "RefBldgPrimarySchoolNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv")).iloc[:, 0].to_numpy(float)
    co2 = pd.read_csv(os.path.join(root, "co2", "co2_caiso.csv")).iloc[:, 0].to_numpy(float)
    hours = np.arange(len(load))
    return pd.DataFrame({
        "hour": hours,
        "pv_kw": ghi / 1000.0 * cfg.pv_kwp * cfg.pr,   # irradiance (W/m2) -> PV output (kW)
        "load_kw": load,
        "price": tou_price(hours),
        "co2_kg_per_kwh": co2,
    })


def step_soc(soc: float, p: float, cfg: Config) -> float:
    """Battery energy balance for one hour."""
    if p >= 0:   # discharge: more energy leaves the cells than reaches the bus
        soc -= p / cfg.eta / cfg.bat_kwh
    else:        # charge: less energy reaches the cells than is taken from the bus
        soc -= p * cfg.eta / cfg.bat_kwh
    return soc


def physical_limits(p: float, soc: float, cfg: Config) -> float:
    """What the hardware itself allows: inverter rating and an empty/full battery.
    This is NOT the safety layer; it only stops physically impossible actions."""
    p = float(np.clip(p, -cfg.bat_kw, cfg.bat_kw))
    p_dis_max = soc * cfg.bat_kwh * cfg.eta
    p_ch_max = (1.0 - soc) * cfg.bat_kwh / cfg.eta
    return float(np.clip(p, -p_ch_max, p_dis_max))
