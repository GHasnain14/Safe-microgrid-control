"""Run every controller on the same data and compare them.

Train period: Jan-Aug (the learned policy imitates MPC here).
Test period:  Sep-Dec (every number in the results table comes from here).

Usage:  python src/run_experiment.py
"""
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from controllers import MPC, LearnedPolicy, RuleBased, features
from microgrid import Config, load_data, physical_limits, step_soc
from safety import safety_filter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
TRAIN_END = 243 * 24          # 1 Sep (hour index)


def make_forecasts(data, seed=1):
    pv, load, price, co2 = data.pv_kw, data.load_kw, data.price, data.co2_kg_per_kwh
    n = len(load)

    def perfect(t, H):
        idx = np.arange(t, t + H) % n
        return load[idx] - pv[idx], price[idx], co2[idx]

    rng = np.random.default_rng(seed)
    # Pre-draw forecast errors that grow with lead time (0 h ahead = measured)
    lead = np.sqrt(np.arange(24) / 24.0)
    pv_err = rng.normal(0, 1, (n, 24)) * 0.30 * lead
    load_err = rng.normal(0, 1, (n, 24)) * 0.10 * lead

    def noisy(t, H):
        idx = np.arange(t, t + H) % n
        pv_f = np.clip(pv[idx] * (1 + pv_err[t, :H]), 0, None)
        load_f = np.clip(load[idx] * (1 + load_err[t, :H]), 0, None)
        return load_f - pv_f, price[idx], co2[idx]

    return perfect, noisy


def simulate(ctrl, data, cfg, t0, t1, forecast=None, use_filter=True, soc0=None, record_xy=False):
    soc = cfg.soc0 if soc0 is None else soc0
    rows, X, y = [], [], []
    for t in range(t0, t1):
        p_prop = 0.0 if ctrl is None else ctrl.act(t, soc, data, forecast)
        if record_xy:
            X.append(features(t, soc, data)); y.append(p_prop)
        if use_filter:
            fr = safety_filter(p_prop, soc, data.load_kw[t], data.pv_kw[t], cfg)
            p, intervened, reason = fr.p_safe, fr.intervened, fr.reason
        else:
            p, intervened, reason = p_prop, False, ""
        p = physical_limits(p, soc, cfg)
        soc = step_soc(soc, p, cfg)
        grid = data.load_kw[t] - data.pv_kw[t] - p
        rows.append(dict(t=t, p_prop=p_prop, p=p, soc=soc, grid=grid,
                         intervened=intervened, reason=reason))
    df = pd.DataFrame(rows)
    return (df, X, y) if record_xy else df


def metrics(df, data, cfg):
    t = df.t.to_numpy()
    imp = df.grid.clip(lower=0).to_numpy()
    exp_ = (-df.grid).clip(lower=0).to_numpy()
    cost = (data.price[t] * imp).sum() - cfg.export_price * exp_.sum() + cfg.deg_cost * df.p.abs().sum()
    pv_used = np.minimum(data.pv_kw[t], data.load_kw[t] - df.p.clip(upper=0).to_numpy()).sum()
    return {
        "Cost ($)": round(float(cost)),
        "Grid import (MWh)": round(float(imp.sum() / 1000), 1),
        "Peak import (kW)": round(float(imp.max()), 1),
        "CO2 (t)": round(float((data.co2_kg_per_kwh[t] * imp).sum() / 1000), 1),
        "Battery cycles": round(float(df.p.abs().sum() / 2 / cfg.bat_kwh), 1),
        "SOC limit breaches (h)": int(((df.soc < cfg.soc_min - 1e-6) | (df.soc > cfg.soc_max + 1e-6)).sum()),
        "Grid limit breaches (h)": int((df.grid > cfg.grid_max_kw + 1e-6).sum()),
        "Lowest SOC (%)": round(float(100 * df.soc.min()), 1),
        "Highest SOC (%)": round(float(100 * df.soc.max()), 1),
        "Filter interventions (h)": int(df.intervened.sum()),
        "Mean filter correction (kW)": round(float((df.p_prop - df.p).abs()[df.intervened].mean()), 1) if df.intervened.any() else 0.0,
        "PV self-use (%)": round(float(100 * pv_used / data.pv_kw[t].sum()), 1),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = Config()
    df_data = load_data(cfg)
    data = SimpleNamespace(**{c: df_data[c].to_numpy() for c in df_data.columns})
    perfect, noisy = make_forecasts(data)
    n = len(data.load_kw)

    # 1) Teacher: MPC with perfect forecasts on the training months -> training data
    t_start = time.time()
    mpc_teacher = MPC(cfg, name="MPC (perfect forecast)")
    _, X, y = simulate(mpc_teacher, data, cfg, 0, TRAIN_END, forecast=perfect, record_xy=True)
    learned = LearnedPolicy(cfg).fit(X, y)
    print(f"Trained learned policy on {len(y)} hours in {time.time() - t_start:.0f} s")

    # 2) Evaluate everything on the test months, same start SOC
    runs = {
        "No battery": (None, None, False),
        "Rule-based": (RuleBased(cfg), None, True),
        "MPC, perfect forecast (benchmark)": (MPC(cfg), perfect, True),
        "MPC, realistic forecast + safety filter": (MPC(cfg), noisy, True),
        "Carbon-aware MPC, realistic forecast + filter": (MPC(cfg, carbon_price=0.10), noisy, True),
        "Learned policy, no safety filter": (learned, None, False),
        "Learned policy + safety filter": (learned, None, True),
    }
    results, traces = {}, {}
    for name, (ctrl, fc, filt) in runs.items():
        t_start = time.time()
        df = simulate(ctrl, data, cfg, TRAIN_END, n, forecast=fc, use_filter=filt)
        results[name] = metrics(df, data, cfg)
        traces[name] = df
        print(f"{name:45s} done in {time.time() - t_start:5.1f} s")

    table = pd.DataFrame(results).T
    base = table.loc["No battery", "Cost ($)"]
    table.insert(1, "Saving vs no battery (%)", (100 * (base - table["Cost ($)"]) / base).round(1))
    table.to_csv(os.path.join(OUT, "results_test_period.csv"))
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    for name, df in traces.items():
        df.to_csv(os.path.join(OUT, "trace_" + "".join(ch if ch.isalnum() else "_" for ch in name)[:40] + ".csv"), index=False)
    pd.set_option("display.width", 250)
    print(table.to_string())
    return cfg, data, traces, table


if __name__ == "__main__":
    main()
