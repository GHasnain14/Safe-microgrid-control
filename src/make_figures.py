"""Run the experiment and draw the figures used in the README.

Usage:  python src/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_experiment import OUT, main

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
C = {"No battery": "#9aa5b1", "Rule-based": "#e69f00", "MPC": "#0072b2",
     "Learned raw": "#d55e00", "Learned safe": "#009e73", "Carbon": "#56b4e9"}


def week_window(data, start_day):
    t0 = start_day * 24
    return np.arange(t0, t0 + 7 * 24)


def main_figs():
    cfg, data, traces, table = main()
    figdir = os.path.join(OUT, "figures")
    os.makedirs(figdir, exist_ok=True)
    tr = {k: v.set_index("t") for k, v in traces.items()}

    # Pick the test week with the highest no-battery peak (most interesting for limits)
    nb = tr["No battery"]
    worst_t = int(nb.grid.idxmax())
    day0 = max(243, worst_t // 24 - 3)
    w = week_window(data, day0)
    hrs = (w - w[0]) / 24

    # Figure 1: what each controller does to the grid in one week
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.4, 1]})
    ax[0].plot(hrs, data.load_kw[w], color="#333", lw=1.2, label="Building load")
    ax[0].fill_between(hrs, data.pv_kw[w], color="#f5b301", alpha=0.45, label="Solar PV")
    ax[0].set_ylabel("kW"); ax[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=8.5)
    ax[0].set_title("A test week: load, solar, and what each controller draws from the grid", loc="left")
    for name, col, lab in [("No battery", C["No battery"], "No battery"),
                           ("Rule-based", C["Rule-based"], "Rule-based"),
                           ("MPC, realistic forecast + safety filter", C["MPC"], "MPC + safety filter")]:
        ax[1].plot(hrs, tr[name].grid.loc[w], color=col, lw=1.4 if "MPC" in name else 1.1, label=lab)
    ax[1].axhline(cfg.grid_max_kw, color="#c0392b", ls="--", lw=1, label=f"Import limit {cfg.grid_max_kw:.0f} kW")
    ax[1].axhline(0, color="#999", lw=0.6)
    ax[1].set_ylabel("Grid power (kW)\n+ import / − export"); ax[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=8.5)
    shade = data.price[w] >= 0.30
    for a in ax[1:]:
        a.fill_between(hrs, 0, 1, where=shade, transform=a.get_xaxis_transform(), color="#c0392b", alpha=0.06, lw=0)
    ax[2].plot(hrs, 100 * tr["Rule-based"].soc.loc[w], color=C["Rule-based"], lw=1.1, label="Rule-based")
    ax[2].plot(hrs, 100 * tr["MPC, realistic forecast + safety filter"].soc.loc[w], color=C["MPC"], lw=1.4, label="MPC + safety filter")
    ax[2].axhspan(100 * cfg.soc_min, 100 * cfg.soc_max, color="#2bb673", alpha=0.07)
    ax[2].set_ylabel("Battery SOC (%)"); ax[2].set_ylim(0, 100); ax[2].set_xlabel("Day of the week")
    ax[2].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.text(0.99, 0.005, "Pale red bands = peak tariff hours (14:00–20:00). Green band = safe SOC window.", ha="right", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.subplots_adjust(hspace=0.32)
    fig.savefig(os.path.join(figdir, "1_week_grid_and_soc.png")); plt.close(fig)

    # Figure 2: the safety filter on a learned controller
    raw, safe = tr["Learned policy, no safety filter"], tr["Learned policy + safety filter"]
    t_bad = int(((raw.soc < cfg.soc_min) | (raw.soc > cfg.soc_max)).idxmax())
    w2 = np.arange(max(raw.index.min(), t_bad - 48), min(raw.index.max(), t_bad - 48 + 7 * 24))
    h2 = (w2 - w2[0]) / 24
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    ax[0].axhspan(100 * cfg.soc_min, 100 * cfg.soc_max, color="#2bb673", alpha=0.08, label="Safe operating window")
    ax[0].plot(h2, 100 * raw.soc.loc[w2], color=C["Learned raw"], lw=1.2, label="Learned policy alone")
    ax[0].plot(h2, 100 * safe.soc.loc[w2], color=C["Learned safe"], lw=1.4, label="Learned policy + safety filter")
    out = (raw.soc.loc[w2] < cfg.soc_min) | (raw.soc.loc[w2] > cfg.soc_max)
    ax[0].scatter(h2[out.to_numpy()], 100 * raw.soc.loc[w2][out], s=10, color=C["Learned raw"], zorder=3)
    ax[0].set_ylabel("Battery SOC (%)"); ax[0].set_ylim(0, 102)
    ax[0].set_title("Same learned controller, with and without the independent safety filter", loc="left", pad=22)
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    corr = (safe.p_prop - safe.p).loc[w2]
    ax[1].bar(h2, corr, width=1 / 24, color="#555")
    ax[1].set_ylabel("Filter correction (kW)\nproposed − applied"); ax[1].set_xlabel("Days")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "2_safety_filter.png")); plt.close(fig)

    # Figure 3: cost vs CO2 trade-off
    rows = ["Rule-based", "MPC, realistic forecast + safety filter",
            "Carbon-aware MPC, realistic forecast + filter", "Learned policy + safety filter"]
    labels = ["Rule-based", "MPC\n+ filter", "Carbon-aware\nMPC + filter", "Learned\n+ filter"]
    cols = [C["Rule-based"], C["MPC"], C["Carbon"], C["Learned safe"]]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for a, col, title in [(ax[0], "Saving vs no battery (%)", "Cost saving vs no battery (%)"),
                          (ax[1], "CO2 (t)", "Grid CO₂, Sep–Dec (t)"),
                          (ax[2], "Grid limit breaches (h)", "Hours above import limit")]:
        vals = table.loc[rows, col].to_numpy(float)
        a.bar(labels, vals, color=cols)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8.5)
        a.set_title(title, loc="left", fontsize=10); a.tick_params(axis="x", labelsize=8)
        if col == "CO2 (t)":
            a.set_ylim(vals.min() - 2, vals.max() + 1)
            a.axhline(table.loc["No battery", "CO2 (t)"], color="#9aa5b1", ls=":", lw=1)
        if col == "Grid limit breaches (h)":
            a.axhline(table.loc["No battery", col], color="#9aa5b1", ls=":", lw=1)
    fig.text(0.01, 0.01, "Dotted line = no battery. Test period Sep–Dec, hourly.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(figdir, "3_cost_co2_limits.png")); plt.close(fig)
    print("Figures saved to", figdir)


if __name__ == "__main__":
    main_figs()
