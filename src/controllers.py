"""Controllers that propose a battery power for the next hour.

1. RuleBased      - common 'self-consumption' logic used in many real systems.
2. MPC            - model predictive control: plans 24 h ahead, applies the first hour,
                    re-plans every hour (receding horizon). Linear program in cvxpy.
3. LearnedPolicy  - a neural network trained to imitate MPC from current measurements
                    only. Fast and cheap to run, but it has no built-in notion of limits.
"""
import cvxpy as cp
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from microgrid import Config
from safety import safe_power_bounds


class RuleBased:
    name = "Rule-based"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def act(self, t, soc, data, forecast=None):
        net = data.load_kw[t] - data.pv_kw[t]      # > 0 deficit, < 0 surplus
        lo, hi = safe_power_bounds(soc, self.cfg)  # a good rule respects the limits itself
        return float(np.clip(net, lo, hi))


class MPC:
    """Receding-horizon economic MPC, solved as a linear program each hour."""

    def __init__(self, cfg: Config, horizon: int = 24, name: str = "MPC", carbon_price: float = 0.0):
        """carbon_price in $/kg CO2: 0 = pure cost; > 0 also penalises grid emissions."""
        self.cfg, self.H, self.name = cfg, horizon, name
        H = horizon
        self.soc0 = cp.Parameter(nonneg=True)
        self.net = cp.Parameter(H)               # forecast load - pv
        self.price = cp.Parameter(H, nonneg=True)
        self.co2 = cp.Parameter(H, nonneg=True)   # grid carbon intensity forecast (kg/kWh)
        self.carbon_price = carbon_price

        c = cp.Variable(H, nonneg=True)          # charge power
        d = cp.Variable(H, nonneg=True)          # discharge power
        imp = cp.Variable(H, nonneg=True)        # grid import
        exp_ = cp.Variable(H, nonneg=True)       # grid export
        s = cp.Variable(H + 1)                   # state of charge
        slack = cp.Variable(H, nonneg=True)      # grid-limit violation (heavily penalised)
        E = cfg.bat_kwh
        cons = [
            s[0] == self.soc0,
            s[1:] == s[:-1] + cfg.eta * c / E - d / (cfg.eta * E),
            s >= cfg.soc_min, s <= cfg.soc_max,
            c <= cfg.bat_kw, d <= cfg.bat_kw,
            imp - exp_ == self.net - d + c,
            imp <= cfg.grid_max_kw + slack,
            s[H] >= cfg.soc0 * 0.6,              # do not empty the battery at the horizon end
        ]
        cost = (self.price @ imp - cfg.export_price * cp.sum(exp_)
                + cfg.deg_cost * cp.sum(c + d) + 10.0 * cp.sum(slack)
                + carbon_price * (self.co2 @ imp))
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self.c, self.d = c, d

    def act(self, t, soc, data, forecast):
        net_f, price_f, co2_f = forecast(t, self.H)
        self.soc0.value = float(np.clip(soc, self.cfg.soc_min, self.cfg.soc_max))
        self.net.value = net_f
        self.price.value = price_f
        self.co2.value = co2_f
        self.prob.solve(solver=cp.HIGHS)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return RuleBased(self.cfg).act(t, soc, data)
        return float(self.d.value[0] - self.c.value[0])


def features(t, soc, data):
    h = t % 24
    nxt = slice(t + 1, t + 7)
    return [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), soc,
            data.pv_kw[t], data.load_kw[t], data.price[t],
            float(np.mean(data.pv_kw[nxt])) if t + 1 < len(data.pv_kw) else data.pv_kw[t],
            float(np.mean(data.load_kw[nxt])) if t + 1 < len(data.load_kw) else data.load_kw[t]]


class LearnedPolicy:
    name = "Learned policy"

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.model = make_pipeline(StandardScaler(),
                                   MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600,
                                                early_stopping=True, random_state=seed))

    def fit(self, X, y):
        self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def act(self, t, soc, data, forecast=None):
        return float(self.model.predict(np.asarray([features(t, soc, data)]))[0])
