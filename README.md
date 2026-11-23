# Safe learning-based control of a solar + battery microgrid

A learned controller is fast and nearly optimal, but it has no idea where the limits are.
This project shows that an **independent safety filter**, a separate layer that only knows the limits, keeps it safe at almost no cost.

The idea comes from my industrial work on wellhead control panels: the basic control system can be clever, but a separate safety system enforces the limits, the way a safety PLC sits beside the process PLC (IEC 61511). Here I apply the same principle to learning-based control of an energy system.

![Safety filter on a learned controller](results/figures/2_safety_filter.png)

## The system

| Part | Value |
|---|---|
| Building load | Primary school, Houston (DOE reference building, hourly, 1 year) |
| Solar PV | 300 kWp, from hourly irradiance (TMY, Houston) |
| Battery | 500 kWh, 125 kW, 95% one-way efficiency |
| Safe operating window | State of charge 10–90% |
| Grid | Import limit 250 kW (contract), time-of-use tariff 0.08 / 0.15 / 0.30 $/kWh (peak 14:00–20:00), export 0.04 $/kWh |
| Battery wear | 0.02 $ per kWh of throughput |
| Train / test | Jan–Aug / Sep–Dec (2,928 test hours) |

## Controllers compared

1. **No battery**: the reference.
2. **Rule-based**: charge from surplus solar, discharge when the building needs power. Typical of many installed systems.
3. **MPC (model predictive control)**: every hour it solves a 24-hour linear program (cost, battery wear, limits) and applies only the first hour. Tested with perfect forecasts (benchmark) and with realistic forecasts whose error grows with lead time (PV ±30%, load ±10% at 24 h).
4. **Carbon-aware MPC**: the same, plus a carbon price of $0.10/kg CO₂ on grid imports.
5. **Learned policy**: a small neural network (2 × 64) trained to imitate MPC from measurements only. It needs no optimiser at run time, but it has no built-in notion of the limits.

## The safety filter

`src/safety.py`, about 30 lines. Every hour it takes the proposed battery power and the measured state, and changes the power as little as possible so that:

1. the state of charge stays in 10–90% after the hour, and
2. grid import stays ≤ 250 kW, **if that is physically possible**. If even a full safe discharge cannot hold the limit, the SOC limit wins and the overload is reported, not hidden.

It does not know which controller is running. **Property test:** 100,000 random states and deliberately wild proposals give 0 SOC breaches and 0 avoidable grid-limit breaches (`tests/test_safety_filter.py`).

## Results (test period, Sep–Dec)

| Controller | Cost saving vs no battery | Grid CO₂ (t) | Hours outside SOC window | Hours above import limit |
|---|---:|---:|---:|---:|
| No battery | 0% | 66.0 | 0 | 36 |
| Rule-based | 4.1% | 63.2 | 0 | 36 |
| MPC, perfect forecast (benchmark) | 16.6% | 65.7 | 0 | 0 |
| MPC, realistic forecast + filter | 16.5% | 65.8 | 0 | 10 |
| Carbon-aware MPC, realistic forecast + filter | 16.4% | 63.7 | 0 | 7 |
| Learned policy, **no** filter | 16.1% | 66.1 | **661** | **86** |
| Learned policy **+ filter** | 15.9% | 66.0 | **0** | 18 |

Full table, including peak import, battery cycles and filter interventions: `results/results_test_period.csv`.

![A test week](results/figures/1_week_grid_and_soc.png)
![Cost, CO2 and limit breaches](results/figures/3_cost_co2_limits.png)

## What I learned

1. **Planning matters.** MPC saves 16.5% with realistic forecasts, four times the rule-based saving (4.1%), and it is the only controller that respects the grid limit almost all the time. The rule empties the battery in the morning and has nothing left for the afternoon peak.
2. **The learned policy is nearly as cheap as MPC but unsafe on its own.** Without the filter it left the safe window in 661 of 2,928 hours (23%), with SOC from 6.4% to 100%, and broke the grid limit in 86 hours.
3. **The safety filter fixes this almost for free.** With it, there are 0 SOC breaches, and the cost rises by only $82 over four months (+0.2%). Most corrections are small (mean 3.9 kW).
4. **A safety layer protects; it cannot plan.** Even with the filter, the learned policy is above the grid limit for 18 hours, because the battery was already empty when the peak arrived. Only a controller that plans ahead (MPC: 10 h) avoids that. Safety and intelligence are different jobs, and both are needed.
5. **Cheapest is not cleanest.** Cost-optimal MPC charges from the grid at night to avoid peak prices, so it emits 2.6 t more CO₂ than the simple rule. A carbon price of $0.10/kg cuts 2.1 t (−3.2%) for only $25 (+0.07%).

## Limitations

- Simulation only, one building and one year, hourly steps.
- Simple battery model: no temperature, no ageing beyond a throughput cost.
- Forecast errors are synthetic, not from a real forecasting model.
- The CO₂ intensity profile is California's (CAISO), while load and PV are Houston's. CO₂ numbers show the trend, not a real site.
- The learned policy is trained with one random seed. Breach counts will vary with retraining. The filter's guarantee does not.

## Next steps

- Learn the policy with reinforcement learning instead of imitation, still behind the filter.
- Replace the filter's one-step check with a multi-step one (e.g. a control barrier function or a short MPC safety check), so it can also prevent "battery already empty" situations.
- Use real forecasts and a real site's data.

## Run it

```bash
pip install -r requirements.txt
python tests/test_safety_filter.py      # property test of the safety filter
python src/make_figures.py              # runs every controller and saves results + figures (~40 s)
```

## Files

```
src/microgrid.py        system model, data loading, tariff
src/safety.py           independent safety filter
src/controllers.py      rule-based, MPC (cvxpy + HiGHS), learned policy (scikit-learn)
src/run_experiment.py   train, test, metrics table
src/make_figures.py     figures
tests/                  property test of the safety filter
results/                table, traces, figures
```

## Data and credits

Hourly load, irradiance and CO₂ profiles are those shipped with [pymgrid](https://github.com/Total-RD/pymgrid) (U.S. DOE commercial reference buildings, TMY weather data, CAISO grid emissions). Tariff and system sizes are my own illustrative choices.

Ghulam Hasnain · [Portfolio](https://sites.google.com/view/ghulam-hasnain/home)
