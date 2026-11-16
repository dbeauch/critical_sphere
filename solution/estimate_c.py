"""
Measure c from pure random walkers 
Rather than simulate branching and absorption, we can model it with population = e^(sigma t) x (pure-walker survival)
Pure walkers leak at rate lambda_esc = Lambda * D / R^2, so criticality (sigma = lambda_esc) gives c = sqrt(Lambda)
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from walkers import generate_walkers, diffuse, remove_external_walkers, resample, fit_growth_rate

# one JSON record per line
# [json.loads(line) for line in open(RESULTS_PATH)]
RESULTS_PATH = Path(__file__).parent / "results" / "estimate_c.jsonl"


def save_record(record, path=RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def timed_escape_rate(check, run_id, **params):
    """
    escape_rate(**params), saving the parameters and result under run_id
    """
    start = time.perf_counter()
    rate = escape_rate(**params)
    if run_id is not None:
        R, D = params.get("R", 1.0), params.get("D", 1.0)
        save_record({"run_id": run_id, "kind": "escape_rate", "check": check, **params,
                     "lambda_esc": rate, "Lambda": rate * R**2 / D, "seconds": time.perf_counter() - start})
    return rate


def escape_rate(R=1.0, D=1.0, dt=1e-4, num_walkers=20_000, t_burn=None, t_meas=None, seed=0, kill_prob=None):
    """
    Long simulated decay rate lambda_esc of pure walkers removed on reaching radius R

    t_burn, t_meas default to 0.3 and 1.0 times tau = R^2 / D
    """
    tau = R**2 / D
    if t_burn is None:
        t_burn = 0.3 * tau
    if t_meas is None:
        t_meas = 1.0 * tau

    n_burn = round(t_burn / dt)
    n_meas = round(t_meas / dt)

    rng = np.random.default_rng(seed)
    walkers = generate_walkers(num_walkers, R, rng)
    log_w = 0.0

    ts = []
    log_pop = []
    for step in range(n_burn + n_meas):
        walkers = diffuse(walkers, D, dt, rng)
        if kill_prob is None:
            walkers = remove_external_walkers(walkers, R)
        else:
            walkers = walkers[rng.random(len(walkers)) >= kill_prob]
        walkers, log_w = resample(walkers, num_walkers, log_w, rng)

        if step >= n_burn:
            ts.append((step + 1) * dt)
            log_pop.append(log_w + np.log(len(walkers) / num_walkers))

    return -fit_growth_rate(ts, log_pop)


def scaling_check(pairs=((1.0, 1.0), (2.0, 1.0), (1.0, 3.0), (0.5, 0.2)), dt_frac=1e-3, num_walkers=20_000, run_id=None):
    """
    check lambda = lambda_esc * R^2 / D is the same for every (R, D)
    
    dt = dt_frac * tau keeps the time-step bias identical across runs
    Each pair gets its own seed so the runs are independent
    Returns the list of lambdas
    If run_id is given, every run is saved to RESULTS_PATH
    """
    lambdas = []
    for seed, (R, D) in enumerate(pairs):
        tau = R**2 / D
        rate = timed_escape_rate("scaling", run_id, R=R, D=D, dt=dt_frac * tau, num_walkers=num_walkers, seed=seed)
        Lambda = rate * R**2 / D
        lambdas.append(Lambda)
        print(f"  R = {R:<4g} D = {D:<4g} Lambda = {Lambda:.3f}")
    print(f"  spread: {np.mean(lambdas):.3f} +/- {np.std(lambdas, ddof=1):.3f}")
    return lambdas


def extrapolate_dt(dts=(4e-4, 2e-4, 1e-4, 5e-5), seeds=5, num_walkers=20_000, run_id=None):
    """
    Check 2: remove the time-step bias (R = D = 1)

    Account for walkers leaking through the wall in between time steps and adjust lambda accordingly
    Returns (Lambda_0, its uncertainty).
    If run_id is given, every run and the fit are saved to RESULTS_PATH
    """
    runs = np.array([[timed_escape_rate("extrapolate_dt", run_id, dt=dt, num_walkers=num_walkers, seed=seed)
                      for seed in range(seeds)] for dt in dts])
    means = runs.mean(axis=1)
    sem = np.sqrt(np.mean(runs.var(axis=1, ddof=1)) / seeds)
    for dt, mean in zip(dts, means):
        print(f"  dt = {dt:.1e}  sqrt(dt) = {np.sqrt(dt):.4f}  Lambda = {mean:.3f} +/- {sem:.3f}")

    # equal error bars
    (slope, Lambda_0), cov = np.polyfit(np.sqrt(dts), means, 1, w=np.full(len(dts), 1 / sem), cov="unscaled")
    Lambda_0_err = np.sqrt(cov[1, 1])
    if run_id is not None:
        save_record({"run_id": run_id, "kind": "extrapolate_dt", "dts": list(dts), "seeds": seeds,
                     "num_walkers": num_walkers, "means": means.tolist(), "sem": sem,
                     "slope": slope, "Lambda_0": Lambda_0, "Lambda_0_err": Lambda_0_err})
    print(f"  fit: Lambda = {Lambda_0:.3f} + ({slope:.1f}) * sqrt(dt)   [wall-shift estimate of slope ~ -16.3]")
    return Lambda_0, Lambda_0_err


if __name__ == "__main__":
    import sys

    # True: small-scale test
    # False: long running for a good estimate of c
    quick = False

    if quick:
        num_walkers, dts, seeds = 2000, (4e-3, 2e-3, 1e-3, 5e-4), 4
    else:
        num_walkers, dts, seeds = 20000, (4e-4, 2e-4, 1e-4, 5e-5), 5

    # run tracking
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    start = time.perf_counter()

    print("Scaling check (dt = 1e-3 tau):")
    scaling_lambdas = scaling_check(num_walkers=num_walkers, run_id=run_id)

    print("Time-step extrapolation (R = D = 1):")
    Lambda_0, Lambda_0_err = extrapolate_dt(dts=dts, seeds=seeds, num_walkers=num_walkers, run_id=run_id)

    # c = sqrt(Lambda), so dc = dLambda / (2 sqrt(Lambda))
    c = np.sqrt(Lambda_0)
    c_err = Lambda_0_err / (2 * c)
    print(f"Lambda_0 = {Lambda_0:.4f} +/- {Lambda_0_err:.4f}")
    print(f"c = {c:.4f} +/- {c_err:.4f}")

    save_record({"run_id": run_id, "kind": "summary", "quick": quick, "num_walkers": num_walkers,
                 "dts": list(dts), "seeds": seeds, "scaling_Lambdas": scaling_lambdas,
                 "Lambda_0": Lambda_0, "Lambda_0_err": Lambda_0_err, "c": c, "c_err": c_err,
                 "seconds": time.perf_counter() - start})
    print(f"Saved to {RESULTS_PATH} (run_id {run_id})")
