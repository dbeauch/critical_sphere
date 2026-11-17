"""
Find c by simulating the full system: diffusion + absorption + branching + escape

From section 1, growth(R) = sigma - k * D / R^2 with k unknown
- simulate the growth rate at several radii
- fit growth against 1 / R^2 and find where it crosses zero: R_crit
- c = R_crit / sqrt(D / sigma)

Checks:
  - convergence: repeat for smaller time steps and watch c settle
  - cases: very different (alpha, beta, nu, D) should all give the same c if only sigma and D matter
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from walkers import generate_walkers, diffuse, remove_external_walkers, resample, fit_growth_rate

# one JSON record per line
# [json.loads(line) for line in open(RESULTS_PATH)]
RESULTS_PATH = Path(__file__).parent / "results" / "find_c.jsonl"

# radii to sweep
RADIUS_FRACS = (2.5, 2.75, 3.0, 3.25, 3.5)

# Parameter sets with the same sigma but very different mixes,
# plus a change in D and a change in sigma
CASES = {
    # name: (D,  alpha, beta, nu)
    "A": (1.0, 0.5, 1.0, 2.5),   # sigma = 1, moderate
    "B": (1.0, 0.0, 0.5, 3.0),   # sigma = 1, no absorption
    "C": (1.0, 2.0, 1.0, 4.0),   # sigma = 1, heavy branching vs heavy absorption
    "D": (2.0, 0.5, 1.0, 2.5),   # doubled D
    "E": (1.0, 0.5, 2.0, 2.0),   # sigma = 1.5
}


def sigma_prod(alpha, beta, nu):
    return beta * (nu - 1) - alpha


def save_record(record, path=RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def interact(walkers, alpha, beta, nu, dt, rng):
    """
    Each walker is absorbed with probability alpha*dt and branches into Poisson(nu) copies with probability beta*dt
    """
    if (alpha + beta) * dt > 0.1:
        raise ValueError("(alpha + beta) * dt is too large")

    # pick the walker outcome
    u = rng.random(len(walkers))
    branched = (u >= alpha * dt) & (u < (alpha + beta) * dt)
    untouched = u >= (alpha + beta) * dt

    offspring = rng.poisson(nu, size=np.count_nonzero(branched))
    children = np.repeat(walkers[branched], offspring, axis=0)
    return np.concatenate([walkers[untouched], children])


def growth_rate(R, D, alpha, beta, nu, num_walkers=10_000, dt_frac=1e-4, seed=0, dt=None, t_burn=None, t_meas=None):
    """
    Long-time growth rate of the full system at radius R

    dt, t_burn, t_meas default to dt_frac, 1.0 and 1.0 times tau = R^2 / D
    burn-in lets the walkers settle from their uniform start into the shape the population keeps
    """
    tau = R**2 / D
    if dt is None:
        dt = dt_frac * tau
    if t_burn is None:
        t_burn = 1.0 * tau
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
        walkers = remove_external_walkers(walkers, R)
        walkers = interact(walkers, alpha, beta, nu, dt, rng)
        walkers, log_w = resample(walkers, num_walkers, log_w, rng)

        if step >= n_burn:
            ts.append((step + 1) * dt)
            log_pop.append(log_w + np.log(len(walkers) / num_walkers))

    return fit_growth_rate(ts, log_pop)


def fit_critical_radius(radii, rates):
    """
    Fit growth = a - b / R^2 and return (R_crit, a, b), where R_crit = sqrt(b / a) is the zero crossing
    """
    slope, a = np.polyfit(1 / np.asarray(radii)**2, rates, 1)
    b = -slope
    return np.sqrt(b / a), a, b


def sweep(D, alpha, beta, nu, dt_frac, seeds, first_seed, num_walkers, pool):
    """
    Measure c once per seed: growth rate at every radius in RADIUS_FRACS, then fit for R_crit
    Every (seed, radius) run is independent, so they can all run in parallel
    """
    sigma = sigma_prod(alpha, beta, nu)
    length = np.sqrt(D / sigma)  # section 1 estimate without c
    radii = length * np.array(RADIUS_FRACS)

    futures = [[pool.submit(growth_rate, R, D, alpha, beta, nu, num_walkers, dt_frac,
                            first_seed + 100 * s + i)
                for i, R in enumerate(radii)]
               for s in range(seeds)]
    rates = np.array([[f.result() for f in row] for row in futures])

    cs = np.array([fit_critical_radius(radii, row)[0] / length for row in rates])
    return radii, rates, cs


def run_sweep(name, dt_frac, seeds, first_seed, num_walkers, pool, run_id):
    D, alpha, beta, nu = CASES[name]
    start = time.perf_counter()
    radii, rates, cs = sweep(D, alpha, beta, nu, dt_frac, seeds, first_seed, num_walkers, pool)
    c, c_err = cs.mean(), cs.std(ddof=1) / np.sqrt(seeds)

    save_record({"run_id": run_id, "kind": "sweep", "case": name, "D": D, "alpha": alpha, "beta": beta, "nu": nu,
                 "sigma": sigma_prod(alpha, beta, nu), "dt_frac": dt_frac, "num_walkers": num_walkers,
                 "seeds": seeds, "first_seed": first_seed, "radii": radii.tolist(), "rates": rates.tolist(),
                 "cs": cs.tolist(), "c": c, "c_err": c_err, "seconds": time.perf_counter() - start})
    return c, c_err


def extrapolate(dt_fracs, cs, c_errs):
    """
    Straight-line fit of c against sqrt(dt); returns (dt -> 0 intercept, its error, slope)
    A walker can overshoot the wall by about one step between checks, and step size ~sqrt(dt)
    """
    (slope, intercept), cov = np.polyfit(np.sqrt(dt_fracs), cs, 1, w=1 / np.asarray(c_errs), cov="unscaled")
    return intercept, np.sqrt(cov[1, 1]), slope


if __name__ == "__main__":
    import sys

    # python find_c.py --quick: small-scale test
    quick = "--quick" in sys.argv[1:]

    if quick:
        num_walkers, seeds, dt_fracs, case_dt_frac = 2000, 3, (4e-3, 2e-3, 1e-3), 1e-3
    else:
        num_walkers, seeds, dt_fracs, case_dt_frac = 10_000, 16, (4e-4, 2e-4, 1e-4, 5e-5), 1e-4

    # fresh seeds so reruns give independent data
    first_seed = int(np.random.default_rng().integers(2**31))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{first_seed}"
    start = time.perf_counter()

    with ProcessPoolExecutor() as pool:
        print(f"Convergence in dt (case A, N = {num_walkers}, {seeds} seeds):", flush=True)
        conv_cs, conv_errs = [], []
        for i, dt_frac in enumerate(dt_fracs):
            c, c_err = run_sweep("A", dt_frac, seeds, first_seed + 10_000 * i, num_walkers, pool, run_id)
            conv_cs.append(c)
            conv_errs.append(c_err)
            print(f"  dt = {dt_frac:.0e} tau   c = {c:.4f} +/- {c_err:.4f}", flush=True)

        c_0, c_0_err, slope = extrapolate(dt_fracs, conv_cs, conv_errs)
        print(f"  straight line in sqrt(dt) -> dt = 0:  c = {c_0:.4f} +/- {c_0_err:.4f}", flush=True)

        print(f"Cases (dt = {case_dt_frac:.0e} tau): same c means only sigma and D matter", flush=True)
        print("  case  D     alpha  beta  nu    sigma  c")
        case_cs = {}
        for j, (name, (D, alpha, beta, nu)) in enumerate(CASES.items()):
            c, c_err = run_sweep(name, case_dt_frac, seeds, first_seed + 100_000 * (j + 1), num_walkers, pool, run_id)
            case_cs[name] = (c, c_err)
            print(f"  {name:<5} {D:<5g} {alpha:<6g} {beta:<5g} {nu:<5g} {sigma_prod(alpha, beta, nu):<6g} "
                  f"{c:.4f} +/- {c_err:.4f}", flush=True)

    save_record({"run_id": run_id, "kind": "summary", "quick": quick, "num_walkers": num_walkers, "seeds": seeds,
                 "dt_fracs": list(dt_fracs), "conv_cs": conv_cs, "conv_errs": conv_errs,
                 "c_0": c_0, "c_0_err": c_0_err, "slope": slope, "case_dt_frac": case_dt_frac,
                 "case_cs": case_cs, "seconds": time.perf_counter() - start})
    print(f"c = {c_0:.4f} +/- {c_0_err:.4f}   (pi = {np.pi:.4f})")
    print(f"Saved to {RESULTS_PATH} (run_id {run_id}, {time.perf_counter() - start:.0f} s)")
