"""
Verify c against the full system: diffusion + ABSORPTION + BRANCHING + escape

At R = c * sqrt(D / sigma) the growth rate is ~0 for very different params (alpha, beta, nu) that share the same sigma
Goal: Sweep R, fit growth = a - b / R^2, and check a ~ sigma and b / D ~ Lambda
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from walkers import generate_walkers, diffuse, remove_external_walkers, resample, fit_growth_rate
from estimate_c import save_record, combine_results, RESULTS_PATH as ESTIMATE_C_PATH

VERIFY_PATH = Path(__file__).parent / "results" / "verify.jsonl"

# discrete-time walkers see the wall shifted out by WALL_SHIFT * sqrt(2 D dt)
WALL_SHIFT = 0.5826


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


def growth_rate(R, D, alpha, beta, nu, num_walkers=10_000, dt_frac=5e-5, seed=0, dt=None, t_burn=None, t_meas=None):
    """
    Long-time growth rate of the full system at radius R
    Same loop as escape_rate, with interact() after remove_external_walkers()

    dt, t_burn, t_meas default to dt_frac, 0.3 and 1.0 times tau = R^2 / D
    """
    tau = R**2 / D
    if dt is None:
        dt = dt_frac * tau
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
        walkers = remove_external_walkers(walkers, R)
        walkers = interact(walkers, alpha, beta, nu, dt, rng)
        walkers, log_w = resample(walkers, num_walkers, log_w, rng)

        if step >= n_burn:
            ts.append((step + 1) * dt)
            log_pop.append(log_w + np.log(len(walkers) / num_walkers))

    return fit_growth_rate(ts, log_pop)


def mean_growth_rate(R, D, alpha, beta, nu, seeds, first_seed, num_walkers, dt_frac):
    """
    growth_rate averaged over seeds; returns (mean, standard error of the mean)
    """
    rates = [growth_rate(R, D, alpha, beta, nu, num_walkers=num_walkers, dt_frac=dt_frac, seed=seed)
             for seed in range(first_seed, first_seed + seeds)]
    return np.mean(rates), np.std(rates, ddof=1) / np.sqrt(seeds)


def critical_radius(D, alpha, beta, nu, radii, seeds=2, first_seed=0, num_walkers=10_000, dt_frac=5e-5):
    """
    Measure growth_rate at each radius, fit growth = a - b / R^2, return (R_c = sqrt(b / a), a, b)
    Expect a ~ sigma and b / D ~ Lambda (dt_frac is the same at every R, so is the time-step bias)
    """
    rates = [mean_growth_rate(R, D, alpha, beta, nu, seeds, first_seed, num_walkers, dt_frac)[0] for R in radii]
    slope, a = np.polyfit(1 / np.asarray(radii)**2, rates, 1)
    b = -slope
    return np.sqrt(b / a), a, b


def best_c(path=ESTIMATE_C_PATH):
    """
    c from all large-scale runs saved by estimate_c.py, combined
    """
    c, c_err, _ = combine_results(path=path)
    return c, c_err


# Parameter sets chosen with same sigma, very different mixes,
# plus a change in D and a change in sigma.
CASES = [
    # case_name,   D,   alpha, beta, nu
    ("A",   1.0, 0.5,   1.0,  2.5),   # sigma = 1, moderate
    ("B",   1.0, 0.0,   0.5,  3.0),   # sigma = 1, no absorption
    ("C",   1.0, 2.0,   1.0,  4.0),   # sigma = 1, heavy branching vs heavy absorption

    ("D",   2.0, 0.5,   1.0,  2.5),   # doubled D
    ("E",   1.0, 0.5,   2.0,  2.0),   # sigma = 1.5
]


if __name__ == "__main__":
    # True: small-scale test
    # False: long running for a good verification
    quick = True

    if quick:
        num_walkers, dt_frac, seeds = 2000, 1e-3, 3
    else:
        num_walkers, dt_frac, seeds = 10000, 5e-5, 4

    first_seed = int(np.random.default_rng().integers(2**31))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{first_seed}"
    start = time.perf_counter()

    c, c_err = best_c()
    # at R = c sqrt(D / sigma) the walkers' escape rate is Lambda(dt) D / R^2 with Lambda(dt) < c^2,
    # so the measured growth should be slightly positive: (1 - 1 / (1 + shift)^2) * sigma

    # expected dt bias from discrete time step walkers
    shift = WALL_SHIFT * np.sqrt(2 * dt_frac)

    # interactions multiply by (1 + sigma dt) per step, a rate of log(1 + sigma dt) / dt;
    # at R_c, tau = c^2 / sigma so sigma dt = c^2 dt_frac
    branch_step = c**2 * dt_frac
    bias = np.log(1 + branch_step) / branch_step - 1 / (1 + shift)**2
    print(f"c = {c:.4f} +/- {c_err:.4f} from estimate_c.py;  expected dt bias in growth: +{bias:.4f} sigma")

    print("Check 1: growth rate at the predicted R_c")
    print("  case  D     alpha  beta  nu    sigma  R_c     growth / sigma")
    for i, (name, D, alpha, beta, nu) in enumerate(CASES):
        sigma = beta * (nu - 1) - alpha
        R_c = c * np.sqrt(D / sigma)

        # separate seeds per case
        case_seed = first_seed + 1000 * i
        rate, rate_err = mean_growth_rate(R_c, D, alpha, beta, nu, seeds, case_seed, num_walkers, dt_frac)
        print(f"  {name:<5} {D:<5g} {alpha:<6g} {beta:<5g} {nu:<5g} {sigma:<6g} {R_c:<7.4f} "
              f"{rate / sigma:+.4f} +/- {rate_err / sigma:.4f}")
        save_record({"run_id": run_id, "kind": "growth_at_R_c", "case": name, "D": D, "alpha": alpha,
                     "beta": beta, "nu": nu, "sigma": sigma, "c": c, "R_c": R_c, "num_walkers": num_walkers,
                     "dt_frac": dt_frac, "seeds": seeds, "first_seed": case_seed,
                     "growth": rate, "growth_err": rate_err, "expected_bias": bias * sigma}, VERIFY_PATH)


    print("Check 2: sweep R and fit growth = a - b / R^2 (case A)")
    name, D, alpha, beta, nu = CASES[0]
    sigma = beta * (nu - 1) - alpha
    radii = c * np.sqrt(D / sigma) * np.array([0.8, 0.9, 1.0, 1.1, 1.25])
    R_c_fit, a, b = critical_radius(D, alpha, beta, nu, radii, seeds, first_seed, num_walkers, dt_frac)
    c_fit = R_c_fit * np.sqrt(sigma / D)

    # the sweep runs at finite dt, so compare with the biased Lambda(dt) rather than c^2
    Lambda_dt = c**2 / (1 + shift)**2
    print(f"  a = {a:.4f} (sigma = {sigma:g})   b / D = {b / D:.4f} (Lambda(dt) = {Lambda_dt:.4f})")
    print(f"  c from sweep = {c_fit:.4f} (sqrt(Lambda(dt)) = {np.sqrt(Lambda_dt):.4f}, c = {c:.4f})")
    save_record({"run_id": run_id, "kind": "sweep", "case": name, "radii": radii.tolist(), "a": a, "b": b,
                 "R_c": R_c_fit, "c": c_fit, "num_walkers": num_walkers, "dt_frac": dt_frac, "seeds": seeds,
                 "first_seed": first_seed, "Lambda_dt": Lambda_dt, "seconds": time.perf_counter() - start}, VERIFY_PATH)
    print(f"Saved to {VERIFY_PATH} (run_id {run_id}, {time.perf_counter() - start:.0f} s)")
