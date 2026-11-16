"""
Measure c from pure random walkers 
Rather than simulate branching and absorption, we can model it with population = e^(sigma t) x (pure-walker survival)
Pure walkers leak at rate lambda_esc = Lambda * D / R^2, so criticality (sigma = lambda_esc) gives c = sqrt(Lambda)
"""
import numpy as np

from walkers import generate_walkers, diffuse, remove_external_walkers, resample, fit_growth_rate


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


def scaling_check(pairs=((1.0, 1.0), (2.0, 1.0), (1.0, 3.0), (0.5, 0.2)), dt_frac=1e-3, num_walkers=20_000):
    """
    check lambda = lambda_esc * R^2 / D is the same for every (R, D)
    
    dt = dt_frac * tau keeps the time-step bias identical across runs
    Each pair gets its own seed so the runs are independent
    Returns the list of lambdas
    """
    lambdas = []
    for seed, (R, D) in enumerate(pairs):
        tau = R**2 / D
        Lambda = escape_rate(R=R, D=D, dt=dt_frac * tau, num_walkers=num_walkers, seed=seed) * R**2 / D
        lambdas.append(Lambda)
        print(f"  R = {R:<4g} D = {D:<4g} Lambda = {Lambda:.3f}")
    print(f"  spread: {np.mean(lambdas):.3f} +/- {np.std(lambdas, ddof=1):.3f}")
    return lambdas


def extrapolate_dt(dts=(4e-4, 2e-4, 1e-4, 5e-5), seeds=5, num_walkers=20_000):
    """
    Check 2: remove the time-step bias (R = D = 1)

    Account for walkers leaking through the wall in between time steps and adjust lambda accordingly
    Returns (Lambda_0, its uncertainty).
    """
    runs = np.array([[escape_rate(dt=dt, num_walkers=num_walkers, seed=seed) for seed in range(seeds)] for dt in dts])
    means = runs.mean(axis=1)
    sem = np.sqrt(np.mean(runs.var(axis=1, ddof=1)) / seeds)
    for dt, mean in zip(dts, means):
        print(f"  dt = {dt:.1e}  sqrt(dt) = {np.sqrt(dt):.4f}  Lambda = {mean:.3f} +/- {sem:.3f}")

    # equal error bars
    (slope, Lambda_0), cov = np.polyfit(np.sqrt(dts), means, 1, w=np.full(len(dts), 1 / sem), cov="unscaled")
    Lambda_0_err = np.sqrt(cov[1, 1])
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

    print("Scaling check (dt = 1e-3 tau):")
    scaling_check(num_walkers=num_walkers)

    print("Time-step extrapolation (R = D = 1):")
    Lambda_0, Lambda_0_err = extrapolate_dt(dts=dts, seeds=seeds, num_walkers=num_walkers)

    # c = sqrt(Lambda), so dc = dLambda / (2 sqrt(Lambda))
    c = np.sqrt(Lambda_0)
    c_err = Lambda_0_err / (2 * c)
    print(f"Lambda_0 = {Lambda_0:.4f} +/- {Lambda_0_err:.4f}")
    print(f"c = {c:.4f} +/- {c_err:.4f}")
