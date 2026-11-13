import numpy as np

def generate_walkers(num_walkers, R, rng):
    """
    Returns an array of (num_walkers, 3) initializing uniformly distributed walkers within radius R
    Uses rejection sampling
    """
    walkers = rng.uniform(-R, R, size=(num_walkers, 3))

    # regenerate walkers outside radius R
    mask = np.linalg.norm(walkers, axis=1) < R
    while not np.all(mask):
        walkers[~mask] = rng.uniform(-R, R, size=(np.sum(~mask), 3))
        mask = np.linalg.norm(walkers, axis=1) < R

    return walkers

def diffuse(walkers, D, dt, rng):
    """
    Returns the walker array after one time step
    """
    displacements = rng.normal(loc=0.0, scale=np.sqrt(2 * D * dt), size=walkers.shape)
    walkers += displacements
    return walkers

def remove_external_walkers(walkers, R):
    """
    Returns walker array with walkers outside R removed
    """
    mask = np.linalg.norm(walkers, axis=1) < R
    return walkers[mask]

def resample(walkers, num_walkers, log_w, rng):
    """
    Keeps the population near num_walkers without losing track of the true growth with resampling
    Returns (walkers, log_w)
    """
    count = len(walkers)
    if count == 0:
        raise RuntimeError("Population went extinct. Cannot resample")

    # between 0.5*num_walkers and 2*num_walkers: leave alone
    if num_walkers / 2 <= count <= 2 * num_walkers:
        return walkers, log_w

    # each walker now stands for count / num_walkers times as many particles
    log_w += np.log(count / num_walkers)
    indices = rng.integers(0, count, size=num_walkers)
    return walkers[indices], log_w

def fit_growth_rate(ts, log_pop):
    """
    Least-squares slope of log(population) vs time (pass post-burn-in data only)
    """
    ts = np.asarray(ts, dtype=float)
    log_pop = np.asarray(log_pop, dtype=float)

    # slope = cov(t, y) / var(t)
    t_centered = ts - ts.mean()
    return np.sum(t_centered * (log_pop - log_pop.mean())) / np.sum(t_centered**2)