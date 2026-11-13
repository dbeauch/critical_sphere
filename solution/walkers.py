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

def diffuse():
    return None