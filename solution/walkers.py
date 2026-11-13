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