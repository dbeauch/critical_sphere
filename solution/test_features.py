import numpy as np

from walkers import generate_walkers, diffuse

def test_generate_walkers():
    rng = np.random.default_rng(seed=535)
    num_walkers = 20000
    R = 1.0
    walkers = generate_walkers(num_walkers, R, rng)

    assert walkers.shape == (num_walkers, 3), "Array shape is incorrect"

    distances = np.linalg.norm(walkers, axis=1)
    assert np.all(distances < R), "Walkers outside the sphere"

    # Uniform in volume: P(r < s) = (s/R)^3
    # <r/R> = 3/4 and P(r < R/2) = 1/8
    # standard errors at num_walkers = 20000
    assert abs(np.mean(distances / R) - 0.75) < 0.01, "Mean radius is not 3R/4"
    assert abs(np.mean(distances < R / 2) - 0.125) < 0.01, "Inner-half fraction is not 1/8"

def test_diffuse_basic():
    rng = np.random.default_rng(seed=535)
    num_walkers = 20000
    R = 1.0
    D = 0.5
    dt = 0.01

    walkers = generate_walkers(num_walkers, R, rng)
    walkers_after_diffusion = walkers.copy()
    walkers_after_diffusion = diffuse(walkers_after_diffusion, D, dt, rng)

    assert walkers_after_diffusion.shape == (num_walkers, 3), "Array shape is incorrect after diffusion"

    displacements = walkers_after_diffusion - walkers
    mean_displacement = np.mean(displacements, axis=0)
    assert np.all(np.abs(mean_displacement) < 0.01), "Mean displacement is not approximately zero"

def test_diffuse_evolve():
    # Start at the origin so positions are displacements
    # after n_steps the spread should match t = n_steps * dt.
    rng = np.random.default_rng(seed=535)
    num_walkers = 20000
    D = 0.7
    dt = 0.01
    n_steps = 50
    t = n_steps * dt

    walkers = np.zeros((num_walkers, 3))
    for _ in range(n_steps):
        walkers = diffuse(walkers, D, dt, rng)

    # Per-axis variance ~ 2Dt; relative standard error ~ sqrt(2 / num_walkers) ~ 1%
    variance_per_axis = np.var(walkers, axis=0)
    assert np.all(np.abs(variance_per_axis / (2 * D * t) - 1) < 0.04), "Per-axis variance is not 2Dt"

    # <r^2> ~ 6Dt; relative standard error ~ sqrt(2/3) / sqrt(num_walkers) ~ 0.6%
    mean_r_squared = np.mean(np.sum(walkers**2, axis=1))
    assert abs(mean_r_squared / (6 * D * t) - 1) < 0.03, "<r^2> is not 6Dt"

if __name__ == "__main__":
    test_generate_walkers()

    test_diffuse_basic()
    test_diffuse_evolve()
    
    print("Tests passed")