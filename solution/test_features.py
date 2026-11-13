import numpy as np

from walkers import generate_walkers, diffuse, remove_external_walkers

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

def test_remove_external_walkers():
    rng = np.random.default_rng(seed=535)
    num_walkers = 20000
    R = 1.0
    walkers = generate_walkers(num_walkers, R, rng)

    # walkers outside the sphere
    walkers[0] = np.array([2.0, 0.0, 0.0])
    walkers[1] = np.array([-2.0, 0.0, 0.0])

    walkers_after_removal = remove_external_walkers(walkers, R)

    assert walkers_after_removal.shape[0] == num_walkers - 2, "Incorrect number of walkers after removal"
    assert np.all(np.linalg.norm(walkers_after_removal, axis=1) < R), "Walkers outside the sphere remain after removal"

def test_remove_external_walkers_edge_cases():
    # R = 5 so (3, 4, 0) sits exactly on the surface with no rounding error,
    # while each of its coordinates is < R (catches per-coordinate checks)
    R = 5.0
    walkers = np.array([
        [0.0, 0.0, 0.0],     # origin: keep
        [4.999, 0.0, 0.0],   # just inside: keep
        [2.0, 2.0, 2.0],     # inside on a diagonal : keep
        [3.0, 4.0, 0.0],     # exactly at R, off-axis: remove (boundary counts as escaped)
        [0.0, 0.0, -5.0],    # exactly at R, on-axis: remove
        [4.0, 4.0, 0.0],     # just outside: remove
        [10.0, 10.0, 10.0],  # far outside: remove
    ])

    walkers_after_removal = remove_external_walkers(walkers, R)
    assert np.array_equal(walkers_after_removal, walkers[:3]), "Wrong walkers kept"

    # Empty population comes back empty with the right shape
    empty = remove_external_walkers(np.zeros((0, 3)), R)
    assert empty.shape == (0, 3), "Empty input should return shape (0, 3)"

if __name__ == "__main__":
    test_generate_walkers()
    test_remove_external_walkers()
    test_remove_external_walkers_edge_cases()

    test_diffuse_basic()
    test_diffuse_evolve()
    
    print("Tests passed")