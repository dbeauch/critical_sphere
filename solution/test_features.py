import numpy as np
import pytest

from walkers import generate_walkers, diffuse, remove_external_walkers, resample, fit_growth_rate
from find_c import interact, growth_rate, fit_critical_radius, sweep, extrapolate, CASES

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

def labeled_walkers(count):
    # returns walkers labeled with row index
    walkers = np.zeros((count, 3))
    walkers[:, 0] = np.arange(count)
    return walkers

def test_resample_preserves_population():
    # log_w + log(len / num_walkers), the log-population, remains invariant
    rng = np.random.default_rng(seed=535)
    num_walkers = 1000
    log_w = 0.7

    for count in (300, 4000):  # below and above the band [num_walkers/2, 2*num_walkers]
        walkers = labeled_walkers(count)
        resampled, new_log_w = resample(walkers, num_walkers, log_w, rng)

        assert len(resampled) == num_walkers, "Resampled population != num_walkers"
        before = log_w + np.log(count / num_walkers)
        after = new_log_w + np.log(len(resampled) / num_walkers)
        assert np.isclose(before, after), "Resampling changed log-population"

def test_resample_inside_band():
    rng = np.random.default_rng(seed=535)
    num_walkers = 1000
    log_w = 0.7

    for count in (500, 1000, 2000):  # both band edges count as inside
        walkers = labeled_walkers(count)
        resampled, new_log_w = resample(walkers, num_walkers, log_w, rng)
        assert resampled is walkers, "Population inside the band should be returned unchanged"
        assert new_log_w == log_w, "log_w should not change inside the band"

def test_resample_extinct():
    rng = np.random.default_rng(seed=535)
    with pytest.raises(RuntimeError):
        resample(np.zeros((0, 3)), 1000, 0.0, rng)

def test_fit_growth_rate_noiseless():
    # nonzero intercepts since a fit through the origin fails
    ts = np.linspace(0.3, 1.3, 1001)
    assert np.isclose(fit_growth_rate(ts, 3.0 - 9.87 * ts), -9.87), "Decaying slope not recovered"
    assert np.isclose(fit_growth_rate(ts, -2.0 + 1.5 * ts), 1.5), "Growing slope not recovered"

def test_fit_growth_rate_noisy():
    # Gaussian noise of std s: slope standard error = s / (sqrt(n) * std(t))
    rng = np.random.default_rng(seed=535)
    ts = np.linspace(0.3, 1.3, 1001)
    s = 0.05
    log_pop = 3.0 - 9.87 * ts + rng.normal(0.0, s, size=ts.size)

    standard_error = s / (np.sqrt(ts.size) * np.std(ts))
    assert abs(fit_growth_rate(ts, log_pop) - (-9.87)) < 4 * standard_error, "Slope off by more than 4 stde"

def test_interact_absorption_only():
    # alpha only: each walker removed with probability alpha * dt, nothing added
    rng = np.random.default_rng(seed=535)
    num_walkers, alpha, dt = 200_000, 50.0, 1e-3
    after = interact(np.zeros((num_walkers, 3)), alpha, 0.0, 2.0, dt, rng)
    expected = num_walkers * alpha * dt
    assert abs((num_walkers - len(after)) - expected) < 4 * np.sqrt(expected), "Absorbed fraction is not alpha * dt"

def test_interact_branching_keeps_position():
    # children start where their parent was
    rng = np.random.default_rng(seed=535)
    walkers = np.array([[1.0, 2.0, 3.0]] * 1000)
    after = interact(walkers, 0.0, 50.0, 3.0, 1e-3, rng)
    assert len(after) > len(walkers), "No walkers were added by branching"
    assert np.all(after == [1.0, 2.0, 3.0]), "A child moved away from its parent"

def test_growth_rate_no_escape():
    # R so large nothing escapes: growth = log(1 + sigma dt) / dt, almost sigma
    alpha, beta, nu = 0.5, 1.0, 2.5   # sigma = 1
    dt = 1e-3
    rate = growth_rate(R=1000.0, D=1.0, alpha=alpha, beta=beta, nu=nu, num_walkers=5000, seed=535,
                       dt=dt, t_burn=0.0, t_meas=2.0)
    assert abs(rate - np.log(1 + dt) / dt) < 0.05, "Growth without escape is not sigma"

def test_fit_critical_radius_noiseless():
    radii = np.array([2.5, 2.75, 3.0, 3.25, 3.5])
    R_crit, a, b = fit_critical_radius(radii, 1.5 - 9.87 / radii**2)
    assert np.isclose(a, 1.5) and np.isclose(b, 9.87), "Line not recovered"
    assert np.isclose(R_crit, np.sqrt(9.87 / 1.5)), "Zero crossing not recovered"

def test_extrapolate_noiseless():
    dt_fracs = np.array([4e-4, 2e-4, 1e-4, 5e-5])
    cs = 3.0 - 2.0 * np.sqrt(dt_fracs)
    c_0, c_0_err, slope = extrapolate(dt_fracs, cs, [0.01] * 4)
    assert np.isclose(c_0, 3.0) and np.isclose(slope, -2.0), "Straight line in sqrt(dt) not recovered"
    assert c_0_err > 0, "Extrapolation uncertainty is not positive"

def test_sweep_quick():
    # coarse end-to-end run: c should land near 3, a bit low from the large time step
    from concurrent.futures import ThreadPoolExecutor
    D, alpha, beta, nu = CASES["A"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        radii, rates, cs = sweep(D, alpha, beta, nu, dt_frac=2e-3, seeds=2, first_seed=535, num_walkers=2000, pool=pool)
    assert rates.shape == (2, len(radii)), "One rate per (seed, radius)"
    assert np.all(np.diff(rates, axis=1) > 0), "Growth does not rise with R"
    assert np.all((2.9 < cs) & (cs < 3.2)), "c is far from 3"

if __name__ == "__main__":
    test_generate_walkers()
    test_remove_external_walkers()
    test_remove_external_walkers_edge_cases()

    test_diffuse_basic()
    test_diffuse_evolve()

    test_resample_preserves_population()
    test_resample_inside_band()
    test_resample_extinct()

    test_fit_growth_rate_noiseless()
    test_fit_growth_rate_noisy()

    test_interact_absorption_only()
    test_interact_branching_keeps_position()
    test_growth_rate_no_escape()
    test_fit_critical_radius_noiseless()
    test_extrapolate_noiseless()
    test_sweep_quick()
    
    print("Tests passed")