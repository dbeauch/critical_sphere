import numpy as np

from walkers import generate_walkers

def test_generate_walkers():
    rng = np.random.default_rng(seed=535)
    num_walkers = 20000
    R = 1.0
    walkers = generate_walkers(num_walkers, R, rng)

    assert walkers.shape == (num_walkers, 3), "Array shape is incorrect"

    distances = np.linalg.norm(walkers, axis=1)
    assert np.all(distances < R), "Walkers outside the sphere"

    # Uniform in volume: P(r < s) = (s/R)^3, so <r/R> = 3/4 and P(r < R/2) = 1/8.
    assert abs(np.mean(distances / R) - 0.75) < 0.01, "Mean radius is not 3R/4"

    # Standard errors at num_walkers = 20000
    assert abs(np.mean(distances < R / 2) - 0.125) < 0.01, "Inner-half fraction is not 1/8"

if __name__ == "__main__":
    test_generate_walkers()

    
    print("Tests passed")