# Critical Spheres

Problem: [critical-radius-problem.svg](critical-radius-problem.svg). Particles diffuse in a sphere, are absorbed or branch, and escape at the surface. Find the radius at which the expected population stays constant.

## Answer

```math
R_{crit} = \pi\,\sqrt{\frac{D}{\beta(\nu - 1) - \alpha}}
```

## Approach

1. A scaling argument gives the form $R_{crit} = c\sqrt{D/(\beta(\nu-1)-\alpha)}$, with an unknown constant $c$.
2. A Monte Carlo simulation of the full system (diffusion, absorption, branching, escape) finds the radius where growth crosses zero, which gives $c = 3.144 \pm 0.003$.
3. Two checks: the result converges as the time step shrinks, and very different mixes of $\alpha$, $\beta$, $\nu$ and $D$ give the same $c$.
4. Compare the textbook critical radius of a bare spherical reactor, $c = \pi$.

Full writeup: [solution/writeup.md](solution/writeup.md)

## Layout

| File | Contents |
|---|---|
| `solution/walkers.py` | walker generation, diffusion, escape, resampling, growth-rate fit |
| `solution/find_c.py` | full-system simulation, radius sweep, time-step and parameter checks |
| `solution/test_features.py` | unit tests for development |
| `solution/results/find_c.jsonl` | saved results, one JSON record per sweep |

## Running

Requires Python 3 with `numpy`

```sh
cd solution
python find_c.py --quick    
python find_c.py
```
