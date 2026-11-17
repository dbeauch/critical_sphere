# Critical Spheres Solution

## 1. Derivation

### Production
The particle production at each step is the difference in new particles produced by $\beta$ and lost particles by the absorption rate $\alpha$:

```math
\sigma_{prod} = \beta(\nu - 1) - \alpha
```

The scattering rate $\gamma$ describes how particles move randomly at each step but does not affect particle production.

### Diffusion
A particle doesn't travel in a straight line, it random walks, as described by $\gamma$. The root mean square distance $d_{rms}$ of a random walk with step size $\ell$ after $n$ steps is $\ell\sqrt{n}$. If a step happens every $\tau$, then after a time $t$ the particle has taken $n = t/\tau$ steps, so

```math
d_{rms} = \ell\sqrt{\frac{t}{\tau}} = \sqrt{\frac{\ell^2}{\tau}\,t}
```

$\ell^2/\tau$ has units of length²/time and describes how fast a particle spreads out: the diffusion coefficient $D$. Therefore,

```math
d_{rms} \sim \sqrt{Dt}
```

Since a sphere is radially symmetric, a leaking particle has to travel a distance on the order of $R$ to escape. Setting $\sqrt{Dt} \approx R$, a typical particle escapes after a time $t \approx R^2/D$. Each particle survives about $t$ before leaking, so the fraction lost per unit time is about $1/t$ and the population leaks at a rate

```math
\sigma_{diff} \sim \frac{1}{t} \sim \frac{D}{R^2}
```

Sanity check: doubling the radius makes particles take four times as long to escape, so it leaks four times slower. 

### Criticality
The sphere reaches criticality when the production rate balances the diffusion rate.

```math
\sigma_{prod} = \sigma_{diff}
```
```math
\beta(\nu - 1) - \alpha = \frac{D}{R^2}
```

Solving for $R$, which is now the critical $R_{crit}$:
```math
R_{crit} = \sqrt{\frac{D}{\beta(\nu - 1) - \alpha}}
```

### Dimensional analysis
This isn't a true derivation. It's done with dimensional analysis and approximation, and I dropped some constant factors along the way. It gets the form right but not the number. However, $D$ and $\sigma_{prod}$ can only combine into one length, $\sqrt{D/\sigma_{prod}}$, so whatever I dropped is a pure constant $c$:

```math
R_{crit} = c\,\sqrt{\frac{D}{\beta(\nu - 1) - \alpha}}
```

$c$ depends on the shape of the region, and I find it with a Monte Carlo simulation.

## 2. Computing c

### Idea
From section 1, the population changes at a rate of production minus leakage:

```math
\text{growth}(R) = \sigma_{prod} - k\,\frac{D}{R^2}, \qquad c = \sqrt{k}
```

Instead of deriving $k$, I simulate the full system at several radii, measure the growth rate at each, and find the radius where it crosses zero. That radius is $R_{crit}$, and $c = R_{crit}/\sqrt{D/\sigma_{prod}}$. Since the growth is a straight line after a $1/R^2$ transformation, fitting a line and taking its zero gives $R_{crit}$.

### Simulation `walkers.py`, `find_c.py`
Each particle is a walker with a 3D position. At every time step $\Delta t$:

- **Diffuse**: add a Gaussian step to each coordinate with variance $2D\,\Delta t$. This is how I define $D$: the spread per axis grows as $\langle x^2\rangle = 2Dt$. Section 1 laid out the convention $\ell^2/\tau$ which matches it up to a constant.
- **Escape**: remove walkers with $r \ge R$.
- **Interact**: each walker is absorbed with probability $\alpha\,\Delta t$, or splits into Poisson($\nu$) walkers at the same spot with probability $\beta\,\Delta t$.
- **Resampling**: the population grows or shrinks exponentially, so whenever the walker count leaves the band $[N/2, 2N]$, I resample it back to $N$ walkers and add $\log(\text{count}/N)$ to a running total. That total plus $\log(\text{count}/N)$ is the true log-population, so resampling doesn't change it. This significantly improves computational efficiency in case the population explodes.

Walkers start uniformly in the sphere. I let them run for $R^2/D$, roughly the time to diffuse across the sphere (section 1), so the population settles into its shape, then fit the slope of log-population over the next $R^2/D$. That slope is the growth rate. Measuring times in units of $R^2/D$ gives every radius the same settling time, measurement window and step size relative to $R$.

The noise in log-population builds up from step to step, so the fit's own error bar isn't trustworthy. Instead, I run every sweep with 16 independent seeds, get one $c$ per seed, and report mean ± standard error.

Settings: $N = 10{,}000$ walkers, radii in $(2.5, 2.75, 3.0, 3.25, 3.5)\times\sqrt{D/\sigma_{prod}}$ and $\Delta t$ given as a fraction of $R^2/D$. I chose the radii based on a quick low-precision run that put the crossing near 3.

### Check 1: discrete time step errors
Walkers are only checked against the wall at the end of each step, so one can step out and back in without being removed. That makes the wall act slightly larger than $R$, so $R_{crit}$ comes out a little low. The overshoot is about one step size, which scales like $\sqrt{\Delta t}$, so I measure $c$ at four time steps and fit a straight line in $\sqrt{\Delta t}$ back to $\Delta t = 0$.

Case A, 16 seeds per time step:

| $\Delta t$  (units of $R^2/D$) | $c$ |
|---|---|
| $4\times10^{-4}$ | $3.0915 \pm 0.0025$ |
| $2\times10^{-4}$ | $3.1093 \pm 0.0024$ |
| $1\times10^{-4}$ | $3.1181 \pm 0.0018$ |
| $5\times10^{-5}$ | $3.1250 \pm 0.0022$ |
| **straight-line fit to $\Delta t = 0$** | $\mathbf{3.1438 \pm 0.0031}$ |

$c$ rises steadily as the step shrinks, as expected from walkers slipping past the wall less often.

### Check 2: only $\sigma_{prod}$ and $D$ matter
Section 1 claims that $\alpha$, $\beta$ and $\nu$ only matter through $\sigma_{prod} = \beta(\nu-1)-\alpha$. If that's right, the same $c$ should come out for very different mixes. Cases A–C share $\sigma_{prod} = 1$ with very different absorption and branching, D doubles the diffusion coefficient, and E changes $\sigma_{prod}$. All of them use $\Delta t = 10^{-4}\,R^2/D$, so they share the same time-step bias and should agree with each other, not with the extrapolated value.

| case | $D$ | $\alpha$ | $\beta$ | $\nu$ | $\sigma_{prod}$ | $c$ |
|---|---|---|---|---|---|---|
| A | 1 | 0.5 | 1 | 2.5 | 1 | $3.1154 \pm 0.0012$ |
| B | 1 | 0 | 0.5 | 3 | 1 | $3.1153 \pm 0.0014$ |
| C | 1 | 2 | 1 | 4 | 1 | $3.1231 \pm 0.0035$ |
| D | 2 | 0.5 | 1 | 2.5 | 1 | $3.1146 \pm 0.0017$ |
| E | 1 | 0.5 | 2 | 2 | 1.5 | $3.1164 \pm 0.0017$ |

A, B, D and E agree to within $0.002$. C sits about 2 standard errors higher, which isn't unusual across five cases. Removing absorption entirely, quadrupling it, doubling $D$ or changing $\sigma_{prod}$ doesn't move $c$. This supports the Section 1 result that $R_{crit}$ depends only on $D$ and $\beta(\nu-1)-\alpha$.

## 3. Result

```math
R_{crit} = c\,\sqrt{\frac{D}{\beta(\nu - 1) - \alpha}}, \qquad c = 3.144 \pm 0.003
```

The simulation gives $c = 3.144 \pm 0.003$, suspiciously close to a well beloved constant: within $0.07\%$ of $\pi = 3.1416$, less than one standard error away. Turning to a reference (Lamarsh & Baratta, *Introduction to Nuclear Engineering*, Example 6.3), the textbook critical radius of a bare spherical reactor confirms that suspicion:

```math
R_{crit} = \pi\,\sqrt{\frac{D}{\beta(\nu - 1) - \alpha}}
```

As a side effect, this whole project is technically a way to estimate $\pi$, though a wildly inefficient one.

Reproduce with `python find_c.py` (~15 minutes on my 14 core CPU) or run with `--quick` flag for a 2-second smoke test. Each sweep is saved to `results/find_c.jsonl`.

## 4. Bonus: less than critical sphere
Only given enough material for a sphere of radius $\kappa R_{crit}$ with $0 < \kappa < 1$, it leaks particles faster than it produces them. A solution would be to surround it with a cheap reflector: a shell of material that scatters particles but absorbs very few and doesn't need to branch. Particles that leave the core can scatter in the reflector and wander back in, getting another chance to branch. That lowers the leakage, so a thick enough reflector can bring the smaller core to criticality while the expensive material stays the same.

Future simulation work: add a shell outside $R$ where walkers keep diffusing (possibly with their own $D$ and a small absorption rate) but don't branch, remove walkers only at the shell's outer edge, and sweep the shell's thickness to find how much reflector material is necessary as a function of $\kappa$.
