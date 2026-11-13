# Critical Spheres Solution

## 1. Derivation

### Production
The particle production at each step is the difference in new particles produced by $\beta$ and lost particles by the absorption rate $\alpha$:

```math
\sigma_{prod} = \beta(\nu - 1) - \alpha
```

The scattering rate $\gamma$ describes how particles move randomly at each step but does not affect particle production.

### Diffusion
A particle doesn't travel in a straight line, it random walks as $\gamma$ describes. The root mean square distance $d_{rms}$ of a random walk with step size $\ell$ after $n$ steps is $\ell\sqrt{n}$. If a step happens every $\tau$, then after a time $t$ the particle has taken $n = t/\tau$ steps, so

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
This isn't a true derivation. It's done with dimensional analysis and approximation, and we dropped some constant factors along the way. It gets the form right but not the number. However, $D$ and $\sigma_{prod}$ can only combine into one length, $\sqrt{D/\sigma_{prod}}$, so whatever we dropped is a pure constant $c$:

```math
R_{crit} = c\,\sqrt{\frac{D}{\beta(\nu - 1) - \alpha}}
```

$c$ depends on the shape of the region and we can find it with a Monte Carlo method.

## 2. Computing c


### Check: escape rate scales as D / R^2


## 3. Verification against the full system
