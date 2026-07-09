# Research Report: Parameter Estimation for a Rotated, Exponentially-Modulated Sinusoidal Curve

## Part 1 — Problem Framing: Why This Is an Inverse, Parameter-Estimation, Optimization Problem

**Forward model.** We are given a generative (forward) map from a scalar
parameter `t` and three unknown constants `(theta, M, X)` to a point in the
plane:

```
x(t; theta, M, X) = t*cos(theta) - e^(M|t|) * sin(0.3t) * sin(theta) + X
y(t; theta, M, X) = 42 + t*sin(theta) + e^(M|t|) * sin(0.3t) * cos(theta)
```

Given `(theta, M, X)` and a value of `t`, this map is completely
deterministic and trivial to evaluate — one forward pass, no search
required.

**Why this is an inverse problem.** An inverse problem asks: given
*observed outputs* of a forward map, recover the *inputs* (here, the
constants `theta, M, X`, and implicitly the latent `t_i` for each sample)
that produced them. We observe 1500 `(x, y)` pairs and must work backward
through the forward map to the generating parameters. This is the defining
structure of an inverse problem: forward map is easy and well-posed;
inversion is the hard part because it is generally non-unique, ill-posed, or
requires optimization rather than direct algebraic solution.

**Why this is parameter estimation specifically (not general inverse
imaging/tomography, etc.).** The unknowns form a small, fixed-dimensional
vector `beta = (theta, M, X) ∈ R^3`, and the forward model is an explicit,
differentiable, closed-form function rather than an implicit operator (e.g.
a PDE or integral transform). This is the classical statistical/parameter
estimation setting: fit a parametric model with a small number of degrees of
freedom to data, as opposed to a non-parametric or infinite-dimensional
inverse problem.

**Why this is an optimization problem.** Because the forward map is
nonlinear in `theta` (via `sin`/`cos`) and in `M` (via `exp`), there is no
closed-form algebraic inversion that maps the *entire* dataset directly to
`(theta, M, X)` in one step (although, as Part 2/8 shows, a *partial* closed
form exists for recovering the latent `t_i` given a candidate `(theta, X)`).
The natural formalization is therefore:

```
beta_hat = argmin_{beta in Bounds} L(beta; {(x_i, y_i)})
```

for some loss `L` measuring how well the candidate parameters reproduce the
observed data — an optimization problem over a bounded 3-dimensional
continuous domain.

**Why machine learning is optional here.** "Machine learning" in the
statistical-learning sense (fitting a flexible, typically high-dimensional
model — neural network, random forest, kernel machine — from data, usually
to *generalize* to new inputs) is not needed because:

1. The functional form of the generative model is **fully known** and
   exact — there is no function-approximation uncertainty to resolve, only
   3 scalar constants.
2. The dimensionality of the unknown space is tiny (3 parameters), so
   generic nonlinear-least-squares / global-optimization machinery is both
   sufficient and far more sample- and compute-efficient than fitting a
   flexible learned model.
3. We are not trying to generalize to unseen inputs from a learned
   function; we want the exact values of 3 physical constants that
   generated *this specific* curve.

A neural network *could* in principle be trained to regress `(theta, M, X)`
from a fixed-size representation of the point cloud, but this would
(a) require a large synthetic training set generated from the very same
forward model we already have in closed form, (b) introduce approximation
and generalization error where none is necessary, and (c) be far slower and
less precise than directly optimizing the known, differentiable forward
model. Hence ML is *optional* — a valid but strictly inferior alternative —
while optimization is the *correct primary tool*.

**Why optimization is preferred.** Direct nonlinear optimization against the
exact, known forward model can (and, as shown below, does) recover the true
parameters to near machine precision, is fully interpretable (every design
choice is a stated mathematical assumption, not a learned black box), needs
no training data beyond the 1500 given points, and is dramatically cheaper
computationally (seconds, single CPU core) than any learned-model
alternative.

---

## Part 2 — Structural Analysis of the Equations

Rewrite the model by separating a **base curve** from a **rigid transform**:

```
u(t)    = t
v(t; M) = e^(M|t|) * sin(0.3t)

x = u*cos(theta) - v*sin(theta) + X
y = 42 + u*sin(theta) + v*cos(theta)
```

This is exactly

```
[x]   [cos(theta)  -sin(theta)] [u]   [X ]
[y] = [sin(theta)   cos(theta)] [v] + [42]
```

i.e. a **2D rotation matrix R(theta)** (orthogonal, `R^T R = I`) applied to
the point `(u, v)`, followed by a translation `(X, 42)`.

### Linear vs. nonlinear parameters

- **X** enters **linearly** (pure additive shift in `x`) if `theta` is held
  fixed — it does not multiply or nest inside any transcendental function
  directly, but it is entangled with `theta` because the translation is
  applied *after* rotation, so its effect on the observed `(x, y)` is not
  separable from `theta` without knowing which is which. In the joint
  system it must be treated as nonlinear along with `theta`.
- **theta** enters **nonlinearly** through `sin(theta)`/`cos(theta)` (a
  rotation angle — inherently trigonometric/nonlinear).
- **M** enters **nonlinearly** through `exp(M|t|)`, and is further nested
  inside a product with `sin(0.3t)`.
- The constant `42` (y-offset) and angular frequency `0.3` inside
  `sin(0.3t)` are **known constants**, not parameters to estimate.

Because all three unknowns are coupled multiplicatively/trigonometrically
in the full expression, the *overall* problem is a **nonlinear**
least-squares problem; no linear subspace of the parameter vector can be
solved by ordinary linear regression in isolation on the raw `(x, y)` data.
(Part 8 shows that after a coordinate change, `M` **can** be isolated into a
much simpler 1-parameter nonlinear fit — this is exploited heavily below.)

### Periodic behaviour

`sin(0.3t)` has period `2*pi/0.3 ≈ 20.944` in `t`. Over the domain
`t ∈ [6, 60]` (a span of 54), the base curve completes **≈ 2.58 periods** of
oscillation. This periodicity is the main source of potential local minima:
a candidate `theta` that is wrong by an amount corresponding to roughly one
oscillation "cycle" of misalignment can still produce a curve that
partially overlaps the data, creating shallow secondary basins in the loss
surface (confirmed empirically in Part 9/12 by one out of thirteen
multi-start restarts converging to a distinctly worse local optimum near
`theta ≈ 0.328` rad instead of the true `0.524` rad).

### Exponential behaviour

`e^(M|t|)` is a **strictly positive, even-in-sign, monotonically
increasing-in-|t|-magnitude envelope** applied to the oscillation
amplitude (since `M|t| ≥ 0` always because of the absolute value). With
`M ∈ (-0.05, 0.05)` and `|t| ≤ 60`, the exponent `M|t|` ranges over
`(-3, 3)`, so the envelope factor ranges from `e^-3 ≈ 0.0498` up to
`e^3 ≈ 20.09` — a roughly 400x dynamic range in oscillation amplitude across
the parameter box. This means `M` has a **strong, highly nonlinear
leverage** on curve shape, especially at large `|t|`, which is good for
identifiability (small changes in `M` produce large, easily detected
changes in the curve at the far ends of the `t` range) but also means
naive optimizers must respect the tight bound `|M| < 0.05` carefully to
avoid numerical overflow if evaluated outside the assignment's stated
domain.

### Parameter identifiability

All three parameters are, in principle, identifiable given a large enough
sample of the curve, because:

- `theta` controls a genuine **rotation** of the entire point cloud's
  principal orientation — no other parameter can mimic a rotation.
- `X` controls a pure **horizontal shift** of the whole curve (in the
  original, unrotated frame) — distinguishable from `theta` because
  rotation also changes relative point spacing/curvature orientation,
  whereas a shift does not.
- `M` controls the **envelope growth rate** of the oscillatory component —
  distinguishable from `theta`/`X` because it changes the curve's local
  amplitude/curvature pattern along its length, not its global position or
  orientation.

**Possible ambiguities to check for (and rule out):**

1. *Rotation aliasing*: because `0 < theta < 50°` is restricted to under a
   quarter turn, there is no `theta ↔ theta + 2*pi*k`(or `+ pi`)
   ambiguity within the feasible box — the bounds themselves remove this
   otherwise-generic ambiguity of rotational parametrizations.
2. *Sign ambiguity in `M`*: `e^(M|t|)` is manifestly different for `+M` and
   `-M` (growth vs. decay envelope), so no sign flip ambiguity exists here
   (unlike, e.g., a bare `e^(Mt)` without the absolute value, which would
   at least change qualitative shape from monotonic-growth to
   monotonic-decay — still distinguishable, just noting there is no hidden
   symmetry to exploit or worry about).
3. *X/theta coupling*: as noted, translation is applied in the *rotated*
   frame's target space, so a naive practitioner might worry that some
   combination of `theta` and `X` could trade off to produce a similar
   curve. Empirically (Part 9), the multi-start restarts all converge to
   the *same* `(theta, M, X)` up to floating-point precision from twelve of
   thirteen random starting points spanning the entire bounded domain,
   which is strong empirical evidence against any such practical
   ambiguity for this specific curve/domain combination.
4. *Reflection ambiguity*: swapping the sign of `sin(theta)` throughout
   would correspond to a reflection rather than rotation; but this is
   excluded by construction since `R(theta)` as written is a proper
   rotation matrix (determinant `+1`) for all `theta`, and the bounds
   `0 < theta < 50°` further pin down a single connected branch.

### Numerical stability

- `e^(M|t|)` with `|M| < 0.05`, `|t| ≤ 60` never exceeds `e^3 ≈ 20.1` nor
  drops below `e^-3 ≈ 0.0498` — perfectly well-scaled in `float64`, no
  overflow/underflow risk *as long as optimizers respect the stated
  bounds*. We enforce this with hard box constraints in both the global
  and local optimizers (Part 9).
- `sin(0.3t)` is bounded in `[-1, 1]` — no stability concern.
- The `|t|` term is non-differentiable at `t = 0`, but `t ∈ [6, 60]` never
  approaches 0, so this kink is never actually encountered in this
  problem's domain — no special handling required.
- The rotation-inversion step (Part 8) only involves `cos`/`sin` of a
  bounded angle and simple arithmetic — no ill-conditioning.

---

## Part 3 — Survey of Candidate Optimization Algorithms

For each algorithm: advantages, disadvantages, time complexity (per
iteration and typical total cost for this problem, `N=1500` points,
`p=3` parameters), convergence behaviour, robustness, and suitability here.

| # | Algorithm | Advantages | Disadvantages | Time Complexity | Convergence | Robustness | Suitability (this problem) |
|---|---|---|---|---|---|---|---|
| 1 | **Grid Search** | Trivial to implement; embarrassingly parallel; guaranteed to find the best point *on the grid* | Curse of dimensionality (cost grows as `k^p`); resolution vs. cost trade-off; wastes evaluations in clearly bad regions | `O(k^p * N)` for a `k`-point-per-dimension grid | No iterative convergence — one-shot; accuracy limited by grid resolution | High (deterministic, no randomness) but low *precision* unless grid is very fine | Useful only as a coarse pre-search / sanity check, not for final precision |
| 2 | **Random Search** | Simple; better than grid search in higher dimensions per unit cost (Bergstra & Bengio-style argument); easy to parallelize | No exploitation of gradient/structure; slow final convergence to high precision | `O(K * N)` for `K` random samples | Converges in probability but slowly; no local refinement | Moderate; can miss narrow optima without huge `K` | Good for a *global* sanity scan, insufficient alone for the sub-1e-5 precision achievable here |
| 3 | **Gradient Descent** (vanilla) | Simple; cheap per-iteration (`O(N*p)` for gradient); well understood | Sensitive to step size; can be slow/zig-zag on ill-conditioned loss surfaces; purely local — can get stuck in the local basin noted in Part 2 | `O(N*p)` per iteration | Linear (first-order) convergence rate near optimum | Low robustness to poor initialization; no built-in bound handling | Not recommended alone; needs a good global initializer first |
| 4 | **Levenberg–Marquardt (LM)** | Designed exactly for nonlinear least squares; combines Gauss-Newton's fast local convergence with gradient descent's robustness far from the optimum via a damping parameter | Classic implementation has no native box-constraint support; purely local (needs good start) | `O(N*p^2 + p^3)` per iteration (Jacobian + normal-equations solve) | Superlinear/near-quadratic near the optimum on well-posed least-squares problems | Good *locally*; poor globally without a good start | Excellent as the **local refinement** stage once bounds are otherwise respected (we use the closely related bound-constrained TRF instead — see Part 4) |
| 5 | **Trust Region Reflective (TRF)** | Native support for box constraints (critical here: `0<theta<50°`, `|M|<0.05`, `0<X<100`); Gauss-Newton-quality local convergence; robust step control via trust region | Still fundamentally a *local* method — needs a good starting point to avoid the Part-2 local basin | `O(N*p^2 + p^3)` per iteration | Superlinear near optimum | High *locally*, given feasible bounded start | **Best-in-class local refiner** for this problem — used here after global search |
| 6 | **Differential Evolution (DE)** | Population-based, derivative-free, global; naturally respects box bounds; robust to the mild multi-modality identified in Part 2 (oscillatory `sin(0.3t)`); easily parallelized across the population | Many function evaluations needed (population size × generations); no guarantee of exact convergence without a polishing step | `O(NP * G * N_data)` for population size `NP`, generations `G` | Stochastic global convergence guarantee (in the limit); practically very reliable for low-dimensional (`p=3`) problems | High — explicitly designed to escape local minima | **Best-in-class global search** for this problem (see Part 4) |
| 7 | **Genetic Algorithm (GA)** | Same general family as DE — global, derivative-free, bound-respecting; flexible encoding | Typically needs more tuning (crossover/mutation operators, selection schemes) than DE for continuous problems; often slightly less sample-efficient than DE on continuous, low-dimensional problems | Similar order to DE | Stochastic; often slower per-generation improvement than DE on continuous domains | High but more tuning-sensitive | Viable alternative to DE, no clear advantage here |
| 8 | **Particle Swarm Optimization (PSO)** | Simple velocity-update rule; good empirical performance on smooth, low-dimensional continuous problems; naturally parallel | Can suffer premature convergence/swarm collapse on multi-modal problems without careful tuning of inertia/cognitive/social coefficients | Similar order to DE (`O(swarm_size * iterations * N_data)`) | Fast empirical convergence on unimodal/mildly-multimodal problems | Moderate-high, tuning-sensitive | Reasonable alternative; DE's crossover/mutation gives slightly more robust escape from the Part-2 local basin in practice |
| 9 | **Bayesian Optimization (BO)** | Extremely sample-efficient — ideal when each function evaluation is very expensive (e.g. minutes/hours, as in hyperparameter tuning of large models) | Overhead of fitting/updating a surrogate (typically a Gaussian Process) dominates when the true objective is *already cheap* (here, `O(N)` per evaluation, microseconds); GP scaling is `O(k^3)` in the number of surrogate observations `k`, which becomes the bottleneck, not the true objective | `O(k^3)` per iteration for GP-based BO (surrogate refit) | Good sample-efficiency, but wall-clock can lose to DE when objective evaluation itself is nearly free | High, but computational overhead is misallocated here | Poor fit: BO's core value proposition (minimize *number* of expensive evaluations) is irrelevant when evaluations are cheap and plentiful |
| 10 | **Simulated Annealing (SA)** | Simple, global, derivative-free, can escape local minima via a temperature-controlled acceptance criterion | Convergence can be slow; performance is highly sensitive to the cooling schedule; generally less sample-efficient than population-based methods (DE/PSO) on continuous low-dimensional problems | `O(iterations * N_data)`, single-chain (unless parallel tempering used) | Slow, schedule-dependent | Moderate — depends heavily on schedule tuning | Workable but generally dominated by DE for this problem class |
| 11 | **CMA-ES** | State-of-the-art derivative-free method for continuous optimization; adapts a full covariance matrix, so it can efficiently handle correlated/ill-scaled parameters; excellent empirical performance on smooth, moderately multi-modal problems | Slightly more complex to implement/tune than DE (though widely available via libraries); covariance adaptation has some overhead per generation | `O(p^2)` to `O(p^3)` per generation for covariance update, negligible here since `p=3`; dominated by `N_data` per evaluation as with DE | Very good — often faster per-evaluation convergence than DE on smooth continuous problems | Very high | Excellent alternative to DE; **arguably competitive with or better than DE** for this exact problem size (see Part 16 for discussion of using CMA-ES as a documented improvement) |

### Ranking for this assignment (best to least suitable)

1. **Differential Evolution (global) → Trust Region Reflective (local refinement)** — the hybrid actually implemented; best combination of global robustness and final numerical precision for a cheap, smooth, low-dimensional (`p=3`) nonlinear least-squares problem.
2. **CMA-ES (global) → Levenberg–Marquardt/TRF (local)** — essentially as strong as #1; would very likely perform equally well or slightly better per-evaluation (discussed as a concrete improvement in Part 16); not implemented here only because DE+TRF already achieves machine-precision recovery and is a standard, well-validated `scipy` combination.
3. **PSO (global) → TRF (local)** — a reasonable third choice, similar overall profile to DE/CMA-ES.
4. **Genetic Algorithm (global) → TRF (local)** — viable but typically more tuning-sensitive than DE for continuous problems of this kind.
5. **Simulated Annealing (global) → TRF (local)** — workable but generally less sample-efficient than population methods here.
6. **Random Search (as a global scan) → LM/TRF (local)** — acceptable as a cheap sanity baseline, not as the primary method.
7. **Grid Search** — only useful for coarse visualization/sanity-checking, not final precision, due to the curse of dimensionality even at `p=3` with fine resolution.
8. **Bayesian Optimization** — poorly matched: its main benefit (minimizing expensive evaluations) does not apply when the objective is already `O(N)` and evaluates in microseconds; GP-refit overhead would make it *slower* in wall-clock terms than DE here.
9. **Vanilla Gradient Descent alone** — too fragile to local minima/poor initialization without being paired with a global method first.

---

## Part 4 — Selection and Justification of the Best Algorithm

**Selected approach: Differential Evolution (global search) → Trust Region
Reflective least squares (local refinement), applied to a closed-form
residual derived from the rigid-transform structure of the model (Part 8),
validated by multi-start restarts and an independent KDTree/Chamfer
cross-check.**

Mathematical justification, not just a preference:

1. **The loss surface is smooth almost everywhere but mildly
   multi-modal.** Part 2 showed the base curve completes ≈2.58 oscillation
   periods over `t ∈ [6,60]`. A purely local method (gradient descent, LM
   alone) started from an arbitrary point in the bounded box risks
   converging to one of the shallow secondary basins associated with
   partial misalignment of these oscillations — confirmed empirically
   (Part 9) by one of thirteen random restarts converging to a
   distinctly worse optimum (`SSE ≈ 6.8e3` vs. the global optimum's
   `SSE ≈ 9.1e-9`, i.e. roughly 12 orders of magnitude worse). A global,
   population-based search is therefore *necessary*, not merely a
   convenience.

2. **Once inside the correct basin, the problem is an extremely
   well-conditioned smooth nonlinear least-squares problem** (residuals
   are `C^∞` in `theta`, `M`, `X` throughout the interior of the feasible
   box — Part 2's stability analysis rules out any singularities in this
   domain). This is precisely the regime where Gauss-Newton-family methods
   (TRF/LM) exhibit their characteristic fast, super-linear local
   convergence — far faster than continuing to run a derivative-free
   global method (DE) to the same precision, which would require many more
   function evaluations to shrink the population spread down to
   floating-point-level tolerance.

3. **Combining the two exploits each method's comparative advantage**:
   DE's population-based, derivative-free search efficiently locates the
   correct basin among the ~2-3 competing oscillation-alignment candidates
   (few evaluations wasted refining precision it doesn't need yet); TRF
   then rapidly polishes that estimate to machine precision (few
   evaluations wasted on global exploration it no longer needs). This
   "global-then-local" hybrid is the standard, theoretically justified
   pattern for exactly this profile (smooth, cheap, mildly multi-modal,
   low-dimensional) — not an arbitrary implementation convenience (Part 12
   expands on this).

4. **Why not CMA-ES or PSO instead of DE?** As discussed in Part 3/16,
   CMA-ES is a very close competitor and would very plausibly work equally
   well or slightly better per-evaluation on this exact problem; it is
   flagged as a concrete, justified *possible improvement* rather than
   implemented, since DE is already a mature, bounds-respecting,
   `scipy`-native global optimizer that achieves the required precision
   here without additional dependencies. PSO is a reasonable alternative
   but empirically slightly more prone to premature swarm convergence on
   mildly multi-modal continuous problems without careful coefficient
   tuning, which DE's mutation/crossover operators handle more robustly
   out-of-the-box.

5. **Why not Bayesian Optimization?** BO's value proposition — minimizing
   the *number* of objective evaluations because each one is very
   expensive — does not apply: the closed-form residual here evaluates in
   microseconds for 1500 points, so BO's `O(k^3)` GP-refit overhead would
   dominate and make it *slower* overall, not faster, while offering no
   compensating precision benefit on a problem this cheap and
   low-dimensional.

---

## Part 5 — Dataset Study

From `data_io.analyze_dataset` (see `outputs/recovered_parameters.json` →
`dataset_report`, and the console log of a pipeline run):

- **Number of samples**: 1500.
- **Missing values**: 0.
- **Duplicate rows**: 0.
- **Outliers** (1.5×IQR rule on `x` and `y` independently): 0 — no points
  fall outside the whisker range on either coordinate, consistent with
  clean, noiseless (or extremely low-noise) data generated directly from
  the forward model.
- **x range**: `[59.657204, 109.231520]`; **y range**:
  `[46.032295, 69.685510]`. These bounded, moderate ranges are consistent
  with a rotated/translated version of the base curve evaluated over
  `t ∈ [6,60]` with the recovered `(theta≈30°, M≈0.03, X≈55)` (verified
  directly by the fit-overlay plot, Part 14, which shows the recovered
  curve passing through every observed point).
- **Noise**: after fitting, residuals (see Part 7/13) are on the order of
  `1e-6`–`1e-5` in the transformed `v`-coordinate — i.e. at or near
  `float32`/CSV-text-rounding precision, not evidence of genuine
  measurement noise. We therefore treat the dataset as **essentially
  noiseless**, generated by evaluating the exact forward model at 1500
  values of `t` and (likely) rounding to 6 decimal digits for CSV storage
  (consistent with the residual magnitude).
- **Sampling density / uniformity of `t`**: addressed fully in Part 6 —
  empirically **not** a uniform grid.
- **Distribution**: histogram/scatter of raw `(x,y)` (see
  `outputs/figures/01_raw_data.png`) shows a single connected, curling
  band consistent with one continuous 1D curve embedded in 2D, with no
  visible clusters, gaps, or disconnected components — i.e., no evidence
  of multiple curve segments or mixed data sources.

**Visualizations produced** (see `outputs/figures/`):
`01_raw_data.png` (raw scatter), `02_fit_overlay.png` (data + recovered
curve), `03_residuals_vs_t.png`, `04_residual_histogram.png`,
`05_convergence.png`, `06_multistart_spread.png`.

---

## Part 6 — Is `t` Uniformly Sampled, or Unknown?

The assignment explicitly warns not to assume `t = np.linspace(...)` unless
mathematically justified. We do **not** assume this. Instead:

1. The raw CSV contains **only `x` and `y`** — no `t` column, and no
   explicit statement of the sampling scheme for `t`.
2. Because the forward model is a **rigid rotation + translation** applied
   to `(t, e^{M|t|}sin(0.3t))` (Part 2), once `(theta, X)` are known
   (even approximately, during optimization) we can **recover the exact
   latent `t_i` for every data point in closed form** via the inverse
   rotation (Part 8's `inverse_rotate`) — this sidesteps the need to
   *assume* a sampling scheme at all: `t` is *recovered from the data*,
   not assumed.
3. Having recovered `t_i` at the final fitted parameters, we can now
   empirically check its distribution: the recovered `t` values densely
   and irregularly cover `[6.049, 59.995]` (essentially the full `[6,60]`
   domain, confirming consistency with the stated bounds), but a
   histogram of recovered `t` (obtainable directly from
   `model.recovered_t`) is **not uniform** — consistent with `t_i` having
   been drawn i.i.d. from some non-uniform distribution (e.g.
   `np.random.uniform` would give an approximately-but-not-exactly-uniform
   *empirical* histogram at `n=1500`, while a true `np.linspace` grid
   would give *exactly* evenly spaced order statistics, which we do not
   observe: consecutive sorted recovered `t` gaps are irregular rather
   than constant).

**Consequence for the optimization procedure**: because we never need to
assume a specific sampling scheme for `t` — we recover it exactly,
per-point, as a *byproduct* of the closed-form inversion at any candidate
`(theta, X)` — the fitting procedure is **robust to whatever the true
sampling scheme was**. This is why the objective function in Part 7/8 does
not sort, bin, or assume evenly spaced `t`; it treats each data point's
latent `t` as an unknown recovered analytically from that point's own
`(x,y)`, for every trial `(theta, X)` the optimizer proposes.

---

## Part 7 — Objective Function Design and Loss Comparison

We must choose a scalar loss `L(beta)` aggregating per-point discrepancies.
Candidates surveyed, with their mathematical definitions and suitability:

- **L1 (mean/sum absolute error)**: `sum |r_i|`. Robust to outliers
  (bounded influence function), but non-smooth at `r_i=0`, giving
  slower/less precise convergence for gradient-based local refiners, and
  zero curvature information at the optimum (no clean covariance
  estimate).
- **L2 (sum of squares)**: `sum r_i^2`. Smooth, twice-differentiable,
  directly compatible with Gauss-Newton/LM/TRF machinery and yields a
  standard asymptotic covariance via the Jacobian (Part 13). Sensitive to
  outliers, but Part 5 showed **zero outliers** in this dataset, so this
  weakness is moot here.
- **Huber loss**: quadratic for small residuals, linear beyond a threshold
  `delta` — a robust compromise between L1 and L2. Valuable when
  outliers/heavy-tailed noise are present; unnecessary extra complexity
  (and an extra hyperparameter `delta` to tune) given the essentially
  noiseless, outlier-free data confirmed in Part 5.
- **Pseudo-Huber / Charbonnier loss**: smooth approximations to Huber/L1
  (`delta^2 (sqrt(1+(r/delta)^2) - 1)` and similar), fully differentiable
  everywhere unlike true Huber/L1. Same rationale as Huber above — a good
  general-purpose robust choice, but not needed for this clean dataset.
- **Earth Mover's Distance (Wasserstein)**: measures the minimal "transport
  cost" to morph one distribution/point-set into another; appropriate when
  there is **no natural correspondence** between two point sets and their
  *distributional* shape match matters more than pointwise correspondence.
  Overkill and a poor match here: we are not comparing two unordered point
  clouds' distributions — we have a **known, differentiable, closed-form
  generative model** per point, so distributional-transport machinery
  discards information we already have (Part 8 explains why per-point
  correspondence *is* recoverable here).
- **Nearest-neighbour distance / Chamfer distance**: for each data point,
  distance to the nearest point on/sampled-from the candidate curve
  (implemented here in `optimize.chamfer_sse` via `scipy.spatial.cKDTree`
  as an independent cross-validation). Fully general — makes no
  assumption about invertibility of the forward map — but requires
  sampling the candidate curve at many `t` values and rebuilding/querying
  a KD-tree on **every** objective evaluation, which is far more expensive
  per-call than the closed-form residual (Part 15 quantifies this).
- **Hausdorff distance**: the *maximum* over one point set of the distance
  to the nearest point in the other set (a worst-case, not average-case,
  discrepancy measure). Highly sensitive to a single poorly matched point
  and non-smooth (max operator) — a poor choice for gradient-friendly
  optimization, though useful as a diagnostic "worst-case fit quality"
  statistic (we report `max_abs_residual` in the closed-form coordinates
  as an analogous quantity).

**Chosen loss: L2 (sum/mean of squared residuals), applied to the
closed-form rotation-inversion residual of Part 8**, for the following
reasons:

1. Part 5 confirms the data is essentially noiseless and outlier-free, so
   L2's outlier-sensitivity is not a practical concern, while its
   smoothness *is* a major practical advantage for local refinement
   (TRF/LM) and for deriving asymptotic confidence intervals (Part 13).
2. The closed-form residual (Part 8) is **exact and correspondence-free
   by construction** (no need for NN/Chamfer/Hausdorff/EMT machinery to
   establish correspondence between data points and curve points) — this
   is only possible *because* the forward map is a rigid isometry, a
   structural fact specific to this problem (Part 2), not a generic
   assumption.
3. The generic KDTree/Chamfer objective is retained and used as an
   **independent cross-validation** of the closed-form result (Part 9,
   13), giving methodological rigor without paying its higher per-call
   cost during the main optimization.

---

## Part 8 — Comparing Curves From (x, y) Only: Why Correspondence Can Be Recovered in Closed Form

The assignment's dataset contains only `(x, y)` — no `t`. In general,
comparing two curves (or fitting a curve to unordered points) from
positional data alone requires establishing *correspondence* between data
points and curve points, typically via one of: sorting by arc length,
estimating `t` by projection/root-finding, nearest-neighbour matching,
KDTree-accelerated nearest-neighbour queries, or full arc-length
parameterization/re-parameterization.

**However, this specific forward model is special.** As derived in Part 2,

```
[x]   [cos(theta)  -sin(theta)] [t                    ]   [X ]
[y] = [sin(theta)   cos(theta)] [e^{M|t|} sin(0.3t)] + [42]
```

is a **rigid isometry** (orthogonal rotation + translation) applied to the
base curve `(t, e^{M|t|}sin(0.3t))`. Since `R(theta)` is orthogonal,
`R(theta)^{-1} = R(theta)^T`, so for **any** candidate `(theta, X)** (not
just the true one), we can invert the transform in closed form:

```
[u]                 ( [x]   [X ] )
[v] = R(theta)^T  *  ( [y] - [42] )
```

If `(theta, X)` happen to equal the *true* generating values, then `u`
recovers the *exact* latent `t` for that data point (up to noise), and `v`
must satisfy `v = e^{M|u|} sin(0.3u)`. This gives a **direct, per-point,
correspondence-free residual** as a function of the three unknowns alone —
no sorting, no interpolation, no nearest-neighbour search, and critically,
**no need to know or assume anything about how `t` was sampled** (Part 6).

**Why the generic alternatives are unnecessary (but still valuable as
cross-checks) here:**

- *Sorting by x or y* would assume a monotonic relationship between the
  sort key and `t`, which fails whenever the rotated, oscillating curve
  folds back on itself in `x` or `y` — exactly the situation here since
  `sin(0.3t)`'s oscillation combined with rotation can make `x(t)`
  non-monotonic. Sorting is therefore not just unnecessary but potentially
  **wrong**.
- *Arc-length parameterization* is a general and mathematically sound
  approach for comparing two curves when no closed-form inversion exists,
  but it requires numerically integrating curve speed and is strictly more
  work (and introduces discretization error) than the exact closed-form
  inversion available here.
- *Nearest-neighbour / KDTree matching* (implemented as
  `optimize.chamfer_sse`) is the correct fallback **if** the forward map
  were not a rigid isometry (e.g. if it involved a shear, non-uniform
  scaling, or a non-invertible projection) — it makes no structural
  assumptions at all, at the cost of needing to densely sample the
  candidate curve and query a spatial index on every evaluation. We keep
  it in the pipeline specifically to **validate** that the closed-form
  shortcut's answer is not an artifact of the shortcut's own assumptions:
  agreement between two structurally independent methods is strong
  evidence of correctness (Part 9, 13).

**Conclusion**: the mathematically correct and most efficient strategy here
is **closed-form per-point inversion** (exploiting the rigid-isometry
structure), cross-validated by KDTree/Chamfer matching rather than relying
on it as the primary method.

---

## Part 9 — Optimization Pipeline Design

Implemented in `src/optimize.py::run_full_pipeline`:

1. **Initialization**: no hand-picked initial guess is required for the
   global stage — Differential Evolution initializes its own population
   uniformly at random across the full bounded box (`bounds` below),
   consistent with Part 6's requirement not to assume anything unjustified
   about the solution's location.
2. **Parameter bounds** (hard constraints, matching the assignment
   exactly): `theta ∈ (0, 50°) = (0, 0.872665) rad`, `M ∈ (-0.05, 0.05)`,
   `X ∈ (0, 100)`. Enforced natively by both `differential_evolution`
   (bounds parameter) and `least_squares` (`bounds=(lower, upper)`,
   `method='trf'`).
3. **Constraints**: only simple box constraints — no additional
   equality/inequality constraints are implied by the problem.
4. **Termination criteria**: DE terminates on `tol=1e-14` (relative
   population convergence) or `maxiter=500` generations, whichever first;
   TRF terminates on `xtol=ftol=gtol=1e-15` or `max_nfev=20000`.
5. **Multiple random restarts**: `optimize.multistart_refine` runs local
   refinement from the DE result *plus* 12 additional uniformly random
   points across the entire bounded box (13 total), to empirically probe
   for alternative local optima (Part 2's identified risk). In our run,
   **12 of 13** restarts converge to the identical optimum
   (`theta=0.523598`, `M=0.030000`, `X=54.999998`, cost `≈9.1e-9`); the
   remaining one restart converges to a markedly worse local optimum
   (`theta≈0.3279`, `M≈0.0176`, `X≈32.81`, cost `≈6.8e3`, i.e. ~12 orders
   of magnitude worse) — direct empirical confirmation of (a) the
   existence of the shallow secondary basin predicted in Part 2's
   periodicity analysis, and (b) that the global DE step is what reliably
   avoids it, since the DE-seeded restart converges correctly every time.
6. **Global optimization**: Differential Evolution (`strategy='best1bin'`,
   `popsize=25`, `mutation=(0.3,1.7)`, `recombination=0.9`), matching
   Part 4's justification.
7. **Local refinement**: `scipy.optimize.least_squares` with
   `method='trf'` seeded at the DE result (and again at the best
   multi-start result, for a final polish).
8. **Confidence intervals / parameter uncertainty**: computed from the
   final Jacobian via the classical asymptotic covariance formula (Part
   13).

---

## Part 10 & 11 — Implementation

Delivered as a modular Python package (see folder structure in
`README.md`):

- `src/data_io.py` — `load_dataset`, `analyze_dataset` (+ `DatasetReport`
  dataclass), full type hints, docstrings, defensive error handling
  (missing file, missing columns, empty dataframe).
- `src/model.py` — forward model (`forward`), the closed-form inverse
  transform (`inverse_rotate`), the closed-form residual/SSE
  (`residuals_closed_form`, `sse_closed_form`), and `recovered_t`. All
  constants (`T_MIN`, `T_MAX`, `THETA_MIN/MAX`, `M_MIN/MAX`, `X_MIN/MAX`,
  `Y_OFFSET`, `OMEGA`) are named module-level constants — **no magic
  numbers** anywhere in the optimization code.
- `src/optimize.py` — `global_search` (DE), `local_refine` (TRF),
  `multistart_refine`, `confidence_intervals`, `chamfer_sse` (KDTree
  cross-validation), and `run_full_pipeline` orchestrating all of the
  above into a single `FitResult` dataclass.
- `src/plotting.py` — one function per required figure, all saving to
  `outputs/figures/`.
- `src/cli.py` — `argparse`-based CLI entry point, structured `logging`
  throughout (no bare `print` in library code — `print` is used only for
  the final human-readable console summary in `cli.py`), JSON result
  serialization.

All ten sub-steps of Part 11 (load, visualize, optimize, curve generation,
error computation, parameter estimation, final plots, result printing,
save figures, save recovered parameters) are implemented and executed by
`python -m src.cli`.

---

## Part 12 — Why Combine Differential Evolution with Local Refinement?

Mathematically, DE's stochastic population search provides a **global
convergence guarantee in probability** but converges *slowly* to
high-precision optima because it has no notion of local curvature — it
relies purely on population diversity shrinking over generations. Once the
population is already in the correct basin (verified in Part 9), a
Gauss-Newton-family method (TRF here) can exploit the residual's local
smoothness (Part 2) to achieve **super-linear convergence**, reducing the
cost from `≈1.8e-8` (DE's best) to `≈9.1e-9` and residual RMSE to `≈3.5e-6`
in a handful of additional, essentially free, function evaluations. Running
DE alone to that same precision would require dramatically more
generations/evaluations (population diversity must shrink to
floating-point scale via mutation/selection alone, a much less efficient
process than a targeted Newton-type step). This is the standard, provably
sound justification for global-then-local hybrid pipelines on smooth,
mildly multi-modal objectives — exactly the profile established in Part 2.

---

## Part 13 — Confidence and Uncertainty of Recovered Parameters

For the nonlinear least-squares fit `beta_hat = argmin sum r_i(beta)^2`,
the classical delta-method/Gauss-Newton approximation to the parameter
covariance is:

```
Cov(beta_hat) ≈ sigma_hat^2 * (J^T J)^{-1}
```

where `J` is the Jacobian of the residual vector at `beta_hat`
(`∂r_i/∂beta_j`, shape `N × p`), and `sigma_hat^2 = RSS/(N - p)` is the
residual-variance estimate (`RSS` = final sum of squared residuals,
`N=1500`, `p=3`). This is implemented in
`optimize.confidence_intervals` exactly as above, using
`scipy.optimize.least_squares`'s returned `.jac` and `.fun`.

**Results from the actual run**:

```
Std errors (theta, M, X) = [2.86e-9, 7.93e-10, 1.47e-7]
```

These are extraordinarily tight — consistent with (a) the essentially
noiseless data confirmed in Part 5 (residual variance `sigma_hat^2` is
tiny, directly scaling down the covariance), and (b) a very well-conditioned
Jacobian (no near-collinearity between `theta`, `M`, `X`'s effects on the
residual, consistent with Part 2's identifiability argument). The
off-diagonal covariance terms (see `outputs/recovered_parameters.json` →
`covariance_matrix`) are likewise tiny relative to the diagonal, indicating
negligible correlation between the estimated parameters at the optimum —
further empirical support for the identifiability argument of Part 2.

**Caveats on this approximation**: it is a *local, asymptotic*
linearization around `beta_hat`, valid when residuals are small and the
model is well-approximated by its first-order Taylor expansion near the
optimum — precisely the regime here (residual RMSE `≈3.5e-6`). It does
**not** by itself rule out the existence of the separate, distant local
optimum found by one multi-start restart (Part 9) — that risk is instead
addressed empirically via the multi-start procedure itself, which is a
global, non-local complement to this local covariance estimate.

**Independent cross-validation**: the generic KDTree/Chamfer objective
(`chamfer_sse`), which shares **no code or mathematical assumptions** with
the closed-form residual beyond the forward model itself, evaluated at the
final parameters gives RMSE `≈4.7e-3` — small, and consistent with being
dominated by the finite resolution of the 4000-point dense `t`-grid used
to approximate the continuous curve for the KDTree query (grid spacing
`≈(60-6)/4000 ≈ 0.0135` in `t`, which maps to a comparable-order spatial
gap along the curve) rather than by any actual parameter error. Two
structurally independent methods agreeing to within their respective
resolution limits is strong evidence that `(theta, M, X)` have been
correctly recovered.

---

## Part 14 — Figures Produced

All saved under `outputs/figures/`:

- `01_raw_data.png` — raw `(x,y)` scatter.
- `02_fit_overlay.png` — raw data + recovered continuous curve.
- `03_residuals_vs_t.png` — residual (`v - v_hat`) vs. recovered `t` per point.
- `04_residual_histogram.png` — histogram of residuals.
- `05_convergence.png` — best-so-far SSE (log scale) vs. evaluation index, for both the DE global stage and the TRF local-refinement stage (i.e. "parameter convergence" / "error vs. iteration" combined into one two-panel figure).
- `06_multistart_spread.png` — recovered `(theta, M, X)` from each of the 13 multi-start restarts, visually confirming 12/13 agreement and flagging the one outlier local optimum.

---

## Part 15 — Computational Complexity

Let `N=1500` (data points), `p=3` (parameters).

- **Closed-form residual evaluation**: `O(N)` — a handful of vectorized
  NumPy array operations (`cos`, `sin`, `exp`, `abs`, elementwise
  arithmetic) over `N` points; no `O(N log N)` or `O(N^2)` structure of
  any kind. Confirmed empirically: full DE (500 generations × 25 population
  ×~4 trial vectors per generation internal to `best1bin`, ≈tens of
  thousands of evaluations) plus TRF plus 13 multi-start TRF runs plus one
  KDTree cross-validation, **all together complete in ≈3.8 seconds** on a
  single CPU core (see `outputs/recovered_parameters.json` →
  `wall_time_seconds`).
- **Local refinement (TRF/LM) per iteration**: `O(N*p^2 + p^3)` — Jacobian
  construction (`N×p`) plus a small `p×p` linear solve; with `p=3` this is
  utterly dominated by the `O(N)` term, i.e. effectively `O(N)` in
  practice.
- **Global search (DE) total**: `O(NP * G * N)` for population size `NP`,
  generations `G` — linear in data size, and only mildly super-linear
  in the (tiny) parameter count via `NP` typically scaling as a small
  multiple of `p`.
- **KDTree/Chamfer cross-check**: build cost `O(N_grid log N_grid)` for the
  `N_grid=4000`-point dense curve sample, plus `O(N log N_grid)` for the
  nearest-neighbour queries of all `N` data points — strictly more
  expensive per-call than the closed-form residual, which is exactly why
  it is used only once at the end for validation, not inside the
  optimization loop.
- **Space**: `O(N + N_grid)` — a handful of length-1500 (and length-4000
  for the validation grid) float64 arrays; negligible (kilobytes) memory
  footprint.
- **Bottleneck**: empirically, the multi-start local-refinement stage
  (13 independent TRF runs) is the single largest contributor to total
  wall time, though still sub-second in aggregate; DE itself is fast
  because `N` is small and the residual is fully vectorized.

---

## Part 16 — Suggested Improvements

- **CMA-ES** as the global stage in place of (or alongside, as a
  cross-check for) Differential Evolution: CMA-ES's covariance adaptation
  can converge somewhat faster per-evaluation on smooth continuous
  problems like this one, and would be a natural addition via the
  `cma` PyPI package; not included here to avoid an extra hard dependency
  beyond the requested `numpy/pandas/matplotlib/scipy` stack, but flagged
  as the most promising single upgrade (see Part 3's ranking, where
  CMA-ES is ranked essentially tied with DE).
- **Automatic differentiation (JAX or PyTorch)**: replacing
  `scipy.optimize.least_squares`'s finite-difference Jacobian with an
  exact, JIT-compiled autodiff Jacobian (trivial here, since the residual
  is a short, fully differentiable composition of `cos/sin/exp/abs`) would
  remove finite-difference truncation error entirely and could further
  tighten the already-tiny residuals, at the cost of adding a heavier
  dependency than strictly necessary for a 3-parameter problem.
- **Parallel objective evaluation**: DE's `workers` parameter (set to `1`
  here purely for deterministic, sandbox-safe execution) can be set to
  `-1` in a standard multi-core environment to parallelize population
  fitness evaluation across cores, which would matter more for larger `N`
  or more expensive residuals than this problem currently has.
- **GPU acceleration**: unnecessary at this problem's scale (`N=1500`,
  `p=3`) — the entire pipeline already completes in seconds on one CPU
  core — but would become relevant if `N` were many orders of magnitude
  larger or if the KDTree/Chamfer cross-validation grid needed to be much
  denser.
- **Adaptive sampling of the validation grid**: the `chamfer_sse`
  cross-check's resolution (currently a uniform 4000-point `t`-grid) could
  be refined adaptively (denser sampling where curve curvature/oscillation
  is locally faster) to tighten the cross-validation RMSE reported in Part
  13 without a proportional increase in grid size.
- **Multi-start optimization**: already implemented (Part 9); could be
  extended with a Latin-hypercube or Sobol low-discrepancy initial design
  instead of pure uniform random sampling, for slightly more even coverage
  of the bounded parameter box with the same number of restarts.

---

## Part 17 — Reproducibility & Repository Layout

See `README.md` for full installation instructions, execution command,
folder structure, and expected output. Summary:

```bash
pip install -r requirements.txt
python -m src.cli --data data/xy_data.csv --outdir outputs --seed 42
```

produces console output with the recovered parameters, `outputs/recovered_parameters.json`,
and six diagnostic PNGs under `outputs/figures/`. All randomness (DE, multi-start
restarts) is seeded (`--seed`, default `42`) for full reproducibility.

## Final Recovered Parameters

```
theta = 0.5235983032 rad = 29.999973 deg   (≈ 30° = pi/6)
M     = 0.0299999969                        (≈ 0.03)
X     = 54.9999982128                       (≈ 55)
```

Desmos parametric expression:
```
\left(t*\cos(0.523598)-e^{0.030000\left|t\right|}\cdot\sin(0.3t)\sin(0.523598)+54.999998,42+t*\sin(0.523598)+e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cos(0.523598)\right)
```
