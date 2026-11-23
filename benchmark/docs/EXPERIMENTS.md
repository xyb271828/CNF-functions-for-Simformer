# Experiment design and findings

## Task and provenance

Estimate the five-dimensional distribution of atomic positions/velocities given
nine beam and magnetic-field controls in 2D MOT simulation data. This benchmark
is a new retrospective reproduction and extension. Its measurements are not
assertions about results obtained during the original research appointment.

The historical helper remains at repository root. The revised implementation is
in `benchmark/src/`; it uses audited joins and simulation-disjoint evaluation.
The uploaded Simformer material describes a three-variable toy example and is
not treated as a comparable five-dimensional cold-atom baseline.

## Data audit

| Stage | Atom records |
|---|---:|
| Source file | 516,218 |
| Exact duplicates removed | 258,109 |
| Deduplicated records | 258,109 |
| After spatial cut | 258,041 |
| Excluded: ambiguous parameter ID | 224,540 |
| Excluded: missing parameter ID | 77 |
| Retained | 33,424 |

The parameter table contains a simulation-ID reset. Across that table, 146,446
IDs map to multiple distinct nine-control configurations. The retained join is
validated as many-to-one. Unique keys are consistent within the supplied tables;
they do not independently prove physical provenance. The retained 12,478 IDs have
no duplicate control vectors across different IDs and no configuration crossing
the train/validation/test boundaries.

| Split | Records | Simulations |
|---|---:|---:|
| Train | 23,195 | 8,734 |
| Validation | 4,857 | 1,872 |
| Test | 5,372 | 1,872 |

Only about 13% of deduplicated atom records are retained. Selection bias limits
generalization to excluded runs; restoring original batch/run identifiers is the
next data-engineering priority. Raw input files are not bundled. Exact source
hashes and split seeds are available in the audit/configuration files. Retained
split membership, row-level losses and observed/generated sample archives are
regenerated locally and are not distributed in this public repository. Model
archives are also excluded; training and checkpoint-loading code is included.

## Models and fixed budgets

- Full-covariance Gaussian: weighted fit.
- Gaussian mixture: validation-selected 4/8 components on group-balanced resampling.
- Conditional Gaussian: two 64-unit hidden layers, diagonal covariance.
- Unconditional/conditional spline flows: eight rational-quadratic spline blocks,
  two 64-unit hidden layers per coupling network and LU permutations.
- Conditional Flow Matching: independent normal-noise/data pairing with the path
  `x_t=(1-t)z+t*x` and velocity target `x-z`; context remains fixed. Four 224-unit
  SiLU hidden layers and time Fourier features, 157,701 trainable parameters.

No minibatch OT pairing is implemented. The term conditional spline flow is used
for the original discrete invertible model to avoid confusing it with the
continuous normalizing flow trained by Flow Matching.

All neural runs use seeds 11/22/33, AdamW, initial learning rate 0.001, weight decay
0.0001, batch size 512, gradient clipping and a cosine schedule. Spline and FM
primary runs have at most 30 epochs; an explicitly separate FM extension has at
most 120 epochs, with early stopping allowed. Schedule length changes with the
budget; the longer run is not continuation of an identical optimization path.

Spline checkpoints minimize validation group NLL. FM checkpoints minimize
validation group-weighted CFM MSE using four shared, fixed Monte Carlo noise/time
draws. These selection criteria and network structures differ. All runs and
budgets remain in the record; no best test seed is substituted for the mean.

## Likelihood and uncertainty

The primary metric averages atom NLL within simulation and then averages across
simulations. It is measured in nats per observation in train-standardized
`(X_mm,Y_mm,Vx,Vy,log1p(Vz))` coordinates. NLL numbers are comparable only within
this representation and cohort. They cannot be compared with historical NLL 3.11.

FM training MSE is not NLL. FM likelihood is evaluated by reversing the velocity
ODE with RK4, using the exact trace of the five-dimensional spatial Jacobian.
The trace is exact; ODE integration remains numerical. Each run compares 64/128
steps on a fixed validation subset before reporting full-test NLL at 128 steps.
All six FM runs passed the recorded convergence thresholds. Analytic translation
and scaling fields test integration direction and the log-density volume term.

Paired bootstrap intervals resample test simulations after seed-averaging losses.
They capture test-group sampling uncertainty, not all optimization, cohort or
provenance uncertainty. Across-seed sample SD is reported separately.

## Recorded comparison

| Setting | Test group NLL, mean ± SD | Mean marginal W1 |
|---|---:|---:|
| Conditional spline, maximum 30 epochs | 3.5517 ± 0.1381 | 0.0356 |
| FM, maximum 30 epochs | 3.8695 ± 0.0021 | 0.0436 |
| FM, maximum 120 epochs | 3.2431 ± 0.0212 | 0.0367 |

The spline-to-extended-FM NLL difference is 0.3086 nats; the paired 95% interval is
[0.2682, 0.3520]. At 30 epochs FM does not outperform the spline baseline in NLL.
The extended setting improves likelihood but does not lead every marginal metric.

On the recorded CPU, two threads and a 512-sample batch, direct spline sampling
takes approximately 19–21 ms versus approximately 147 ms for FM RK4-32. Euler and
RK4 sweeps with common initial noise are included. Each RK4 step evaluates the
velocity network four times. These are offline batch timings; no end-to-end
serving latency or physical-simulator speedup is claimed. Training wall times are
recorded as reference measurements from separate runs, not a controlled speedup.

## Generative diagnostics

Common-bin marginal JS uses training-defined edges and overflow bins for all
three conditional model families. Additional diagnostics include weighted
Wasserstein distances, KS effect sizes, correlations, 64-direction sliced
Wasserstein on targets and on concatenated standardized context/targets,
context permutation, physical support checks, and marginal interval coverage.

Coverage uses 256 fixed test simulations, one observed atom per simulation and
128 generated samples per condition. This is a finite-sample marginal diagnostic,
not joint coverage. Neither marginal similarity nor a sliced projection proves
full conditional fidelity. Generated samples are not clipped/rejected to improve
reported metrics. The model does not predict total atom yield or zero-atom runs.

## Interpretation and next steps

The existing shared test set was reused for the FM extension, and the 120-epoch
budget was added after inspecting the 30-epoch result. These are exploratory
results. Further tuning should use validation data and reserve an independent
confirmation set. No claim is made about the optimum achievable performance of
all FM paths, couplings, networks or optimizers.

Recover batch/run provenance, evaluate parameter-range or source-batch holdout,
obtain repeated simulator samples at fixed controls, and measure the original
simulator on the same hardware before making physical-surrogate claims.

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- [Official ODE solver and likelihood documentation](https://facebookresearch.github.io/flow_matching/generated/flow_matching.solver.ODESolver.html).

The local PyTorch ODE implementation follows the mathematical definition and is
tested independently; it does not call the official solver.
