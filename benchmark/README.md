# Reproducible cold-atom benchmark

This package records a retrospective extension of the original cold-atom
generative-modeling work. Source, aggregate evidence,
plots and execution logs are included. Source CSVs, sample archives, per-simulation
losses, row-level ID manifests and model archives are not redistributed or tracked by Git.

## Matched-data Simformer baseline (v3)

See [the full comparison](SIMFORMER_COMPARISON_ZH.md) for results and commands.
The new JAX/Haiku baseline uses the official minimal-example two-layer/two-head
architecture, specialized to nine observed controls and five generated states.
It is a standalone conditional-likelihood adaptation, not integration of flows
into Simformer or reproduction of all arbitrary-conditioning paper experiments.
Source provenance and the upstream MIT license are in `src/vendor_simformer/`.

Maximum 30-epoch NLL: Simformer **4.3517 ± 0.0255**, conditional spline
**3.5517 ± 0.1381**, FM **3.8695 ± 0.0021**. Maximum 120-epoch NLL:
Simformer **4.3895 ± 0.3940**, FM **3.2431 ± 0.0212**. Each uses three seeds and
byte-identical splits/scalers. Epochs are upper limits with validation early stopping.

```bash
python src/simformer_cold_atom.py --data-dir /path/to/data/compiled --baseline-dir results --out results_simformer_30 --epochs 30 --seeds 11 22 33
python src/simformer_cold_atom.py --data-dir /path/to/data/compiled --baseline-dir results --out results_simformer_120 --epochs 120 --seeds 11 22 33
python src/report_simformer.py --root .
```

First reproduce the spline and FM runs to regenerate private row-level artifacts
needed by the comparison report. The public repository includes aggregate metrics,
convergence checks, training histories, plots, logs and configuration; local
training regenerates checkpoints, sample archives and per-simulation losses.

## Data contract

Supply a directory containing:

| File | Required content |
|---|---|
| `compiled_vectors-0.05.csv` | Atom output records, including simulation ID and X/Y/Vx/Vy/Vz |
| `input_data(full).csv` | Simulation ID and nine experimental control parameters |

Expected source hashes, exclusions, cohort sizes and split sizes are recorded in
[`results/data_audit.json`](results/data_audit.json). The retained subset contains
33,424 atoms across 12,478 simulation IDs. IDs mapping to multiple different
control vectors are excluded, never resolved by choosing an arbitrary row.

Targets are X/Y in mm, Vx, Vy and log1p(Vz), standardized using training data only.
Conditions use the same training-only standardization. Each simulation has equal
total weight during preprocessing, neural training, checkpoint validation and
primary evaluation. Atom-weighted NLL is also reported separately.

## Install and run

Python 3.12 was used for the recorded CPU runs; exact library versions are pinned.

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py --data-dir /path/to/data/compiled
python run_flow_matching.py --data-dir /path/to/data/compiled --extended
```

The first runner fits five model families, generates local row-level artifacts,
diagnostics and a report. Run it before reproducing FM: the public repository
omits the per-simulation losses that FM comparisons require. The FM runner uses
the regenerated baseline files, verifies identical preprocessing
and split manifests, and runs the 30-epoch and optional 120-epoch FM budgets.
Rerunning these commands updates the corresponding results directories.
The original fixed split seeds are 42/43 and initialization seeds are 11/22/33.

## Offline sampling

Train the models first, or supply compatible checkpoints separately.
`example_context.json` contains illustrative values for all nine named physical
controls, not a recorded simulation configuration. Replace them with a desired
experimental configuration, then run:

```bash
python src/predict.py --model spline --results results \
  --context-json example_context.json --samples 1000 --output samples.csv

python src/predict.py --model flow_matching --results results \
  --fm-results results_flow_matching_120 --context-json example_context.json \
  --solver rk4 --steps 32 --samples 1000 --output fm_samples.csv
```

Output X/Y are in metres, Vz is inverse-transformed, and Vx/Vy retain source units.
No physical clipping or rejection filter is applied. The models are not constrained
to valid physical support and do not predict atom yield or zero-output probability.

## Artifact guide

| Location | Contents |
|---|---|
| `src/` | Data preparation, training, exact-divergence ODE likelihood, sampling and reports |
| `tests/` | Eight integrity, sampling and analytic ODE tests |
| `results/` | Gaussian/GMM/spline baselines, 11 total fitted model/seed runs |
| `results_flow_matching/` | Three FM seeds with a maximum 30 epochs |
| `results_flow_matching_120/` | Three FM seeds with a maximum 120 epochs |
| `budget_comparison.csv` | Full per-seed budget comparison |
| `budget_comparisons.json` | Paired simulation-bootstrap NLL differences |
| `budget_comparison.png` | All-seed NLL and fixed-sampler CPU cost |
| `verification_v2.json` | Completed integrity and archival checks |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | English methodology, findings and limits |

Each results directory contains raw `run.log`, configuration/protocol JSON,
training histories and aggregate metrics. Saved `.pt` checkpoints, per-simulation
NLL, split manifests and sampling NPZs are generated locally but excluded from
this public repository. FM directories contain numerical convergence checks, solver sweeps,
common-bin quality diagnostics, interval coverage and plots.

For detailed generated Chinese research reports, see:

- [`EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md)
- [`FLOW_MATCHING_REPORT_ZH.md`](FLOW_MATCHING_REPORT_ZH.md)
- [`FLOW_MATCHING_120_REPORT_ZH.md`](FLOW_MATCHING_120_REPORT_ZH.md)
- [`FM_DECISION_ZH.md`](FM_DECISION_ZH.md)

The NLL representation, retained data subset, unequal extended training budgets,
test-set reuse and lack of a comparable cold-atom Simformer run all constrain
interpretation. Read the experiment report before comparing headline numbers.
