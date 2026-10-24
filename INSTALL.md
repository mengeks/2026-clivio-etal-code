# causal-embedding: installation and usage

`causal_embedding` estimates low-dimensional **prognostic** and **treatment**
directions in covariate space with score-ratio dimension reduction. Score
networks are trained by binary cross-entropy (treatment) and implicit score
matching (treatment and outcome jointly). Subtracting the treatment score from
the joint score, sample by sample, isolates the directions that drive the
control outcome.

To reproduce the paper, see [README.md](README.md). This file covers using
the package on your own data.

## Installation

Python 3.9 or 3.10, TensorFlow 2.10–2.15 (Keras 2 API).

```bash
pip install git+https://github.com/mengeks/causal-embedding.git
```

For development, or to run the paper scripts, install from a clone.

```bash
git clone https://github.com/mengeks/causal-embedding.git
cd causal-embedding
uv sync                      # exact versions from uv.lock (TensorFlow 2.10.0, CPU)
# or: python -m venv .venv && .venv/bin/pip install -e .
```

`pyproject.toml` lists the supported version ranges. `uv.lock` pins the
versions that produced `outputs/`: Python 3.9, TensorFlow 2.10.0, NumPy 1.24.2,
SciPy 1.11.3, scikit-learn 1.6.1, pandas 2.0.1 and PyYAML 6.0. TensorFlow uses
a CUDA GPU automatically when it has a compatible build and driver. The lockfile
does not install CUDA. Set `CUDA_VISIBLE_DEVICES=""` to force CPU.

## Minimum working example

```python
import numpy as np
from causal_embedding.models.pipeline import fit_directions
from causal_embedding.simulations.dgp import gen_rank1, make_rank1_directions
from causal_embedding.utils.config import pipeline_defaults

# Mock data: quadratic single-index DGP (p = 30, n = 500)
alpha, beta = make_rank1_directions(p=30, q=5)
X, A, Y, _ = gen_rank1(n=500, alpha=alpha, beta=beta, seed=1)

cfg = pipeline_defaults()                 # paper settings (configs/default.yaml)
fit = fit_directions(X, A, Y, cfg, seed=1)

cos2 = lambda u, v: float((u @ v) ** 2 / ((u @ u) * (v @ v)))
print(f"prognostic direction: cos2 = {cos2(fit['alpha'], alpha):.3f}")
print(f"treatment direction:  cos2 = {cos2(fit['beta'], beta):.3f}")

# One-dimensional prognostic representation (directions refer to standardised X)
Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
prognostic_score = Xs @ fit['alpha']
```

This takes one to two minutes on two CPU threads and prints squared cosines
near 0.9 for both directions (0.921 and 0.870 when tested). For your own data, pass `X` (n × p), a binary `A`,
and `Y`. Change settings by editing the `cfg` dict, for example:

- `cfg['init']['method'] = 'lasso'` uses a LASSO warm start, the better choice
  when the mechanisms are close to linear.
- `cfg['joint']['rank'] = 3` widens the joint-score bottleneck.

`fit_directions` returns:

| key | content |
|---|---|
| `alpha`, `beta` | unit prognostic / treatment directions (standardised covariates) |
| `alpha0`, `beta0` | warm-start directions (SAVE or LASSO) |
| `H0` | H₀ matrices: `per_sample` (used), `difference_ctrl`, `difference_all` |
| `w_A`, `d` | treatment scores (all units), per-sample differences (controls) |
| `info` | selected epochs and validation losses |

The package estimates directions only. Downstream ATT estimators that use them
as deconfounding scores are in `experiments/clivio/` (R).

## Package map

```text
src/causal_embedding/
├── default.yaml                 # symlink to configs/default.yaml (shipped defaults)
├── models/
│   ├── pipeline.py              # fit_directions (real data), run_pipeline (simulation + evaluation)
│   ├── propensity.py            # BCE propensity network, treatment scores w_A
│   ├── ism.py                   # implicit score matching: exact-QR / Hutchinson trace, Stiefel step
│   ├── score_ratio_reduction.py # ScoreRatioNet bottleneck network (Baptista et al.), ScoreRatioDR
│   └── baselines.py             # SAVE, SIR, LASSO / L1-logistic directions
├── simulations/dgp.py           # linear (Clivio et al.) and nonlinear DGPs, oracle scores
└── utils/                       # config merging and hashing, metrics, seeding and provenance
```

Extension points:

- New treatment-score models go in `models/propensity.py` and are selected in
  `pipeline.fit_treatment`.
- New warm starts go in `pipeline.initial_directions`.
- New DGPs go in `simulations/dgp.py`, together with the `simulate` dispatcher.
