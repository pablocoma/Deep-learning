# Technical guide

This directory contains the implementation and final artifacts for the
Fashion-MNIST Concept Bottleneck Model study.

For the project overview and headline results, start at the
[repository README](../README.md).

## Research question

Can an interpretable concept layer provide useful control and explanation
without sacrificing too much predictive quality?

Eight binary concepts are assigned to Fashion-MNIST classes. The experiment
compares five model families:

1. **Baseline classifier**: `x -> y`
2. **Concept predictor**: `x -> c`
3. **Independent CBM**: `x -> c -> y`, trained in two stages
4. **Joint CBM**: `x -> c -> y`, trained end-to-end
5. **Hybrid CBM**: `y = f(c) + s(x)`, with a direct image side channel

## Repository structure

```text
Project/
├── notebooks/submission/CBMs_project.ipynb
├── report/deep_learning.pdf
├── results/
├── scripts/
├── src/
├── outputs/
├── context.md
└── requirements.txt
```

- `notebooks/submission/`: executed, self-contained course deliverable.
- `report/`: compact final report.
- `results/`: curated tables from the final notebook and report.
- `src/`: reusable dataset, model, training, evaluation and intervention code.
- `scripts/`: experiment, HPO, dropout and intervention runners.
- `outputs/`: local generated artifacts; ignored by Git except placeholders.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The code supports CPU, CUDA and Apple Metal Performance Shaders where
available. Fashion-MNIST is downloaded automatically.

## Run development checks

From this directory:

```bash
python scripts/check_models.py
python -m compileall -q src scripts
```

## Run experiments

The reusable modules and command-line runners support the development workflow:

```bash
python scripts/run_experiment.py --help
python scripts/run_dropout_sweep.py --help
python scripts/run_interventions.py --help
```

Generated checkpoints, metrics, plots and summary tables are written below
`outputs/` and deliberately excluded from version control.

## Final artifacts and provenance

The final notebook is the canonical executed deliverable. The PDF report
summarises the same model comparison, side-channel dropout experiment and
concept interventions. Curated CSV files in `results/` make the headline
numbers easy to inspect without committing raw checkpoints or local caches.

The final reported experiments use a fixed seed of 42. Development outputs from
short smoke runs are not included in this portfolio branch.
