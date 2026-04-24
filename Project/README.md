# Fashion-MNIST Concept Bottleneck Models

This folder contains the course project for studying Concept Bottleneck Models (CBMs) on Fashion-MNIST with PyTorch.

The current state is a scaffold only. The repository structure is prepared for later implementation of:

1. Baseline classifier: `x -> y`
2. Concept predictor: `x -> c`
3. Independent CBM: `x -> c`, then `c -> y`
4. Joint CBM: `x -> c -> y`
5. Hybrid CBM with side channel: `f(c) + s(x)`

## Structure

- `src/`: source code
- `scripts/`: command-line entry points and experiment runners
- `notebooks/`: exploratory notebooks
- `outputs/`: generated artifacts
- `report/`: report materials

## Planned Training Style

The implementation will follow the course conventions:

- PyTorch `nn.Module`
- `DataLoader`
- standard optimizer training loops
- `CrossEntropyLoss` for labels
- `BCEWithLogitsLoss` for concepts

## Outputs

Generated artifacts should be written to:

- `outputs/checkpoints/`
- `outputs/metrics/`
- `outputs/plots/`
- `outputs/tables/`

Figures for the report should go in:

- `report/figures/`

## Notes

- Keep all work inside `Project/`
- Do not implement the side-channel dropout experiment yet
- Preferred GitHub account for project work: `pablocoma`
