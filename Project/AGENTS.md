# AGENTS.md

## Scope

All project development must happen inside `Project/`.
Do not modify files outside `Project/` unless explicitly instructed.

## Project Goal

This project studies Concept Bottleneck Models (CBMs) on Fashion-MNIST using PyTorch, aligned with the course material and coding style.

## Current Phase

Scaffold only.
Do not implement full models, full training pipelines, or experiment logic yet unless explicitly requested.

## Planned Model Variants

1. Baseline classifier: `x -> y`
2. Concept predictor: `x -> c`
3. Independent CBM: train `x -> c`, then train `c -> y`
4. Joint CBM: train `x -> c -> y` with classification loss plus `lambda * concept loss`
5. Hybrid CBM with side channel: final logits `f(c) + s(x)`

## Planned Experiment

Later, add the side-channel dropout experiment with:

- `p = 0.0`
- `p = 0.1`
- `p = 0.3`
- `p = 0.5`
- `p = 0.7`
- `p = 0.9`

Do not implement that experiment yet.

## Technical Constraints

- Use PyTorch and `torchvision`
- Use small CNNs where image models are needed
- Define models with `torch.nn.Module`
- Use `DataLoader` for batching
- Use standard training loops with:
  - `optimizer.zero_grad()`
  - `loss.backward()`
  - `optimizer.step()`
- Use `CrossEntropyLoss` for class prediction
- Use `BCEWithLogitsLoss` for multi-label concept prediction
- Use dropout where required by the design

## Repository Conventions

- Keep the structure simple and course-oriented
- Prefer readable, modular code over abstraction-heavy design
- Store generated artifacts under `outputs/`
- Store report assets under `report/figures/`
- Keep notebooks exploratory and reproducible

## Collaboration

Preferred GitHub account for project-related work: `pablocoma`
