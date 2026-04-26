# Concept Bottleneck Models on Fashion-MNIST

## Overview

This repository contains a Deep Learning course project on Concept Bottleneck Models (CBMs) using Fashion-MNIST.

The project compares standard image classifiers with concept-based models and studies the trade-off between:

- predictive performance
- interpretability
- steerability through concept interventions

## Delivery Strategy

The final university deliverable should be a notebook-first submission.

The intended workflow is:

1. develop and validate the project with `.py` files
2. keep the reusable logic clean and modular
3. consolidate the final pipeline into a polished notebook for submission

This means the notebook is the final presentation layer, while the Python files are the development layer used to reach a stable implementation.

## Project Goal

The core question is:

Can we build models that remain accurate while exposing an interpretable concept layer that can be inspected and manipulated?

The planned model families are:

1. Baseline classifier: `x -> y`
2. Concept predictor: `x -> c`
3. Independent CBM: `x -> c -> y` trained in two stages
4. Joint CBM: `x -> c -> y` trained end-to-end
5. Hybrid CBM: `y = f(c) + s(x)`

## Dataset

The project uses Fashion-MNIST:

- 28x28 grayscale images
- 10 classes

Concept labels are derived deterministically from the class labels.

## Recommended Repository Structure

```text
Project/
├── AGENTS.md
├── README.md
├── context.md
├── requirements.txt
├── notebooks/
│   ├── exploration/
│   └── submission/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── plots/
│   └── tables/
├── report/
│   └── figures/
├── scripts/
└── src/
```

Use each area as follows:

- `src/`: reusable implementation code
- `scripts/`: a very small number of runnable development entrypoints
- `notebooks/exploration/`: optional scratch notebooks during development
- `notebooks/submission/`: the final polished submission notebook
- `outputs/`: generated artifacts from experiments
- `report/figures/`: optional figures if you later write a separate report

## Development Schema

The project should not grow into many standalone scripts.

The target is:

- most logic lives in `src/`
- only a few development runners live in `scripts/`
- the final notebook reuses the stabilized logic and then becomes the submission artifact

Good examples of development runners are:

- one experiment runner for training and evaluation
- one intervention runner

The final notebook should then present:

1. the project objective
2. dataset and concept construction
3. model definitions
4. training and evaluation
5. intervention analysis
6. results and conclusions

## Current Repository Status

This repository currently contains the scaffold and documentation, not the finished implementation.

What is expected to be added during development:

- dataset utilities
- concept construction utilities
- model definitions
- training and evaluation code
- intervention analysis
- one final submission notebook

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Working Principle

During development, prioritize clarity and correctness over convenience in the notebook.

In practice:

- prototype and debug with Python modules
- keep scripts minimal
- avoid duplicating core logic across files
- only consolidate into the final notebook once the implementation is stable

This keeps the development process manageable while still matching the notebook-oriented course style at submission time.
