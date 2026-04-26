# AGENTS.md

## Scope

- Work only inside `Project/`.
- Do not modify course notes or files outside `Project/` unless explicitly instructed.
- Treat this repository as a project-only workspace.

## Role of This File

This file defines how work should be carried out.

Project requirements live in `context.md`.
When there is any ambiguity, use `context.md` as the source of truth for project scope and use this file as the source of truth for implementation style.

## Submission-Oriented Workflow

The final deliverable is expected to be notebook-first.

Therefore the working approach should be:

1. build and validate the project with `.py` files
2. keep the reusable logic organized and simple
3. consolidate the stable implementation into a final submission notebook

Do not treat the final notebook as an afterthought.
The notebook is part of the target architecture from the beginning.

## Operating Principles

### 1. Build incrementally

- Implement only the requested step.
- Avoid jumping ahead to future phases.
- Keep the repository runnable after each meaningful change.

### 2. Prefer simple solutions

- Favor readability over abstraction.
- Use standard PyTorch patterns.
- Keep code close to course level.

### 3. Match the intended structure

Use the repository as follows:

- `src/` for reusable code
- `scripts/` for a very small number of development runners
- `notebooks/exploration/` for optional experimentation
- `notebooks/submission/` for the final polished notebook
- `outputs/` for generated artifacts
- `report/figures/` only if a separate written report is later needed

If a requested feature needs new files, place them according to this structure.

## Technical Constraints

Preferred stack:

- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib / seaborn
- scikit-learn

Avoid unless explicitly requested:

- PyTorch Lightning
- Hydra or advanced config frameworks
- experiment tracking platforms
- pretrained foundation models
- overly complex architectures

## Model-Level Expectations

The project is expected to support:

1. baseline classifier
2. concept predictor
3. Independent CBM
4. Joint CBM
5. Hybrid CBM

Follow the definitions in `context.md` when implementing any of these models.

## Code Style

- Keep modules small and direct.
- Prefer explicit training loops over framework magic.
- Add helper functions only when they reduce repeated logic.
- Avoid premature generalization.
- Use clear names that reflect the project vocabulary: concepts, logits, interventions, side channel, and so on.

## Experiment Discipline

When adding scripts or utilities:

- make inputs and outputs obvious
- keep the number of scripts low
- save artifacts in the correct `outputs/` subdirectory
- keep experiment behavior reproducible
- avoid hidden side effects

If a script trains a model, it should eventually make it easy to identify:

- which model was trained
- with which settings
- where metrics and checkpoints were saved

## Documentation Discipline

When the implementation evolves:

- keep `README.md` aligned with the real repository state
- update `context.md` only if project scope changes
- update `AGENTS.md` only if the working rules or structure change
- keep the submission strategy explicit

Do not describe code as implemented if it does not exist yet.

## Notebook Consolidation Rule

As the project stabilizes:

- move from many experimental fragments toward one coherent notebook
- avoid copying inconsistent versions of the same logic into notebook cells
- consolidate only after the Python implementation is understood and working

The final notebook should be readable in course style:

1. introduction
2. dataset and concepts
3. models
4. training
5. evaluation
6. interventions
7. conclusions

## Default Development Path

Unless the user asks for a different order, the sensible build sequence is:

1. dataset and concept-label utilities
2. baseline classifier
3. concept predictor
4. Independent CBM
5. Joint CBM
6. Hybrid CBM
7. dropout experiment
8. intervention analysis
9. final notebook consolidation

This is a planning reference, not a mandate to implement multiple phases at once.

## Expected Behavior When Editing

Before changing code:

1. inspect the relevant files
2. make the smallest coherent change
3. preserve consistency with `context.md`
4. keep documentation honest
5. explain how the change should be tested
