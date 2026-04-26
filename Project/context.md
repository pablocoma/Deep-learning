# Project Context: Concept Bottleneck Models on Fashion-MNIST

## Purpose

This file is the canonical project brief.

Use it to understand:

- what the project is trying to achieve
- which models must be implemented
- which experiments matter
- what "done" should look like

Implementation workflow and agent behavior belong in `AGENTS.md`.

## Delivery Constraint

The project should ultimately be deliverable as a notebook-oriented university submission.

That affects the implementation strategy:

- development can use `.py` files
- the final artifact should be a clean notebook
- the notebook should be able to present the full pipeline coherently

The brief below defines the scientific and technical scope regardless of whether a step is first developed in Python files or later consolidated into a notebook.

## Objective

The project studies Concept Bottleneck Models (CBMs) in a supervised image classification setting using Fashion-MNIST.

The main objective is to compare standard predictive performance with interpretability and steerability obtained through intermediate concepts.

The central question is:

Can an interpretable concept layer provide useful control and explanation without sacrificing too much predictive quality?

## Dataset

Fashion-MNIST:

- 28x28 grayscale images
- 10 classes

Class mapping:

- `0`: T-shirt/top
- `1`: Trouser
- `2`: Pullover
- `3`: Dress
- `4`: Coat
- `5`: Sandal
- `6`: Shirt
- `7`: Sneaker
- `8`: Bag
- `9`: Ankle boot

## Concepts

The project uses 8 binary concepts derived deterministically from the class label:

1. `is_footwear` -> classes 5, 7, 9
2. `is_closed_footwear` -> classes 7, 9
3. `is_footwear_or_bag` -> classes 5, 7, 8, 9
4. `has_sleeves` -> classes 0, 2, 3, 4, 6
5. `has_collar` -> classes 4, 6
6. `is_long_garment` -> classes 3, 4
7. `is_outerwear_layer` -> classes 2, 4
8. `is_legwear_or_footwear` -> classes 1, 5, 7, 9

These concepts are shared across multiple classes and act as the interpretable bottleneck.

## Model Families

### 1. Baseline classifier

Mapping: `x -> y`

- input: image
- output: 10 class logits
- role: performance reference

### 2. Concept predictor

Mapping: `x -> c`

- input: image
- output: 8 concept logits
- task: multi-label prediction
- loss: `BCEWithLogitsLoss`

### 3. Independent CBM

Mapping: `x -> c -> y`

Training flow:

1. train the concept predictor
2. freeze it
3. train a label predictor from predicted concepts

Main property:

- strong interpretability
- concept errors can propagate to label prediction

### 4. Joint CBM

Mapping: `x -> c -> y`

Training objective:

`classification_loss + lambda_concept * concept_loss`

Main property:

- end-to-end optimization
- concepts may become less semantically clean

### 5. Hybrid CBM

Mapping: `y = f(c) + s(x)`

- `f(c)`: concept-based path
- `s(x)`: direct image-based side channel

Main property:

- intended to balance interpretability and accuracy

## Core Experiments

### Model comparison

Compare the baseline, concept predictor, Independent CBM, Joint CBM, and Hybrid CBM.

### Side-channel dropout

For the Hybrid CBM, apply dropout only to the side channel and compare:

`p in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}`

Goal:

- measure reliance on direct image features versus concepts

### Concept interventions

For a predicted concept vector:

1. flip one concept
2. recompute the final prediction
3. measure the prediction change

Goal:

- quantify steerability
- identify which concepts most influence the model

## Evaluation

Classification metrics:

- accuracy
- AUROC using one-vs-rest

Concept metrics:

- per-concept accuracy
- per-concept F1
- macro averages

Intervention metrics:

- change in prediction probability
- percentage of label changes
- concept influence ranking

## Training Expectations

The implementation should support:

- train/validation/test splits
- reproducible seeds
- early stopping based on validation loss
- saving outputs for later analysis

## Outputs

Generated artifacts should be organized under:

- `outputs/checkpoints/`
- `outputs/metrics/`
- `outputs/plots/`
- `outputs/tables/`

## Definition of Done

At project completion, the work should support both of these views:

1. a development view:
   the code is clear enough to iterate on in `.py` files
2. a submission view:
   the final notebook is coherent, readable, and sufficient for course delivery

The final notebook should include:

- problem framing
- dataset and concept setup
- model training and evaluation
- intervention analysis
- plots and conclusions

## Scope Boundary

This is a course project.

The priority is:

- correctness
- readability
- interpretable experimentation

The project does not require advanced infrastructure or research-scale engineering.

Optional extensions are acceptable only after the core pipeline is complete.
