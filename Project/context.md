# Project Context

## Topic

Deep Learning course project on Concept Bottleneck Models over Fashion-MNIST.

## Dataset

Fashion-MNIST is the main dataset.
The project will stay close to the course material and avoid unnecessary complexity.

## Intended Learning Focus

- Supervised image classification with PyTorch
- Concept prediction as a multi-label task
- Comparing direct prediction against concept-based pipelines
- Studying the role of side information and side-channel dropout

## Planned Comparisons

The project will compare:

1. A direct baseline classifier from image to class
2. A concept predictor from image to concept vector
3. An independent CBM trained in two stages
4. A joint CBM trained with class and concept losses together
5. A hybrid CBM with a direct image side channel

## Losses

- Class prediction: `CrossEntropyLoss`
- Concept prediction: `BCEWithLogitsLoss`
- Joint CBM: class loss plus weighted concept loss

## Architecture Direction

- Small CNN backbones for image processing
- Lightweight MLP-style heads where appropriate
- Explicit `nn.Module` classes for each model family
- Dropout only where justified by the experimental design

## Planned Outputs

- Checkpoints
- Metrics tables
- Plots
- Report figures

## Current Status

Only the scaffold is being created right now.
Model code, data preparation, training scripts, and experiments are intentionally left for later steps.
