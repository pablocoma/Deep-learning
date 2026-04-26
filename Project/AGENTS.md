# AGENTS.md

## 🔒 Scope of work

- Work ONLY inside the `Project/` folder.
- Do NOT modify any files outside `Project/` unless explicitly instructed.
- Do NOT modify course notes or lab notebooks.

---

## 🎯 Project goal

This is a Deep Learning course project on Concept Bottleneck Models (CBMs) using Fashion-MNIST.

The objective is to compare:

1. Baseline classifier: x → y
2. Concept predictor: x → c
3. Independent CBM: train x → c, then train c → y
4. Joint CBM: train x → c → y with combined loss
5. Hybrid CBM: y = f(c) + s(x)

The project must analyze:

- Performance (accuracy, AUROC)
- Interpretability
- Steerability (concept interventions)

---

## ⚠️ Critical working rules

### 1. Work in small steps
- Implement ONLY what is explicitly requested.
- Do NOT implement future phases.
- Do NOT anticipate upcoming tasks.

### 2. Keep it simple
- Code must be simple, readable, and aligned with course labs.
- Avoid over-engineering.
- Avoid unnecessary abstractions.

### 3. Stay within course level
Use only:
- PyTorch (`nn.Module`, `DataLoader`, training loops)
- Small CNN architectures
- Standard losses (CrossEntropyLoss, BCEWithLogitsLoss)
- Dropout (required in Hybrid CBM)
- Optional BatchNorm (simple usage only)

Do NOT use:
- PyTorch Lightning
- Hydra or advanced config systems
- Weights & Biases
- Optuna (unless explicitly requested later)
- Pretrained models or transfer learning
- Complex architectures

---

## 🧠 Model requirements

### Baseline
- Input: image
- Output: 10 logits
- Loss: CrossEntropyLoss

### Concept predictor
- Input: image
- Output: 8 concept logits
- Loss: BCEWithLogitsLoss
- Multi-label classification

### Independent CBM
- Step 1: train x → c
- Step 2: train c → y
- Final pipeline: x → ĉ → ŷ

### Joint CBM
- Architecture: x → c → y
- Loss:
  total_loss = classification_loss + lambda_concept * concept_loss

### Hybrid CBM
- Two paths:
  - Concept path: f(c)
  - Direct path: s(x)
- Final logits:
  logits = f(c) + s(x)
- Apply dropout ONLY to the side channel s(x)

---

## 📊 Metrics

### Classification
- Accuracy
- AUROC (one-vs-rest)

### Concepts
- Accuracy per concept
- F1 per concept
- Macro-average

---

## 🎮 Interventions

- Predict concepts
- Flip one concept (0 ↔ 1)
- Recompute prediction
- Measure:
  - Change in probability
  - % of label changes
- Rank concepts by influence

---

## 🧪 Training setup

- Use train / validation / test split
- Use early stopping based on validation loss
- Use reproducible seeds
- Save metrics and results

---

## 📁 Outputs

- Save metrics in `outputs/metrics/`
- Save plots in `outputs/plots/`
- Save checkpoints in `outputs/checkpoints/`

---

## 🧭 Development workflow

When making changes:

1. Inspect existing files first
2. Make minimal modifications
3. Keep code runnable
4. Explain what changed
5. Suggest how to test it

---

## 🚫 What NOT to do

- Do NOT rewrite large parts of the project unnecessarily
- Do NOT introduce new frameworks
- Do NOT add complexity beyond course level
- Do NOT implement multiple features at once
- Do NOT ignore instructions in this file

---

## 🧩 Optional extensions (ONLY if explicitly requested)

- Small hyperparameter tuning (e.g. Optuna)
- Comparison between different lambda values in Joint CBM
- Additional plots

These are NOT part of the core implementation.

---

## ✅ Expected usage

The project should be runnable via scripts such as:

python scripts/train_baseline.py  
python scripts/train_concept_predictor.py  
python scripts/train_cbm_independent.py  
python scripts/train_cbm_joint.py  
python scripts/train_hybrid.py  
python scripts/run_dropout_experiment.py  
python scripts/run_interventions.py