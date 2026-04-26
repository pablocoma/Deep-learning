# Project Context: Concept Bottleneck Models over Fashion-MNIST

## 🎯 Objective

This project studies Concept Bottleneck Models (CBMs) in a supervised Deep Learning setting.

The main goal is to compare standard neural networks with models that use human-interpretable intermediate concepts.

We want to analyze the trade-off between:

- Performance (accuracy, AUROC)
- Interpretability
- Steerability (ability to control predictions via concepts)

---

## 🧵 Dataset

We use Fashion-MNIST:

- 28x28 grayscale images
- 10 classes:
  0: T-shirt/top  
  1: Trouser  
  2: Pullover  
  3: Dress  
  4: Coat  
  5: Sandal  
  6: Shirt  
  7: Sneaker  
  8: Bag  
  9: Ankle boot  

---

## 🧩 Concepts

We define 8 binary concepts derived from class labels:

1. is_footwear → classes 5, 7, 9  
2. is_closed_footwear → classes 7, 9  
3. is_footwear_or_bag → classes 5, 7, 8, 9  
4. has_sleeves → classes 0, 2, 3, 4, 6  
5. has_collar → classes 4, 6  
6. is_long_garment → classes 3, 4  
7. is_outerwear_layer → classes 2, 4  
8. is_legwear_or_footwear → classes 1, 5, 7, 9  

Concept labels are generated deterministically from the class label.

---

## 🧠 Model families

### 1. Baseline classifier

Mapping:

x → y

- Input: image
- Output: class label (10 classes)
- Architecture: small CNN

This model serves as a performance reference.

---

### 2. Concept predictor

Mapping:

x → c

- Output: 8 binary concepts
- Multi-label classification
- Loss: BCEWithLogitsLoss

This model evaluates how well concepts can be predicted from images.

---

### 3. Independent CBM

Training procedure:

Step 1: train x → c  
Step 2: freeze concept model and train c → y  

Final model:

x → ĉ → ŷ

Key idea:
- Concepts are learned independently from the final task.

Pros:
- High interpretability
- Clean interventions

Cons:
- Errors in concept prediction propagate to final prediction

---

### 4. Joint CBM

Mapping:

x → c → y

Training loss:

Loss = classification_loss + λ * concept_loss

Key idea:
- Concepts and labels are learned jointly.

Pros:
- Better performance (usually)
- End-to-end optimization

Cons:
- Concepts may lose semantic meaning
- Lower interpretability

---

### 5. Hybrid CBM

Mapping:

y = f(c) + s(x)

- f(c): concept-based prediction
- s(x): direct image-to-label path (side channel)

Final logits:

logits = f(c) + s(x)

Key idea:
- Combine interpretability and performance

---

## ⚡ Side-channel dropout experiment

We train Hybrid CBM with dropout applied ONLY to the side channel.

Dropout probabilities:

p ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}

Goal:
- Study how much the model relies on concepts vs raw image features

Expected behavior:
- Low dropout → higher accuracy, less reliance on concepts
- High dropout → lower accuracy, more reliance on concepts

---

## 🎮 Steerability (Concept Interventions)

Procedure:

1. Predict concepts for an input image
2. Flip one concept (0 ↔ 1)
3. Recompute class prediction

Metrics:

- Change in prediction probability
- Percentage of label changes
- Ranking of concepts by influence

Interpretation:

- Measures how much each concept affects predictions
- Higher influence → more controllable model

---

## ⚖️ Expected trade-offs

| Model | Performance | Interpretability | Steerability |
|------|------------|----------------|--------------|
| Baseline | High | Low | Low |
| Independent CBM | Medium | High | High |
| Joint CBM | High | Medium | Medium |
| Hybrid CBM | High | Medium | Medium |

---

## 🧪 Training setup

- Train / validation / test split
- Early stopping based on validation loss
- Small CNN architecture
- Standard PyTorch training loop

---

## 📊 Evaluation

### Classification
- Accuracy
- AUROC (one-vs-rest)

### Concepts
- Per-concept accuracy
- Per-concept F1
- Macro averages

---

## 🚀 Optional extension

If time allows:

- Hyperparameter tuning (e.g. learning rate, dropout, λ in Joint CBM)
- Focus on improving baseline or Hybrid CBM

This is not the main objective.

---

## 💡 Key idea of the project

The core question is:

Can we build models that are both accurate and interpretable?

CBMs force the model to reason through human-understandable concepts.

This allows:

- Better explanations
- Direct control over predictions
- Insight into model behavior