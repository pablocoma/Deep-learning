# Concept Bottleneck Models on Fashion-MNIST

## 📘 Overview

This project studies Concept Bottleneck Models (CBMs) in a supervised Deep Learning setting using Fashion-MNIST.

The goal is to compare standard neural networks with concept-based models and analyze the trade-off between:

- Performance (accuracy, AUROC)
- Interpretability
- Steerability (control via concept interventions)

---

## 🧵 Dataset

Fashion-MNIST:

- 28×28 grayscale images
- 10 clothing classes

---

## 🧩 Concepts

We define 8 binary concepts derived from class labels, such as:

- is_footwear
- has_sleeves
- is_long_garment
- etc.

These concepts are shared across classes and are used as an intermediate representation.

---

## 🧠 Models implemented

1. **Baseline classifier**  
   x → y

2. **Concept predictor**  
   x → c

3. **Independent CBM**  
   x → c → y (trained in two stages)

4. **Joint CBM**  
   x → c → y (trained end-to-end)

5. **Hybrid CBM**  
   y = f(c) + s(x)

---

## ⚡ Experiments

### 1. Model comparison
Compare performance and interpretability across models.

### 2. Side-channel dropout (Hybrid CBM)
Train with:

p ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}

Analyze how performance changes with reliance on concepts.

### 3. Concept interventions
Flip individual concepts and observe:

- Change in prediction probabilities
- Percentage of label changes
- Most influential concepts

---

## 📁 Project structure