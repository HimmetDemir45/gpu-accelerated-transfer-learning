# 🚀 GPU-Accelerated Transfer Learning for Pharmaceutical Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

This repository demonstrates a **GPU-accelerated deep learning pipeline** for pharmaceutical drugs and vitamins image classification. The project evolved from a basic transfer learning experiment into a **hardware-aware, performance-oriented deep learning engineering study**.

Rather than treating artificial intelligence as a black box, this work focuses on **explicit control over training behavior**, including GPU utilization, memory management, mixed precision training, and systematic evaluation.

---

## 📊 Dataset

**Pharmaceutical Drugs and Vitamins – Synthetic Images (Kaggle)**
[https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images](https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images)

* **Total Images:** 10,000
* **Classes:** 10
* **Split:** 80% Train / 20% Test (with validation split from training set)

---

## 🧠 Model Architecture

* **Backbone:** ResNet50 (ImageNet pretrained)
* **Input Size:** 224×224 RGB
* **Architecture Design:**

  * Global Average Pooling
  * Dense (512, ReLU)
  * Dropout (0.3)
  * Dense (256, ReLU)
  * Dropout (0.3)
  * Softmax output layer

Residual connections in ResNet50 allow deeper feature extraction while preventing vanishing gradients.

---

## ⚙️ Training Strategy

### Transfer Learning Phase

* Convolutional backbone frozen
* Only custom classification head trained
* Goal: learn domain-specific features efficiently

### Fine-Tuning Phase

* Upper layers of ResNet50 unfrozen
* Lower learning rate applied
* Enables task-specific feature refinement

---

## 🖥️ GPU Acceleration & Optimization

### GPU Management

* Explicit GPU availability checks
* CUDA compatibility verification
* Memory growth enabled to avoid OOM errors

### Mixed Precision Training

* `mixed_float16` policy enabled
* Reduced VRAM consumption
* Faster matrix operations on GPU

### Batch Size Optimization

* Batch size increased to **64** (GPU-optimized)
* Improved throughput compared to CPU training

---

## ⚡ CPU vs GPU Training Comparison

| Aspect               | CPU Training | GPU Training            |
| -------------------- | ------------ | ----------------------- |
| Training Speed       | Very Slow    | 🚀 Significantly Faster |
| Batch Size           | Small (8–16) | Large (64+)             |
| Memory Handling      | Limited      | Efficient VRAM usage    |
| Experiment Iteration | Slow         | Rapid                   |
| Scalability          | Poor         | Excellent               |

GPU acceleration enabled faster experimentation and more stable convergence.

---

## 📈 Results

### Transfer Learning Results

* **Training Accuracy:** 84.42%
* **Validation Accuracy:** 86.62%
* **Training Loss:** 0.4491
* **Validation Loss:** 0.3968

### Fine-Tuning Results

* **Final Test Accuracy:** **89.90%**
* **Macro F1-score:** 0.90

Balanced precision and recall across all classes confirm strong generalization.

---

## 📊 Evaluation & Diagnostics

* Classification Report
* Confusion Matrix
* Random sample predictions with confidence scores

These metrics ensure that improvements are meaningful and not caused by overfitting.

---

## 🧠 What I Learned from This Project

* GPU usage must be explicitly verified and managed.
* Mixed precision training improves speed but requires careful numerical handling.
* Larger batch sizes are only effective when aligned with GPU memory constraints.
* Checkpointing and resumable training are critical for long-running experiments.
* Hardware monitoring (temperature, VRAM usage) is essential for sustainable training.
* Model accuracy must always be supported by detailed evaluation metrics.

Most importantly, I learned how to **use AI-assisted development as a productivity tool**, while keeping full control over architectural and optimization decisions.

---

## ▶️ How to Run

```bash
git clone https://github.com/HimmetDemir45/gpu-accelerated-transfer-learning.git
cd gpu-accelerated-transfer-learning
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn
python transferLearning.py
```

---

## 👨‍💻 Author

**Himmet**
GitHub: [https://github.com/HimmetDemir45](https://github.com/HimmetDemir45)

This repository is created for **educational, experimental, and portfolio purposes**, demonstrating applied deep learning engineering skills.
