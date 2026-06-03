# 🖼️ CIFAR-10 Image Classification using CNN
**MSc Data Analytics Project — Poojitha Kalyanam (2024)**

End-to-end image classification using a deep Convolutional Neural Network (CNN) on the CIFAR-10 benchmark dataset. The model classifies 32×32 colour images into 10 object categories using batch normalisation, dropout regularisation, and two-phase training with data augmentation.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | `tf.keras.datasets.cifar10` |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32×32 pixels (RGB) |
| Classes | 10 |

**Classes:** airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

---

## 🧠 Model Architecture

```
Input (32×32×3)
  → Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPooling
  → Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPooling
  → Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPooling
  → Flatten → Dropout(0.2)
  → Dense(1024, ReLU) → Dropout(0.2)
  → Dense(10, Softmax)
```

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Regularisation | BatchNormalization + Dropout(0.2) |
| Phase 1 | 50 epochs — no augmentation |
| Phase 2 | 10 more epochs — with augmentation |
| Augmentation | Width/height shifts ±10%, horizontal flip |

---

## 📈 Results

| Metric | Score |
|---|---|
| Test accuracy | **68%** |
| Evaluation | Confusion matrix · Precision · Recall · F1-score per class |

> Significant accuracy improvement over baseline achieved by stacking BatchNorm + Dropout across 3 convolutional blocks. Two-phase training (baseline → augmentation) lets the model first learn core features, then generalise.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

**Libraries:** `tensorflow` · `keras` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

**Techniques:** CNN · BatchNormalization · Dropout · MaxPooling · Data Augmentation · ImageDataGenerator · Confusion Matrix · Classification Report

---

## 📁 Repository Structure

```
CIFAR10-Image-Classification/
│
├── cifar10_classification.py   # Full pipeline: EDA → model → evaluation
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Codewithpuji/CIFAR10-Image-Classification.git
cd CIFAR10-Image-Classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run** *(CIFAR-10 dataset downloads automatically — no manual setup needed)*
```bash
python cifar10_classification.py
```

---

## 📊 Visualisations Included

- 5×5 sample image grid from the training set
- Training vs validation accuracy curves (augmentation phase)
- Single image prediction with original vs predicted label
- Confusion matrix heatmap (10×10, annotated)
- Full classification report (precision, recall, F1-score per class)

---

## 📚 Research Context

Completed as part of the **MSc Data Analytics** programme at the **University for the Creative Arts, Germany (2024)**. Focuses on evaluating CNN architecture decisions — depth, regularisation, and augmentation strategy — on a standard computer vision benchmark.

---

## 👩‍💻 Author

**Poojitha Kalyanam** — Data Analyst | MSc Data Analytics  
📍 Berlin, Germany  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/poojitha-kalyanam)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Codewithpuji)
