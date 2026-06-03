# 🧠 CNN Image Classification — Healthcare & CIFAR-10
**MSc Data Analytics Projects — Poojitha Kalyanam (2024)**

Two end-to-end Convolutional Neural Network (CNN) projects demonstrating deep learning for image classification: a **healthcare application** targeting COVID-19 detection, and a **benchmark multi-class classifier** on the standard CIFAR-10 dataset.

---

## 📂 Projects Overview

| | Project 1 | Project 2 |
|---|---|---|
| **Task** | COVID-19 CT Scan Classification | CIFAR-10 Multi-class Classification |
| **Type** | Binary classification | 10-class classification |
| **Dataset** | SARS-CoV-2 CT Scan (Kaggle) | CIFAR-10 (Kaggle / tf.keras) |
| **Images** | 930 (744 train / 186 val) | 60,000 (50k train / 10k test) |
| **Epochs** | 10 | 50 |
| **Result** | **100% accuracy** | **68% accuracy** |
| **Colab** | [Open](https://colab.research.google.com/drive/16EPBee4FmvsI5pvG2T13UUgJOwkZhYbT) | [Open](https://colab.research.google.com/drive/1dFJO-Fd2ftZtaaFFSZ2QW43V8u6ziD5Q) |

---

## 🦠 Project 1 — COVID-19 CT Scan Classification

### Problem
Build a CNN to classify chest CT images as **COVID-19 positive or negative**, targeting autonomous hospital robots that need to identify at-risk patients and manage medical equipment without human contact.

### Dataset
| Property | Detail |
|---|---|
| Source | [Kaggle — SARS-CoV-2 CT Scan Dataset](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) |
| Total images | 930 |
| Training set | 744 images |
| Validation set | 186 images |
| Split | 80/20 |
| Task | Binary: COVID-19 Positive / Negative |

### Model Architecture
```
Input (224×224×3)
    → Conv2D + MaxPooling
    → Conv2D + MaxPooling
    → Flatten
    → Dense (ReLU)
    → Dense (1, Sigmoid)  ← binary output
```

| Setting | Value |
|---|---|
| Optimizer | Adam (lr = 0.001) |
| Loss | Binary cross-entropy |
| Output activation | Sigmoid |
| Data augmentation | Rotation, flipping, scaling |
| Epochs | 10 |

### Results
| Metric | Score |
|---|---|
| Training accuracy | **1.00** |
| Validation accuracy | **1.00** |
| Precision | 1.0 |
| Recall | 1.0 |
| F1-score | 1.0 |

> All 186 validation images correctly classified. Confusion matrix shows zero misclassifications.

---

## 🖼️ Project 2 — CIFAR-10 Multi-class Classification

### Problem
Train a CNN on the CIFAR-10 benchmark dataset to classify 32×32 colour images into **10 object categories**: aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

### Dataset
| Property | Detail |
|---|---|
| Source | [Kaggle CIFAR-10](https://www.kaggle.com/c/cifar-10) / `tf.keras.datasets` |
| Total images | 60,000 |
| Training set | 50,000 |
| Test set | 10,000 |
| Image size | 32×32 pixels (colour) |
| Classes | 10 |

### Model Architecture
```
Input (32×32×3)
    → Conv2D → BatchNorm → MaxPooling → Dropout
    → Conv2D → BatchNorm → MaxPooling → Dropout
    → Conv2D → BatchNorm → MaxPooling → Dropout
    → Flatten
    → Dense (ReLU) → Dropout
    → Dense (10, Softmax)  ← 10-class output
```

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Output activation | Softmax |
| Data augmentation | Random shifts, horizontal flips, whitening (ImageDataGenerator) |
| Epochs | 50 |
| Regularisation | BatchNormalization + Dropout |

### Results
| Metric | Score |
|---|---|
| Test accuracy | **68%** |
| Evaluation | Confusion matrix, precision, recall, F1-score per class |

> Significant improvement over the baseline CNN achieved by adding batch normalization, dropout layers, and data augmentation. Confusion matrix identified class-level misclassification patterns for future refinement.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

**Libraries:** `tensorflow` · `keras` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `Pillow`

**Techniques:** CNN · BatchNormalization · Dropout · MaxPooling · Data Augmentation · ImageDataGenerator · Adam Optimizer · Confusion Matrix · ROC Curve

---

## 📁 Repository Structure

```
CNN-Image-Classification/
│
├── covid19-ct-classification/
│   ├── covid19_cnn.py            # Full script: data loading, model, evaluation
│   └── README_covid19.md
│
├── cifar10-classification/
│   ├── cifar10_cnn.py            # Full script: data loading, model, evaluation
│   └── README_cifar10.md
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
- COVID-19 project: [Open in Colab](https://colab.research.google.com/drive/16EPBee4FmvsI5pvG2T13UUgJOwkZhYbT)
- CIFAR-10 project: [Open in Colab](https://colab.research.google.com/drive/1dFJO-Fd2ftZtaaFFSZ2QW43V8u6ziD5Q)

You'll need a `kaggle.json` API key for the COVID-19 dataset — download from [Kaggle account settings](https://www.kaggle.com/settings).

### Option B — Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Codewithpuji/CNN-Image-Classification.git
cd CNN-Image-Classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. COVID-19 project — download dataset manually**

Download from [Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset), unzip into `covid19-ct-classification/data/`. Remove Kaggle shell commands from top of script before running.

**4. CIFAR-10 project — dataset loads automatically**
```python
# Dataset is downloaded automatically via tf.keras.datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

**5. Run**
```bash
python covid19-ct-classification/covid19_cnn.py
python cifar10-classification/cifar10_cnn.py
```

---

## 📈 Visualisations Included

Both projects include:
- Training vs validation accuracy curves (per epoch)
- Training vs validation loss curves (per epoch)
- Confusion matrix heatmap
- Classification report (precision, recall, F1-score per class)
- Sample image grid from dataset

---

## 💡 Key Learnings

- CNNs achieve high accuracy on binary medical imaging tasks even with small datasets (930 images) when data augmentation is applied
- CIFAR-10's 32×32 resolution makes fine-grained classification challenging — batch normalization and dropout significantly improve generalisation
- Data augmentation is critical: without it, both models overfit within a few epochs

---

## 📚 Research Context

Both projects completed as part of the **MSc Data Analytics** programme at the **University for the Creative Arts, Germany (2024)**. The COVID-19 project addresses real-world healthcare robotics applications; the CIFAR-10 project benchmarks advanced CNN architectures on a standard computer vision dataset.

---

## 👩‍💻 Author

**Poojitha Kalyanam** — Data Analyst | MSc Data Analytics  
📍 Berlin, Germany  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/poojitha-kalyanam)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Codewithpuji)
