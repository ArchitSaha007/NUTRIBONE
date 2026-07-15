# 🦴 NutriBone
## AN EXPLAINABLE ENSEMBLE DEEP LEARNING FRAMEWORK FOR OSTEOPOROSIS DETECTION FROM KNEE X-RAY IMAGES.

> An end-to-end DL framework for automated osteoporosis screening using knee X-ray images, combining image enhancement, transfer learning, ensemble deep learning, confidence calibration, and explainable AI.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📖 Overview

**NutriBone** is a research-oriented Medical Artificial Intelligence framework developed to assist in the **early screening of osteoporosis** using **knee X-ray images**.

Osteoporosis is a progressive skeletal disorder characterized by reduced bone mineral density and deterioration of bone microarchitecture, significantly increasing the risk of fractures. Since the disease often progresses without noticeable symptoms, early detection remains one of the biggest clinical challenges.

The goal of this project is to investigate whether conventional **knee radiographs**, enhanced by modern Artificial Intelligence techniques, can serve as an accessible and cost-effective screening alternative to support clinical decision making.

NutriBone combines advanced Computer Vision and Deep Learning techniques including:

- Medical Image Enhancement
- Transfer Learning
- Ensemble Deep Learning
- Test-Time Augmentation (TTA)
- Confidence Calibration
- Explainable AI (Grad-CAM)

The framework classifies knee X-ray images into three clinically significant categories:

- 🟢 Normal
- 🟡 Osteopenia
- 🔴 Osteoporosis
- 
---

# ✨ Key Features

- 🩻 Automated osteoporosis screening from knee X-rays
- 🔍 Medical image enhancement using CLAHE
- 🧠 Transfer Learning with state-of-the-art CNN architectures
- 📊 Multi-model Ensemble Learning
- 📈 Test-Time Augmentation (TTA)
- 🎯 Confidence Calibration using Temperature Scaling
- 🔥 Explainable AI with Grad-CAM visualization
- ⚕️ Three-class osteoporosis classification
- 📉 Medical threshold optimization for improved sensitivity

---

# 🔬 Methodology

## 1. Image Preprocessing

Medical images are enhanced using **OpenCV** before being passed to the neural networks.

Preprocessing includes:

- Image loading
- Image resizing
- Image normalization
- CLAHE enhancement
- Contrast improvement
- Noise reduction

This stage improves the visibility of trabecular bone structures and prepares the images for deep feature extraction.

---

## 2. Deep Feature Extraction

Three independent CNN architectures learn complementary representations of bone structures.

### EfficientNetV2B2

- Efficient feature learning
- High accuracy-to-parameter ratio
- Excellent transfer learning performance

### DenseNet201

- Dense connectivity
- Superior feature reuse
- Strong texture representation
- Effective for medical image analysis

### ConvNeXtTiny

- Modern CNN architecture
- Enhanced global feature learning
- Improved generalization capability

---

## 3. Ensemble Learning

Instead of relying on a single model, predictions from all three networks are combined using **Probability Averaging Ensemble**.

Benefits include:

- Improved robustness
- Better generalization
- Reduced overfitting
- More stable predictions

---

## 4. Confidence Calibration

Deep neural networks often produce overconfident probability estimates.

Temperature Scaling is applied to calibrate prediction confidence, producing more reliable probabilities suitable for medical decision support.

---

## 5. Explainable AI

Grad-CAM generates visual heatmaps highlighting image regions responsible for each prediction.

This improves:

- Model transparency
- Clinical interpretability
- Trust in AI-assisted diagnosis

---

### Classification Categories

- Normal
- Osteopenia
- Osteoporosis

The dataset contains labeled knee radiographs suitable for supervised deep learning and transfer learning research.

---

# 🛠 Technology Stack
### Frontend
-html

### Backend

- Python

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV

### Machine Learning

- Scikit-learn

### Scientific Computing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Grad-CAM

---

# 📁 Project Pipeline

Dataset
   │
   ▼
Image Preprocessing
(OpenCV + CLAHE)
   │
   ▼
Data Augmentation
   │
   ▼
Transfer Learning
   │
   ▼
Feature Extraction
   │
   ▼
Ensemble Learning
   │
   ▼
Confidence Calibration
   │
   ▼
Explainable AI
   │
   ▼
Prediction

---

# 🌍 Project Contributions

NutriBone integrates multiple modern AI techniques into a unified osteoporosis screening framework.

| Conventional Approach | NutriBone Framework |
|-----------------------|---------------------|
| Single CNN | Multi-model Ensemble |
| Standard preprocessing | CLAHE Enhancement |
| Raw Softmax confidence | Temperature Scaling |
| Black-box prediction | Explainable AI (Grad-CAM) |
| Single inference | Test-Time Augmentation |
| Limited robustness | Ensemble-based prediction |

# ⚠ Disclaimer

NutriBone is a **research and educational project** developed to explore the application of Artificial Intelligence in osteoporosis screening.

It is **not a certified medical diagnostic system** and should not be used as a replacement for professional clinical diagnosis or medical advice.

Healthcare decisions should always be made by qualified medical professionals using appropriate clinical evaluation and diagnostic procedures.
