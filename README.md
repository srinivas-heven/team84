# ⚠️ Diaster Detector

**An advanced Streamlit web application for assessing fire and flood risks using cutting-edge computer vision, machine learning, and geospatial feature analysis.**

---

## 🚀 Overview

This platform leverages advanced preprocessing, vegetation and terrain analysis, and ML integration to provide robust risk assessments for:

- 🔥 **Fire Risk**
- 💧 **Flood Risk**
- 📊 **Combined Risk**

Designed for environmental professionals, researchers, and emergency responders to analyze satellite or drone imagery.

---

## 🎯 Features

### 🔥 Fire Risk Analysis

- NDVI-based vegetation health detection
- Color stress mapping (brown/yellow vegetation)
- Texture analysis using LBP and Gabor filters
- KMeans-based fire-prone color clustering
- Vegetation fragmentation, shadow detection
- Entropy and brightness statistics
- Rule-based and ML-based scoring

### 💧 Flood Risk Analysis

- Multi-spectral water detection (clear, muddy, dark)
- Terrain slope and elevation analysis
- Impervious surface vs vegetation coverage
- Water body morphology and size detection
- Reflective surface + entropy metrics
- Rule-based and ML-based flood predictions

### 📊 Visualization & Reporting

- Interactive risk level pie/bar charts (Matplotlib)
- Confidence indicators with contextual notes
- Downloadable detailed risk report (.txt)
- Combined fire & flood analysis

---
## 🧠 Technologies Used

- `Streamlit` – web UI
- `OpenCV` – image processing
- `NumPy` / `Pandas` – data handling
- `Matplotlib` / `Seaborn` – visualization
- `Scikit-learn` – ML models
- `scikit-image` – advanced image features
- `Pillow` – image I/O
- `Joblib` – (placeholder for model loading)
- `TensorFlow` – (imported, not currently used)

---


## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-risk-assessment-platform.git
cd ai-risk-assessment-platform
