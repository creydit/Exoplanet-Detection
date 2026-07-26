# 🌌 Exoplanet Detection using Machine Learning

> A machine learning project that detects exoplanets from stellar light curves using data from NASA's Kepler Mission.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 Overview

Exoplanets are planets that orbit stars outside our solar system. One of the most successful methods for discovering them is the **Transit Method**, where a planet passing in front of its host star causes a small dip in the observed brightness.

This project applies **Machine Learning** to automatically classify stars as containing an exoplanet or not by analyzing their light curves obtained from the **NASA Kepler Mission** dataset.

The project also addresses the severe class imbalance present in the dataset using **SMOTE (Synthetic Minority Oversampling Technique)** and compares the performance of multiple machine learning models.

---

## 🚀 Features

- Preprocesses real astronomical data
- Visualizes stellar light curves
- Handles highly imbalanced datasets using SMOTE
- Trains multiple machine learning models
- Compares model performance
- Displays confusion matrices and classification reports
- Easy to reproduce using Jupyter Notebook

---

## 📂 Project Structure

```
.
├── Exoplanet Detection by ML.ipynb     # Main notebook
├── exoTrain.csv                        # Dataset (download separately)
├── README.md
└── images/                             # Optional plots/screenshots
```

---

## 📊 Dataset

**Source:** NASA Kepler Mission (via Kaggle)

Dataset Characteristics:

- 3,197 flux measurements for each star
- Binary classification problem
- Label **1** → Exoplanet detected
- Label **0** → No exoplanet
- Extremely imbalanced dataset (≈99% negative samples)

Each row represents the brightness (flux) of a star measured over time.

---

## 🧠 Machine Learning Pipeline

### 1. Data Loading

- Read the Kepler dataset
- Separate labels and flux values

### 2. Data Visualization

- Plot sample light curves
- Observe transit dips

### 3. Data Preprocessing

- Feature scaling using **StandardScaler**
- Train-test split
- Handle class imbalance using **SMOTE**

### 4. Model Training

Implemented models:

- Logistic Regression
- K-Nearest Neighbors (KNN)

### 5. Evaluation

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/exoplanet-detection-ml.git

cd exoplanet-detection-ml
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
Exoplanet Detection by ML.ipynb
```

---

## ▶️ Running the Project

1. Download the Kepler dataset (`exoTrain.csv`).
2. Place it in the project directory.
3. Open the notebook.
4. Run all cells sequentially.

---

## 📈 Results

The models were compared on their ability to correctly detect exoplanets.

Key observations:

- SMOTE significantly improved the detection of minority-class samples.
- K-Nearest Neighbors achieved better **Recall** than Logistic Regression.
- Higher recall is particularly valuable because missing an actual exoplanet is more costly than generating additional false positives.

---

## 🔍 Sample Workflow

```
Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Scaling
     │
     ▼
Train-Test Split
     │
     ▼
SMOTE Balancing
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     └── KNN
     │
     ▼
Model Evaluation
     │
     ▼
Prediction
```

---

## 🌟 Future Improvements

- Deep Learning using 1D Convolutional Neural Networks (1D CNN)
- Random Forest and XGBoost models
- Hyperparameter optimization
- Cross-validation
- Integration with TESS mission data
- Deployment as a web application

---

## 📚 Learning Outcomes

Through this project, the following concepts were explored:

- Machine Learning for classification
- Astronomical time-series analysis
- Data preprocessing
- Feature scaling
- Handling imbalanced datasets
- Model evaluation techniques
- Scientific data visualization

---

## 📜 License

This project is intended for educational and academic purposes.
