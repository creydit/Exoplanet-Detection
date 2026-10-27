# Exoplanet Detection from Kepler Light Curves using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Detecting stars that host exoplanets from their brightness over time, using signal processing and machine learning.

The final pipeline cleans each star's light curve, converts it into **FFT frequency-band features**, and classifies it with a **class-weighted Random Forest**. Compared with the original approach (raw flux + SMOTE + KNN), it improves planet-class **F1 from 0.13 to 0.78** and **ROC-AUC from 0.68 to 0.94**, while cutting false alarms from **146 to 3**.

---

## Table of Contents
- [Problem](#problem)
- [Dataset](#dataset)
- [Why Accuracy Is Misleading](#why-accuracy-is-misleading)
- [Approach](#approach)
- [Results](#results)
- [Experiments and Key Findings](#experiments-and-key-findings)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Using the Saved Model](#using-the-saved-model)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Problem

When a planet passes in front of its star (a **transit**), the star's brightness drops slightly. Because the planet keeps orbiting, these dips **repeat at a regular interval**.

**Task:** given a star's light curve (3,197 brightness measurements), predict whether the star hosts an exoplanet.

**Challenges:**
- **Extreme class imbalance:** fewer than 1% of stars have planets.
- **Small, noisy signals:** transit dips are often smaller than the star's natural variability and instrument noise.
- **Position variance:** a transit can occur at any point in the curve, so the same pattern appears at different time positions in different stars.

---

## Dataset

**Source:** [Exoplanet Hunting in Deep Space — Kepler Labelled Time Series Data (Kaggle)](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)

**Download:**
- **Kaggle page (manual):** open the link above → **Download** → you get a `.zip` containing `exoTrain.csv` and `exoTest.csv`.

Place `exoTrain.csv` and `exoTest.csv` in the same folder as the notebook.

| File | Stars | Planets | Non-planets |
|---|---|---|---|
| `exoTrain.csv` | 5,087 | 37 | 5,050 |
| `exoTest.csv` | 570 | 5 | 565 |
| **Merged (used)** | **5,657** | **42** | **5,615** |

- Each row is one star: a `LABEL` column (2 = planet, 1 = non-planet) followed by 3,197 flux values (`FLUX.1` … `FLUX.3197`).
- Both files were **merged** to maximise the number of planet examples (42 vs 37), and models were evaluated with **stratified 5-fold cross-validation** so every planet is tested exactly once.
- Planet fraction: **0.74%**.

---

## Why Accuracy Is Misleading

A model that predicts *"no planet"* for every star scores **99.26% accuracy** while finding zero planets. Accuracy is therefore reported only alongside the planet-class metrics:

| Metric | Question it answers |
|---|---|
| **Precision** | Of the stars flagged as planets, how many really are? |
| **Recall** | Of the real planets, how many did the model find? |
| **F1-score** | Balance of precision and recall |
| **ROC-AUC** | How well the model ranks planets above non-planets, independent of threshold (0.5 = random) |

---

## Approach

```
Raw light curve ──► Per-star cleaning ──► FFT band features ──► Prediction Model ──► Threshold 0.13 ──► Planet / No planet
   (3,197 values)                          (100 values)          (class-weighted)
```

### 1. Per-star light-curve cleaning
Each star is processed independently:

| Step | Purpose |
|---|---|
| Subtract a median filter (window 101) | Removes slow drift that hides small transit dips |
| Divide by the star's own standard deviation | Makes bright and faint stars comparable |
| Clip values above +4σ | Removes upward spikes (transits only cause downward dips) |
| Gaussian smoothing (σ = 2) | Reduces high-frequency noise |

The original project scaled each *time step* across all stars (column-wise `StandardScaler`), which has no physical meaning for time series and was fit before the train/test split (data leakage). Both issues are fixed.

### 2. FFT band features
The **Fast Fourier Transform** decomposes a curve into how strongly it repeats at each frequency. A planet orbiting regularly creates a peak in the spectrum **regardless of when the first transit occurs**, which makes these features shift-invariant.

- Compute the magnitude spectrum of each cleaned curve (~1,598 frequencies).
- Apply `log1p` to compress large values.
- Group the spectrum into **100 bands** and keep the maximum in each band.

This reduces each star from 3,197 time values to **100 frequency features**.

### 3. Random Forest with class weighting
- `RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample")`
- **Class weights replace SMOTE:** missed planets are penalised more heavily, without generating synthetic light curves (SMOTE interpolates between curves whose dips occur at different times, producing unrealistic samples).
- Chosen because trees select the informative frequency bands, capture interactions between bands, and handle imbalance and small data with minimal tuning.

### 4. Decision threshold
The default 0.5 threshold is too strict for a rare class. A threshold of **0.13** was selected from the precision-recall trade-off on cross-validated predictions.

---

## Results

All results below use **stratified 5-fold cross-validation on 5,657 stars (42 planets)**. Precision, recall and F1 refer to the **planet class**.

### Full model comparison

| # | Model | Features | Imbalance handling | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | Raw flux | SMOTE | 0.5 | 0.9477 | 0.0432 | 0.2857 | 0.0750 | 0.6066 |
| 2 | KNN (k=5) | Raw flux | SMOTE | 0.5 | 0.9691 | 0.0818 | 0.3095 | 0.1294 | 0.6821 |
| 3 | Logistic Regression | Cleaned + FFT | Class weights | 0.5 | 0.9855 | 0.3077 | **0.7619** | 0.4384 | **0.9679** |
| 4 | KNN (k=5) | Cleaned + FFT | SMOTE | 0.5 | 0.9604 | 0.1301 | **0.7619** | 0.2222 | 0.8783 |
| 5 | **Random Forest (final)** | **Cleaned + FFT** | **Class weights** | **0.13** | **0.9972** | **0.9062** | 0.6905 | **0.7838** | 0.9419 |

Rows 1–2 are the **baseline** (original approach). Rows 3–5 use the **same FFT features**, so they isolate the effect of the model choice.

### What the comparison shows

- **Features drive most of the improvement.** Switching from raw flux to cleaned FFT features lifts every model far above the baseline: ROC-AUC rises from 0.61–0.68 to 0.88–0.97, and recall more than doubles.
- **Logistic Regression ranks planets best** (highest ROC-AUC, 0.968) and finds the most planets (32 of 42), but at the default threshold it raises about 72 false alarms (precision 0.31).
- **KNN finds as many planets** but with many more false alarms (precision 0.13). It weighs all 100 bands equally in its distance measure, so uninformative bands add noise.
- **Random Forest gives the best balance** (F1 0.78): 29 planets found with only 3 false alarms. Precision is roughly 3× higher than Logistic Regression for a small loss in recall.
- **Choosing a model depends on the goal:**
  - **Fewest false alarms / best F1** → Random Forest (final model).
  - **Catch as many planets as possible** (e.g. shortlisting candidates for follow-up) → Logistic Regression.
- **Threshold note:** Random Forest uses a tuned threshold (0.13); Logistic Regression and KNN use 0.5. ROC-AUC is threshold-independent, and Logistic Regression's high value suggests that tuning its threshold could narrow the F1 gap.

### Confusion matrices

| | Baseline KNN + SMOTE | Final Random Forest |
|---|---|---|
| True negatives | 5,469 | **5,612** |
| False positives (false alarms) | 146 | **3** |
| False negatives (missed planets) | 29 | **13** |
| True positives (planets found) | 13 | **29** |

### Improvement summary

| Metric | Before | After | Change |
|---|---|---|---|
| Planets found | 13 / 42 | 29 / 42 | **2.2× more** |
| False alarms | 146 | 3 | **98% fewer** |
| Precision | 0.082 | 0.906 | **+0.82** |
| F1-score | 0.129 | 0.784 | **6× higher** |
| ROC-AUC | 0.682 | 0.942 | **+0.26** |

### Additional tree model tested

| Model | Features | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (class-weighted) | Cleaned + FFT | 0.906 | 0.690 | 0.784 | 0.942 |
| XGBoost (`scale_pos_weight`) * | Cleaned + FFT | 0.933 | 0.667 | 0.778 | 0.962 |

\* Evaluated at its own best-F1 threshold during experimentation. Random Forest and XGBoost perform comparably; Random Forest was kept for simplicity.

### Original project (for reference)
The original notebook used a single 80/20 split with only 7 test planets:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression + SMOTE | 0.9479 | 0.0208 | 0.1429 | 0.0364 |
| KNN + SMOTE | 0.9558 | 0.0476 | 0.2857 | 0.0816 |

---

## Experiments and Key Findings

| Experiment | Precision | Recall | F1 | ROC-AUC | Outcome |
|---|---|---|---|---|---|
| Baseline KNN on raw flux | 0.082 | 0.310 | 0.129 | 0.682 | Starting point |
| KNN on cleaned curves | 0.690 | 0.476 | 0.563 | 0.737 | Large jump, but see shift test |
| KNN on cleaned curves, randomly time-shifted | 0.000 | 0.000 | 0.000 | 0.499 | Collapsed to random |
| Random Forest on 14 hand-crafted features | 0.400 | 0.095 | 0.154 | 0.790 | Better ranking, weak detection |
| **Random Forest on FFT bands** | **0.906** | **0.690** | **0.784** | **0.942** | **Final model** |

**1. Accuracy hides failure.** Every model scored above 94% accuracy, including ones that found almost no planets.

**2. Per-star cleaning matters.** Removing drift and normalising each star individually was the first major improvement.

**3. The shift test exposed a misleading result.** KNN on cleaned curves looked strong (F1 0.56), but randomly shifting each curve in time dropped it to random (ROC-AUC 0.50). KNN compares curves point by point, so it was matching time-aligned structure shared across stars rather than detecting transits. A genuine transit detector should not depend on *when* a dip occurs.

**4. Shift-invariant features are the key.** FFT features describe *how regularly* the brightness dips, not *where*, and produced the largest and most trustworthy improvement.

**5. Hand-crafted features were not enough.** Statistics such as dip count, dip depth and spacing regularity separated the classes only weakly; most stars with several dips were non-planets with noisy or flaring light curves.

---

## Project Structure

```
├── Exoplanet_Detection_by_ML_v2.ipynb   # Full pipeline: data → cleaning → features → models → evaluation
├── exoTrain.csv                         # Kaggle training data (download separately)
├── exoTest.csv                          # Kaggle test data (download separately)
├── exoplanet_rf_fft.joblib              # Saved final model + threshold (generated)
└── README.md
```

### Notebook sections
1. Load and merge data
2. Per-star light-curve cleaning (with raw vs cleaned plots)
3. FFT band feature extraction
4. Evaluation function (metrics, classification report, confusion matrix)
5. Baseline: raw flux + SMOTE + Logistic Regression / KNN
6. Final model: Random Forest on FFT features
   - 6b. Logistic Regression and KNN on FFT features
7. ROC curve comparison and summary table
   - 7b. Extra check: single 80/20 train-test split
8. Train on all data, save model, predict a single star

---

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn joblib
```

1. Download `exoTrain.csv` and `exoTest.csv` from Kaggle into the project folder.
2. Open `Exoplanet_Detection_by_ML_v2.ipynb` and run all cells.

The baseline section (Section 5) runs SMOTE and KNN on 3,197-dimensional data and takes a few minutes.

---

## Using the Saved Model

```python
import joblib

bundle = joblib.load("exoplanet_rf_fft.joblib")
model, threshold = bundle["model"], bundle["threshold"]

# flux: 1D array of 3,197 brightness values
features = fft_bands(clean_star(flux)[None, :])   # functions defined in the notebook
prob = model.predict_proba(features)[0, 1]
is_planet = prob >= threshold
```

The notebook also provides `predict_star(flux)`, which returns `(probability, is_planet)`.

---

## Limitations

- **Small number of positives.** With 42 planets, one planet changes recall by about 2.4 percentage points, so differences of a few points between models are within noise. A realistic F1 range for the final model is **0.75–0.80**.
- **Threshold chosen on cross-validated predictions.** This makes the reported F1 slightly optimistic; a nested evaluation would give a stricter estimate.
- **Single 80/20 split is noisy.** It contains only about 8 test planets and is included as a supplementary check, not the main result.
- **Dataset scope.** The Kaggle data covers one observing campaign; performance on other missions (for example TESS) is not tested.

---

## Future Work

- **More labelled data:** add confirmed planets and false positives from the NASA Exoplanet Archive using the `lightkurve` library.
- **Box Least Squares (BLS) features:** the standard astronomical transit search, giving period, depth and duration estimates.
- **Data augmentation:** time shifts, flips and noise on planet curves to enlarge the minority class.
- **1D CNN:** learn transit shapes directly from cleaned or phase-folded curves.
- **Nested cross-validation** for unbiased threshold selection and hyperparameter tuning.

---

## 📜 License

This project is intended for educational and academic purposes.
