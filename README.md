# Quantitative Easing (QE) Prediction with Machine Learning  
**Universität Liechtenstein – Winter Semester 2025/26**  
**Prof. Dr. Merlin Bartel**  
**Authors:** Andrea Landini (<andrea.landini@uni.li>), Ali Yaghoubi (<ali.yaghoubi@uni.li>)

---

## 📘 Overview

This project investigates how **nonlinear interactions** between financial stress, monetary policy stance, and macroeconomic conditions can help **anticipate future Quantitative Easing (QE) actions** by the U.S. Federal Reserve.

We begin with a **baseline econometric framework** (logistic regression) using limited FRED variables, and then extend it through **Python-based feature engineering** and **machine learning models** (Random Forest, XGBoost).  
The expanded approach builds a richer dataset from daily macro-financial data and aggregates it to a quarterly format to improve QE prediction accuracy.

---

## 🧭 Project Structure

```

├── Handbook-Part-1.ipynb                     # Handbook notebook explaining theory, equations, and implementation
├── README.md                                 # This file
├── ai-and-machine-learning.Rproj             # R project configuration file
│
├── data/                                     # All raw, intermediate, and model outputs
│   ├── closes_clean.csv                      # Historical S&P 500 close prices
│   ├── expanded_daily_all.csv                # Daily merged FRED dataset (policy, market, macro variables)
│   ├── expanded_qe_dataset.rds               # R-based QE dataset (baseline model)
│   ├── expanded_qe_dataset_daily_to_quarterly_py.parquet  # Python-engineered quarterly dataset
│   ├── expanded_quarterly_features_py.csv    # Final feature matrix (quarterly, with lags & flags)
│   ├── model_comparison_py.csv               # RF vs XGBoost performance metrics (Accuracy, AUC)
│   ├── output.txt                            # Model training and evaluation logs
│   ├── qe-results01.png                      # ROC curve from R baseline model
│   ├── rf_feature_importance_py.csv          # Random Forest feature importance (Python)
│   ├── roc_expanded_model_py.png             # ROC comparison plot (RF vs XGB)
│   ├── sp500_tickers.csv                     # Reference equity ticker list (for future extensions)
│   └── xgb_feature_importance_py.csv         # XGBoost feature importance (Python)
│
├── final/
│   ├── src/
│   │   ├── fred-expanded-model.Rmd           # RMarkdown: Extended QE model (Random Forest + Logistic comparison)
│   │   ├── fred-model01.Rmd                  # RMarkdown: Baseline logistic and RF model
│   │   └── fred-model02.Rmd                  # RMarkdown: Additional analysis and model extensions
│   └── utils/
│       ├── data-processing.Rmd               # Data cleaning and preprocessing functions
│       └── df_stats.Rmd                      # Utility for descriptive statistics
│
├── midterm/
│   └── main.pdf                              # Midterm report / presentation slides
│
├── qe_expanded_daily_to_quarterly.py         # Main Python script (daily → quarterly FRED model, RF + XGB training)
│
└── shortpaper/
├── Master_Pitch_Template_with_cues_for_pitcher.docx   # Presentation draft
├── ai_ml_shortpaper.pdf                # Short paper summarizing the ML contribution
└── berahas2021.pdf                     # Reference paper on optimization and ML techniques

````

---

## ⚙️ Methodology Summary

1. **Data Acquisition**  
   - FRED series via API (`fredapi`): Fed balance sheet, interest rates, VIX, unemployment, CPI, M2, etc.  
   - Daily frequency merged → forward-filled to ensure complete panel coverage.

2. **Feature Engineering**  
   - Compute within-quarter growth, realized volatility, rate slopes, and binary stress indicators.  
   - Aggregate to quarterly level:  
     \[
     X_t = [\text{Fed\_Growth}_t, \text{M2\_Growth}_t, \text{VIX\_realized\_vol}_t, \text{Rate\_Slope}_t, \ldots]
     \]

3. **Target Variable (QE Decision)**  
   \[
   QE_{t+1} =
   \begin{cases}
   1 & \text{if } \text{Fed\_Securities}_{t+1} - \text{Fed\_Securities}_t > 100 \\
   0 & \text{otherwise}
   \end{cases}
   \]
   Predicts whether the Fed increases securities holdings by more than \$100B next quarter.

4. **Models Implemented**
   - **Random Forest (RF):**  
     \[
     f_{\text{RF}}(X_t) = \frac{1}{B}\sum_{b=1}^{B} T_b(X_t)
     \]
     Ensemble of 500 trees with class balancing and feature subsampling.

   - **XGBoost (XGB):**  
     \[
     \mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \big(\gamma T_k + \tfrac{1}{2}\lambda \|w_k\|^2\big)
     \]
     Gradient-boosted trees with regularization and sequential error correction.

5. **Validation**  
   - Time-based 70/30 split to avoid look-ahead bias.  
   - Metrics: Accuracy, ROC AUC, confusion matrix, and feature importance.

---

## 📊 Results Summary

| Model | Accuracy | AUC | Key Insight |
|-------|-----------|-----|-------------|
| Logistic Regression | 0.73 | 0.76 | Captures linear effects but limited flexibility |
| Random Forest | 0.77 | 0.85 | Detects nonlinear & interaction effects |
| XGBoost | ~0.80 | ~0.87 | More stable with regularization, learns sequentially |

**Key takeaway:** Ensemble methods outperform the linear baseline, capturing **nonlinear**, **state-dependent** policy responses.

---

## 🧩 Extensions

- Add higher-frequency stress indices (MOVE, FCI).
- Implement **rolling backtests** and **model stability checks**.
- Explore **interpretable ML** (e.g., SHAP values) for economic insights.
- Extend horizon: predict \( QE_{t+h} \) for \( h=2,3 \) quarters ahead.

---

## 🧠 Citation

If referencing this work, please cite:

> Landini, A., Yaghoubi, A. (2025). *Predicting Federal Reserve Quantitative Easing Decisions Using Machine Learning.*  
> Universität Liechtenstein, Winter Semester 2025/26. Prof. Dr. Merlin Bartel.

---

## 💻 Environment Setup

```bash
# Create and activate environment
conda create -n qe_ml python=3.11
conda activate qe_ml

# Install dependencies
pip install pandas numpy matplotlib scikit-learn xgboost fredapi pyarrow

# Run the main pipeline
python qe_expanded_daily_to_quarterly.py
````

Outputs (CSV + PNG) will be stored automatically in `data/`.

---

## 📬 Contact

For questions or collaboration:

* **Andrea Landini** — [andrea.landini@uni.li](mailto:andrea.landini@uni.li)
* **Ali Yaghoubi** — [ali.yaghoubi@uni.li](mailto:ali.yaghoubi@uni.li)

---

**Universität Liechtenstein**
*Master in Finance – Artificial Intelligence and Machine Learning (Winter 2025/26)*

