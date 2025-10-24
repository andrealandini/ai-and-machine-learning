# Quantitative Easing (QE) Prediction with Machine Learning  
**Universität Liechtenstein – Winter Semester 2025/26**  
**Prof. Dr. Merlin Bartel**  
**Authors:** Andrea Landini, Ali Yaghoubi

---

## 📘 Overview

This project investigates how **nonlinear interactions** between financial stress, monetary policy stance, and macroeconomic conditions can help **anticipate future Quantitative Easing (QE) actions** by the U.S. Federal Reserve.

We compare a **baseline econometric framework** (logistic regression) with **machine learning approaches** (Random Forest, XGBoost) using expanded feature engineering from daily financial data.

---

## 🧭 Project Structure

```
├── data/              # Raw data, processed datasets, and model outputs
├── final/src/         # RMarkdown analysis files
├── midterm/           # Midterm report
├── shortpaper/        # Final paper and presentation
└── qe_expanded_daily_to_quarterly.py  # Main Python pipeline
```

---

## ⚙️ Methodology

**Data & Features**
- FRED macroeconomic and financial series (daily frequency)
- Feature engineering: growth rates, volatility measures, rate spreads
- Quarterly aggregation for policy prediction

**Target Variable**
$$
QE_{t+1} = \begin{cases}
1 & \text{if Fed expands balance sheet significantly} \\
0 & \text{otherwise}
\end{cases}
$$

**Machine Learning Models**
- **Random Forest:** Ensemble of decision trees
  $$
  f_{\text{RF}}(X_t) = \frac{1}{B}\sum_{b=1}^{B} T_b(X_t)
  $$
- **XGBoost:** Gradient boosting with regularization
  $$
  \mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
  $$

**Validation**
- Time-series split to prevent look-ahead bias
- Performance metrics: Accuracy, AUC, feature importance

---

## 📊 Key Findings

- **Tree-based models** outperform logistic regression by capturing nonlinear relationships
- **Feature interactions** between financial stress and macroeconomic conditions are crucial for QE prediction
- **Expanded feature set** improves predictive accuracy over baseline models


---

## 📬 Contact

**Andrea Landini** — andrea.landini@uni.li  
**Ali Yaghoubi** — ali.yaghoubi@uni.li

---

*Universität Liechtenstein – AI and Machine Learning in Finance*