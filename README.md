# Energy Load Forecasting — Tabular Regression vs Foundation Models

> **Educational Resource** — This repository teaches practical time-series forecasting by contrasting two fundamentally different paradigms: *classical gradient boosting with hand-crafted features* vs. *large pre-trained foundation models*. Each notebook is self-contained and heavily annotated for learning.

---

## Learning Objectives

By working through these notebooks you will be able to:

1. **Reframe** a time-series problem as supervised tabular regression and understand when that is appropriate
2. **Engineer** temporal features (calendar, lag, rolling statistics, cyclical encoding) from a raw timestamp + target series
3. **Contrast** two multi-step forecasting strategies — *direct multi-output* vs. *recursive walk-forward* — and understand their trade-offs
4. **Apply** gradient boosting (CatBoost) to a structured forecasting task with proper chronological train/test splits
5. **Use** a pre-trained foundation model (IBM TTM Granite) for zero-shot time-series forecasting without any training
6. **Evaluate** and **compare** models with horizon-aware metrics (MAE, RMSE per forecast step)

---

## Overview

This project compares **traditional ML (CatBoost)** against a **foundation model (IBM TTM Granite)** for predicting Spain's total electricity load **96 hours (4 days) ahead**, using 4 years of historical grid and weather data (2015–2018).

Three forecasting strategies are provided in this repository, all benchmarked on the same test set (2018) for a fair comparison:

| # | Notebook | Strategy | Training Required | Feature Engineering |
|---|----------|----------|-------------------|---------------------|
| 1 | `1.TTM_Granite_Zeroshot.ipynb` | IBM TTM Granite R2 — pure zero-shot inference | ❌ None | ❌ None |
| 2 | `2.Catboost_Tabular_WalkForward.ipynb` | Single t+1 model applied recursively 96 times | ✅ Yes | ✅ 70+ features |
| 3 | `3.Catboost_Tabular_Direct.ipynb` | 96 independent CatBoost models (direct multi-output) | ✅ Yes | ✅ 70+ features |

---

## Results

All models evaluated on overlapping 96-hour forecast windows across the 2018 test year:

| Rank | Model (Notebook) | MAE (MW) | RMSE (MW) | Training | Features |
|------|------------------|----------|-----------|----------|----------|
| 🥉 | TTM Granite Zero-Shot (Notebook 1) | 2,181.5 | 2,920.5 | 0 (pre-trained) | 0 |
| 🥈 | CatBoost Walk-Forward (Notebook 2) | 1,757.7 | 2,478.3 | 1 model | 70+ |
| 🥇 | **CatBoost Direct (Notebook 3)** | **1,380.7** | **1,998.6** | 96 models | 70+ |

**Key lesson**: Carefully engineered tabular features still give gradient boosting a meaningful accuracy advantage over a zero-shot foundation model on domain-specific data — but TTM achieves competitive results *without a single line of training code*.

> **Note on TTM:** Uses `ibm-granite/granite-timeseries-ttm-r2` (branch `1536-96-r2`) configured for 1,536-hour context (~64 days) and 96-step output.

---

## Core Concepts Explained

### Concept 1 — Tabular Regression for Time Series

The classical deep learning approach treats time series as sequential data (RNNs, LSTMs, Transformers). The **tabular approach** instead converts the problem into standard supervised regression where each row is independent:

| Feature Type | Examples | What It Captures |
|---|---|---|
| Calendar | hour, weekday, month, week-of-year | Daily / weekly / seasonal cycles |
| Cyclical Encoding | `sin(2π·hour/24)`, `cos(2π·hour/24)` | Circular continuity (hour 23 → hour 0) |
| Lag Features | value at t−1, t−3, t−6, …, t−144 | Memory of past values at fixed offsets |
| Rolling Means | MA-1h, MA-6h, MA-24h, MA-7d | Smoothed trend at different time scales |
| Rolling Std Dev | Std-12h, Std-24h, Std-7d, Std-30d | Demand volatility signals |
| Differencing | Δ1, Δ3, Δ24, Δ48, Δ96 | Rate of change / momentum |

This produces 70+ features per row, letting a tree-based model like CatBoost learn complex non-linear temporal patterns *without any sequential architecture*.

### Concept 2 — Multi-Step Forecasting Strategies

Once we have a trained model, there are two main ways to produce a multi-step forecast:

**Strategy A — Direct Multi-Output**
- Train a *separate* model for each forecast horizon: model₁ predicts t+1, model₂ predicts t+2, …, model₉₆ predicts t+96
- Each model is independently optimised → no error accumulation
- Requires training and storing 96 models

**Strategy B — Recursive Walk-Forward**
- Train *one* model to predict t+1 only
- At inference time, feed each prediction back as input to predict the next step
- Errors compound across the 96 steps, but only one model needed

### Concept 3 — Foundation Models (Zero-Shot)

IBM's Tiny Time Mixer (TTM) is a lightweight transformer pre-trained on a large corpus of diverse time series. Key ideas:

- **No feature engineering**: the model reads raw historical values as its input context
- **No retraining**: zero-shot inference — load the weights, pass context, get forecast
- **Long context window**: 1,536 continuous hours (~64 days) as input vs. discrete lag offsets in the tabular approach

---

## Prerequisites

| Topic | Assumed Knowledge |
|---|---|
| Python | Comfortable with pandas, numpy, matplotlib |
| Machine Learning | Knows what train/test split means; familiar with regression metrics (MAE, RMSE) |
| Time Series | Basic understanding of seasonality and trend is helpful but not required |
| Deep Learning | Not required for CatBoost notebooks; basic familiarity helpful for TTM notebook |

---

## Project Structure

```
├── 1.TTM_Granite_Zeroshot.ipynb            # Notebook 1: Zero-shot foundation model (IBM TTM Granite R2)
├── 2.Catboost_Tabular_WalkForward.ipynb    # Notebook 2: Recursive walk-forward (1 CatBoost model)
├── 3.Catboost_Tabular_Direct.ipynb         # Notebook 3: Direct multi-output (96 CatBoost models)
├── Data/
│   ├── energy_data.csv                   # Spanish electricity grid data, hourly 2015–2018
│   └── weather_data.csv                  # Hourly weather observations (temperature, wind, etc.)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/patcharafyo/Forecasting_Tabular_Sharing.git
cd Forecasting_Tabular_Sharing
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 3. Recommended notebook order

Start with Notebook 1 to see how a foundation model can forecast with zero feature engineering, then move to Notebook 2 to learn the walk-forward recursive strategy with hand-crafted features, and finally Notebook 3 to see how the direct multi-horizon approach achieves the best accuracy.

```
Notebook 1  →  Notebook 2  →  Notebook 3
(Zero-Shot TTM)  (Walk-Forward)   (Direct)
```

---

## Data

- **Source**: Spanish electricity grid — generation by source, total load, price (hourly, 2015–2018) + paired weather observations
- **Target variable**: `total load actual` (MW)
- **Training period**: 2015-01-01 to 2017-10-28 (~24,800 hours)
- **Test period**: 2018-01-01 onwards (~8,760 hours)
- **Deliberate gap**: A gap exists between the training end and test start to prevent leakage from rolling features that look back in time

---

## Suggested Discussion Questions

After running the notebooks, consider:

1. Why does the Direct strategy outperform Walk-Forward? At which horizon does the gap become significant?
2. Why are cyclical (sin/cos) encodings preferred over raw hour/month integers for gradient boosting?
3. TTM achieves reasonable accuracy with **zero feature engineering and zero training**. What does this imply for future forecasting workflows?
4. What would happen to CatBoost performance if you removed all lag features? Try it!
5. The test set starts in 2018. Why is a *time-based* split essential here, and what goes wrong with a random split?

## Key Takeaways

1. **Feature engineering + CatBoost is hard to beat.** The direct multi-output strategy with 70+ hand-crafted features achieves the lowest MAE by a wide margin.

2. **Recursive walk-forward compounds errors.** The single-model recursive approach is simpler but accumulates prediction errors over the 96-step horizon, widening the gap vs direct multi-output.

3. **Zero-shot foundation models are viable baselines.** TTM Granite requires zero effort (no features, no training) and still produces reasonable forecasts — useful for quick prototyping or when domain expertise is limited.

4. **Fine-tuning the prediction head bridges the gap.** By freezing the backbone and training only the prediction head, we combine pre-trained temporal knowledge with domain-specific adaptation — all with zero feature engineering.

## Requirements

- Python 3.10+
- CatBoost, pandas, numpy, matplotlib, scikit-learn
- `granite-tsfm[notebooks]` + PyTorch (for TTM Granite notebook only)
- See `requirements.txt` for full dependency list
