# Quantitative ML Trading Strategy

A machine learning-based quantitative trading project that explores short-term trading signals using financial time-series data.

The goal of this project is not to present a production-ready trading system, but to build a realistic research pipeline and understand the challenges of applying machine learning to financial markets.

---

## Overview

This project implements an end-to-end trading research workflow:

- Downloading historical market data
- Creating technical and statistical features
- Training machine learning models on time-series data
- Using walk-forward validation to avoid data leakage
- Generating long/short trading signals
- Backtesting a multi-asset portfolio
- Evaluating performance using cumulative returns and Sharpe ratio

---

## Key Features

### Feature Engineering
The strategy uses features such as:

- Daily returns
- Moving averages
- Momentum
- Volatility
- RSI
- Trend indicators

### Machine Learning
The model predicts short-term price direction using historical features.

The output is treated as a confidence score, which is converted into trading signals using ranking-based thresholds.

### Walk-Forward Validation
The model is trained only on past data and tested on future periods.

This helps reduce lookahead bias and better simulates real-world trading conditions.

### Strategy Logic
The strategy uses:

- Long positions on high-confidence signals
- Short positions only on extreme low-confidence signals
- Trend filtering
- Transaction cost modeling
- Multi-asset portfolio construction

---

## Results

The strategy showed mixed results. It performed better in some market periods but struggled to generalize consistently across changing market regimes.

This was an important part of the project: understanding that building a profitable trading strategy is significantly harder than training a machine learning model.

### Strategy Performance

![Strategy Performance](images/strategy_performance.png)

---

## Key Learnings

Some of the main lessons from this project:

- Financial markets are noisy and hard to predict
- Model accuracy does not necessarily translate into trading profitability
- Avoiding data leakage is critical
- Walk-forward validation is essential for realistic evaluation
- Signal quality matters more than trade quantity
- Short signals were harder to model reliably than long signals

---

## Research Notes

This project intentionally focuses on building a realistic research pipeline rather than presenting a guaranteed profitable strategy.

During experimentation, the model showed mixed results and struggled to generalize across different market regimes. This highlighted several important challenges in quantitative trading:

- Financial data is noisy

- Prediction confidence is often concentrated around 0.5

- More trades do not necessarily improve performance

- Short signals were less reliable than long signals

- Walk-forward validation is essential to avoid unrealistic results

---

## Project Structure

```text
.
├── main.py
├── src/
│   └── data_loader.py
├── images/
│   └── strategy_performance.png
├── requirements.txt
└── README.md
```
---
## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

## 👨‍💻 Author

Yaniv Shaino
https://www.linkedin.com/in/yaniv-shaino-768213225/

