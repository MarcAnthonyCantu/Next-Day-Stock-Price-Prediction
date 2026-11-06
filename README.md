# Next-Day Stock Price Prediction with PCA + Neural Networks

Predicting next-day closing prices for a basket of four stocks from a 10-day
window of engineered intraday features, using principal component analysis for
dimensionality reduction and a multilayer perceptron for multi-output
regression. The focus is on a disciplined modeling pipeline, feature
construction, from-scratch PCA, and a controlled comparison of network
architectures rather than on claiming a tradeable edge.

## Overview

The task is framed as **multi-output regression**: from recent price and volume
history, predict the next day's closing price for **GE, JPM, MSFT, and RTX**
simultaneously. The pipeline standardizes and compresses the feature space with
PCA, then trains and compares several MLP architectures, selecting width, depth,
and training length on held-out data.

- **Data:** 5 minute intraday bars (2010–2023), reduced to daily close/volume
  and engineered into per-day summary features.
- **Features:** each sample stacks a **10-day window** of engineered
  price/volume features; targets are the four stocks' next-day closes.
- **Reduction:** standardization → correlation-matrix PCA, keeping the smallest
  number of components explaining ≥ 95% of variance (r = 128).
- **Model:** scikit-learn `MLPRegressor`, with a comparison of one wide layer
  vs. two stacked layers at matched parameter budgets.

## Method

1. **Data preparation** — Intraday bars are filtered to 2010–2023, reduced to
   closing price and volume, aligned to a common set of trading days across all
   stocks, and cleaned.
2. **Feature engineering** — For each stock and day, a set of summary features
   is computed from that day's intraday prices and volumes. A sample `X(n)` is
   the flattened **10-day window** ending on day *n*; the target `Y(n)` is the
   next day's close for the four selected stocks.
3. **Dimensionality reduction** — Features are standardized and PCA is performed
   **from scratch** via eigendecomposition of the correlation matrix. The
   percentage of variance explained (PVE) curve sets the number of components
   retained (smallest *r* with PVE ≥ 0.95).
4. **Architecture selection** — The hidden-layer size is chosen relative to the training-set size. Training uses Adam, batch size 50, and MSE loss.
5. **Training diagnostics** — Train and test RMSE are tracked per epoch, along
   with a normalized gradient norm, and the optimal epoch is chosen at minimum
   test RMSE (early-stopping by selection).
6. **Architecture comparison** — Three networks are compared at similar parameter constraints:
   a single wide layer `MLP(2g)`, a wider single layer `MLP(5g)`, and a deeper
   two-layer network `MLP(g, g)`.
7. **Per-stock evaluation** — Predictions are inverted back to raw prices and
   average relative error is reported per stock, on both train and test sets.

## Results

Test RMSE (standardized targets) across the three architectures:

| Model | Hidden units | Train RMSE | Test RMSE |
|-------|--------------|-----------|-----------|
| MLP(2g) | (168) | 0.096 | 0.169 |
| **MLP(5g)** | **(420)** | **0.088** | **0.130** |
| MLP(g, g) | (84, 84) | 0.085 | 0.173 |

The deeper network `MLP(g, g)` had the lowest *training* RMSE but the highest
*test* RMSE which is a sign of overfitting. At matched parameter count, the
**wider** single-hidden-layer network `MLP(5g)` generalized best.

Average test relative error per stock (best model): GE ≈ 4.9%, JPM ≈ 6.6%,
MSFT ≈ 16.3%, RTX ≈ 3.0%.

## Repository contents

| File | Description |
|------|-------------|
| `code.ipynb` | Full pipeline — data prep, feature engineering, PCA, MLP training and comparison |
| `Report.pdf` | Written report with figures, derivations, and discussion of results |

## Tech stack

- **Python** — NumPy, pandas, matplotlib
- **scikit-learn** — `MLPRegressor`, `train_test_split`, metrics
- Jupyter Notebook

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook code.ipynb
```

Because the raw intraday files aren't included, start from the feature-matrix cell (the one that loads tables/W.csv) and run forward — PCA, the architecture comparison, and the per-stock evaluation all reproduce from the included tables/. The earlier data-cleaning and feature-engineering cells are kept in the notebook to document how tables/ was produced, but need the raw 5-minute data to re-run.

## Background

This project was developed for a graduate course in mathematical machine
learning. The emphasis is on building the pipeline, feature construction, PCA
by eigendecomposition, and principled model selection, and on
evaluating the result honestly rather than optimizing for a headline number.
