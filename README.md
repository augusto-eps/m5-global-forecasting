# Global Time Series Forecasting with First Differences (M5)

This project explores whether global forecasting models trained on first-differenced demand across related products
can outperform traditional per-series models when historical data is limited.

## Motivation
In many real-world settings, individual time series are short (1–2 years), making seasonality estimation unreliable.
However, related series (e.g., same category or retailer) often share behavioral dynamics.

This project tests whether those shared dynamics can be learned and exploited.

## Dataset
- M5 Forecasting Dataset (Walmart daily sales)

## Key Ideas
- Global vs local (per-series) models
- First-difference transformation to remove level effects
- Category-level shared signals
- Short-history (<2 cycle) forecasting

## Status
Work in progress
