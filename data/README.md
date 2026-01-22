# Data

This project uses the **M4 Forecasting Dataset**, a benchmark dataset widely used in time series forecasting research.

## Source

- Competition: M4 Forecasting Competition
- Official site: https://www.m4.unic.ac.cy/

## Structure

The dataset contains time series grouped by frequency:

| Frequency | Series count |
|----------|--------------|
| Yearly   | 23,000+      |
| Quarterly| 24,000+      |
| Monthly  | 48,000+      |
| Weekly   | 359          |
| Daily    | 4227         |
| Hourly   | 414          |

Each series:
- Contains **positive values**
- Has a **limited historical length**
- Belongs to exactly one frequency group

## Directory Layout

We will focus on the daily frequency dataset
```text
data/
├── raw/
│   ├── M4-info.csv
│   ├── Daily-train.csv
│   └── Daily-test.csv
├── processed/
│   └── (generated datasets)