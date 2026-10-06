# Demand Forecasting & Late Delivery Risk (Supply Chain)

Forecasts daily category-level demand and flags orders likely to arrive late, using the DataCo Smart
Supply Chain dataset.

## Problem

Poor demand forecasts cause stockouts and overstock, and late deliveries damage customer trust and can
trigger contractual penalties. This project (1) forecasts daily demand per product category to support
inventory planning, and (2) predicts, at order placement, which orders are at risk of arriving late so they
can be expedited before they miss their promised window.

## Data

[DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
(Kaggle), about 180,000 order-line records from 2015 to 2017. The data file is not included in this repository
(see `.gitignore`).

To reproduce:
1. Download the dataset from Kaggle.
2. Place `DataCoSupplyChainDataset.csv` in the `data/` folder.

Note: monthly order counts drop sharply from October 2017, so the analysis uses complete months only (to 30 September 2017).

## Approach

1. **Cleaning:** standardised column names, removed personal and empty fields, and cut off the incomplete tail of the data.
2. **Exploratory analysis:** demand trend and seasonality, late-delivery rate by shipping mode, region, and category, and volume vs. value by category.
3. **Feature engineering:** aggregated order lines into a daily category-level demand series, then built day, month,
   lag (1/7/14/28-day), and rolling-average features. Rolling averages exclude the current day to prevent leakage.
4. **Demand forecasting:** rolling-average baseline vs. Linear Regression vs. Random Forest vs. a residual Random Forest,
   evaluated with MAE, RMSE, and R² on a date-based split. Random Forest hyperparameters were chosen on a validation period.
5. **Late delivery classification:** Logistic Regression vs. Random Forest, predicting late delivery at order placement
   from order-time information only. Post-delivery fields (actual shipping days, delivery status) are excluded to avoid leakage.

## Key Results

- **Demand:** the best model (residual Random Forest) reduces forecast MAE by about 2% vs. the naive rolling-average
  baseline (5.19 vs. 5.31). The 28-day rolling average carries most of the predictable signal.
- **Late delivery:** shipping mode and the scheduled delivery window are the strongest drivers. First Class orders are
  late 95% of the time, Standard Class 38%. Both classifiers reach ROC-AUC ≈ 0.74 and PR-AUC ≈ 0.80 on the test set.
- Threshold tuning did not change the test outcome (0 and 6 of 15,812 decisions changed), so the default threshold is kept.

## Business Recommendations

- Review First and Second Class delivery promises against actual performance before changing the carrier mix.
- Use the late-risk flag at order placement to prioritise expediting and proactive customer communication.
- Set safety stock by category demand volatility (for example Lacrosse, Golf Bags & Carts, Tennis & Racquet).

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook supply_chain_demand_forecasting_project.ipynb
```
Then run all cells from the top.

## Project Structure

```
supply_chain_demand_forecasting_project.ipynb   # full analysis
data/                                           # place the Kaggle CSV here (not committed)
requirements.txt
README.md
```

## Tools

Python, pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn

## About

Built as part of a transition into data science, drawing on 2.5 years of supply chain planning experience to ground
the feature engineering and business framing in operational knowledge.
