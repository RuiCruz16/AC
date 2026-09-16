# WNBA Machine Learning Project

Machine learning project that predicts WNBA outcomes using historical league data (seasons 1–10, 1997–2006) and applies the trained models to forecast **Season 11** (2007). The work is divided into three predictive problems, each tackled with its own feature engineering, feature selection and modelling pipeline.

## Project Overview

The end-of-season results of a basketball season are known only after the season ends. The goal of this project is to anticipate those results *before* the tape, using only information available from previous seasons:

1. **Team Ranking Problem** — predict each team's final conference ranking.
2. **Coach Change Problem** — predict whether a team's head coach will be replaced after the season.
3. **Player Awards Problem** — predict the winners of the league's individual end-of-season awards.

All models are trained on seasons 1–9/10 and evaluated on future, unseen seasons (hold-out seasons 9 and 10, plus a fully fresh **Season 11** used for real predictions).

## Repository Structure

```
├── dataset/                  Raw WNBA datasets (seasons 1-10)
├── dataset_cleaned/          Cleaned versions of the raw datasets
├── dataset_for_modeling/     Engineered feature sets, one per problem/award
├── Season_11/                New season data used for forecasting
├── doc/
│   └── presentation.pdf      Project presentation/slides
├── jupyter-notebooks/
│   ├── data_cleaning.ipynb
│   ├── data_analysis.ipynb
│   ├── feature_engineering_and_selection.ipynb
│   └── data_modeling.ipynb
└── README.md
```

## Data

The raw data is stored in `dataset/` and describes WNBA seasons 1 through 10:

| File | Description |
|------|-------------|
| `teams.csv` | Team-level regular-season statistics |
| `teams_post.csv` | Team-level playoff statistics |
| `players.csv` | Player biographical information |
| `players_teams.csv` | Player-season statistics |
| `coaches.csv` | Coach-season statistics |
| `awards_players.csv` | Individual awards won per season |
| `series_post.csv` | Playoff series results |

The cleaned versions live in `dataset_cleaned/`, and the engineerized feature matrices used by the models live in `dataset_for_modeling/`. Season 11 input data used only for forecasting is in `Season_11/`.

## Workflow

The project follows a standard ML pipeline, implemented in the four notebooks:

1. **Data Cleaning** (`data_cleaning.ipynb`)
   Removes redundant columns (e.g. `lgID`), columns with only `NaN`/zero values, and rows for players/coaches that never played or coached. Output is written to `dataset_cleaned/`.

2. **Data Analysis** (`data_analysis.ipynb`)
   Exploratory data analysis of every dataset: descriptive statistics, missing-value analysis, time-series evolution of wins/points, correlation heatmaps and pairplots, boxplots for outliers, and comparative playoff vs. regular-season analysis.

3. **Feature Engineering & Selection** (`feature_engineering_and_selection.ipynb`)
   - **Engineering:** builds per-problem features such as per-game stats, lag-1 features (last year's stats), 3-year average trends, efficiency/offensive-net ratings, z-score standardization per season, and award-specific features (e.g. rookie identification, coach tenure, playoff impact metrics).
   - **Selection:** uses correlation analysis, Variance Inflation Factor (VIF) to remove multicollinearity, and RFE/RFECV (Recursive Feature Elimination with cross-validation) to keep the most predictive features per problem.

4. **Data Modeling** (`data_modeling.ipynb`)
   Trains, tunes and evaluates models per problem, using temporally structured splits, and produces final predictions for Season 11.

## The Three Problems

### 1. Team Ranking Problem

Predict each team's final **conference ranking** (EA / WE) for a season using only information from previous seasons.

- **Task:** regression on the `rank` target.
- **Features:** current-year stats, lag-1 features (previous season), and 3-year average trends (e.g. `net_rating`, `win_pct`, four-factor metrics).
- **Models:** Linear Regression, Ridge Regression, SVR, Random Forest, Gradient Boosting, AdaBoost.
- **Evaluation:** MAE, RMSE and Spearman rank correlation, against a baseline that predicts the previous year's rank. Additional business-like metrics: NDCG and playoff qualification precision (predicting whether a team finishes in the top 4).

Results on the Season 10 test set: best model **Gradient Boosting** reached **Test MAE ≈ 1.57** (baseline 2.46), NDCG **0.942** and average playoff precision **62.5%**. The most important features were defensive/offensive efficiency and win-percentage trends.

### 2. Coach Change Problem

Predict whether a team will **change its head coach** at the end of the season.

- **Task:** binary classification (`coach_changed` = 1/0).
- **Features:** coach performance (win percentage, playoff wins), tenure, team results compared to the previous season, and historical trends.
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, SVC, AdaBoost.
- **Evaluation:** F1-score and ROC AUC over rolling-window validation splits.

The model ranked the Season 11 coaches most likely to be fired, e.g. MIN, CHI, SAC, SAS and NYL.

### 3. Player Awards Problem

Predict the winner of each end-of-season individual award:

| Award | Note |
|-------|------|
| Most Valuable Player (MVP) | |
| Most Improved Player (MIP) | |
| Rookie of the Year (ROY) | restricted to rookies only |
| Sixth Woman of the Year | |
| Defensive Player of the Year (DPOY) | |
| Coach of the Year (COY) | |
| WNBA Finals MVP | playoff stats |
| All-Star Game MVP | |
| Kim Perrot Sportsmanship Award | |

- **Task:** classification / ranking of the most likely candidates per award (only players that could realistically win are kept in the candidate pool).
- **Features:** per-award engineered features (per-game stats, defensive "stocks", assist-to-turnover, fouls, playoff impact, coach/team context) standardized per season with z-scores.
- **Models:** Random Forest (with/without scaling), XGBoost, Logistic Regression — tuned with Grid Search and rolling expanding-window splits.
- **Evaluation:** ROC AUC, average precision and Top-K hit rate (is the actual winner among the top-K predicted candidates?).

## Methodology & Evaluation

To avoid **data leakage**, splits are always temporal: training on past seasons and testing on future seasons (e.g. train seasons 1–8 → validate on 9 → test on 10), or expanding rolling windows where each year is validated against models trained only on earlier years. Forecasts for Season 11 are made by retraining the best models on all available data (seasons 1–10).

## Requirements

The notebooks were developed in Python 3 with:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `xgboost`
- `jupyter`

## Usage

Work through the notebooks in order inside a Jupyter environment:

```bash
jupyter notebook jupyter-notebooks/
```

1. `data_cleaning.ipynb`
2. `data_analysis.ipynb`
3. `feature_engineering_and_selection.ipynb`
4. `data_modeling.ipynb`

## Results Summary

| Problem | Best model(s) | Key metric |
|---------|---------------|------------|
| Ranking (Year 10 test) | Gradient Boosting | Test MAE 1.57 (baseline 2.46), NDCG 0.942 |
| Coach change | Logistic Regression / Gradient Boosting | F1 up to ~0.73, AUC up to ~0.79 (validation) |
| Player awards (each) | Random Forest / XGBoost / Logistic Regression | ROC AUC up to 1.00 on validation (Small positive class; n≈7-10 winners) |

## Authors

Machine Learning academic project — Data source: historical WNBA seasons (1997–2007).
