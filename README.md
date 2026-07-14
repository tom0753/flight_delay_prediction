# Flight delay prediction — neural networks on imbalanced data

A machine learning study on predicting US domestic flight delays, built as a university project for the Artificial Intelligence course at Bialystok University of Technology. The main finding is a negative one: **a model can reach 80% accuracy while being completely useless.** Most of this project is about why that happens and what to do about it.

## Problem

Given flight metadata known before departure (date, carrier, origin, destination, distance), predict whether a flight will arrive more than 15 minutes late.

- **Data:** [Airline Delay Analysis](https://www.kaggle.com/datasets/sherrytp/airline-delay-analysis) (Kaggle), file `2019.csv` — US domestic flights, 2019
- **Sample size:** 1,500,000 rows loaded; 1,454,959 after dropping rows with missing `ARR_DELAY`
- **Target:** `IS_DELAYED` = 1 if `ARR_DELAY` > 15 minutes, else 0
- **Class distribution:** roughly 80% on time / 20% delayed

Columns describing the *cause* of delay (`CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`) were dropped — they are only known after the fact and would leak the target.

## Method

**Feature importance.** A decision tree (`max_depth=12`, `class_weight='balanced'`) ranked the eight candidate features by Gini importance. The strongest were `DAY_OF_MONTH` (0.210), `ORIGIN` (0.184) and `DEST` (0.155); the weakest were `DISTANCE` (0.081) and `DAY_OF_WEEK`.

**Grid of experiments.** Four rounds of 36 experiments each (144 total) with `MLPClassifier`, varying:

| Parameter | Values |
|---|---|
| Test split | 0.1, 0.2, 0.3 |
| Hidden layers | (64,), (32, 16) |
| Activation | relu, tanh |
| Max epochs | 10, 50, 150 |

The four rounds cover the cross-product of two factors:
- **Feature set:** 8 features → 4 features (`DAY_OF_MONTH`, `ORIGIN`, `OP_UNIQUE_CARRIER`, `DEST`) → 2 features (`ORIGIN`, `DEST`)
- **Class balance:** raw distribution vs. undersampling the majority class to 50/50

Features were standardised with `StandardScaler`; early stopping was enabled.

## Results

| Configuration | Accuracy | Precision | Recall |
|---|---|---|---|
| 8 features, imbalanced | ~80% | ~55–65% (near-zero TP) | ~0.0002 |
| 8 features, undersampled | 57–59% | 56–59% | 52–64% |
| 4 features, imbalanced | ~80% | 0 | 0 |
| 4 features, undersampled | 53–56% | 52–55% | 46–62% |
| 2 features, imbalanced | ~80% | 0 | 0 |
| 2 features, undersampled | 51–53% | 51–53% | 45–71% |

## What this shows

**The 80% accuracy was an artifact of the class distribution.** On the imbalanced data, the networks learned to predict "on time" for essentially every flight. True positives were zero across most of the grid, so recall was zero — the model never once identified a delayed flight, yet scored 80% because 80% of flights are on time. No change to architecture, activation function or epoch limit made any difference; the confusion matrices were identical across all 36 runs. Early stopping consistently halted training after exactly 12 epochs, confirming the network had stopped learning anything.

**Undersampling forced the model to actually predict.** Balancing the classes 50/50 dropped headline accuracy from 80% to ~59% — while making the model meaningfully better. Recall rose from ~0 to 52–64%, and training ran for 83–87 epochs instead of 12. The lower number describes a model that works; the higher number described one that did not.

**Feature richness set the ceiling.** With all 8 features the balanced model reached ~59% accuracy and ~64% recall. Cutting down to 2 features (`ORIGIN`, `DEST`) collapsed it to ~52% — barely better than a coin flip, with a flood of false positives. Route identity alone does not predict delay.

**Honest conclusion:** pre-departure metadata alone is a weak signal for delay prediction. The realistic drivers — weather, inbound aircraft status, airport congestion at that hour — are not in this feature set. A ~59% balanced accuracy is the honest ceiling for what these inputs can support, not a failure of tuning.

## Tech stack

Python, pandas, NumPy, scikit-learn (`MLPClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `StandardScaler`, `LabelEncoder`), matplotlib, kagglehub. Developed in Google Colab.

## Running it

Open `ProjektSI.ipynb` in Google Colab or Jupyter and run all cells. The dataset downloads automatically via `kagglehub` — a Kaggle API token is required.

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib psutil
```

Note: the full grid trains 144 networks on up to 1.45M rows and takes a long time on CPU.

## Known limitations

- `ORIGIN` and `DEST` are label-encoded as integers, which imposes a false ordinal relationship on categorical airport codes. One-hot or target encoding would be more appropriate.
- A single `LabelEncoder` instance is reused across columns, so the encodings are not independently fitted per column.
- Undersampling discards ~80% of the majority class. Class weights or SMOTE would preserve more data.

## Authors

University project, Faculty of Computer Science, Bialystok University of Technology.

- Tomasz Chodorowski — [@tom0753](https://github.com/tom0753)
- Miłosz Michalski
- Patryk Romanowicz
- Michał Chomontowski
