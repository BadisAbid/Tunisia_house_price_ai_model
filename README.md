# Tunisia House Price Prediction

Machine-learning model and Streamlit app that estimate residential property prices in Tunisia (TND) from location, size, condition, and amenities.

> **Status:** `[TODO: confirm project status, e.g. prototype / maintained]`

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Known Issues](#known-issues)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Overview

Property listings in Tunisia vary widely by governorate, city, age, and condition, and there is no simple reference for what a given property should cost. This project trains a regression model on Tunisian listing data and serves it through a Streamlit web interface. Given a property's location, area, room counts, age, condition, and amenities, the app returns an estimated price in Tunisian dinar (TND) and an approximate euro equivalent.

- **Input:** governorate, city/location, area (m²), pieces, rooms, bathrooms, age bracket, condition, and 11 binary amenities.
- **Output:** estimated price in TND and EUR.
- **Target:** the model predicts a log-transformed price (`log1p`); the app converts it back with `expm1`.

## Key Features

- Regression model trained in a Jupyter notebook (`Notebook.ipynb`) on a cleaned listings dataset.
- Feature engineering: composite amenity scores, area ratios, distance to the capital, and target-based location encodings.
- Streamlit interface with dependent governorate → city selectors and validated numeric inputs.
- Serialized preprocessing artifacts (`.pkl`) so inference reuses the exact encodings from training.
- Fallback to mean encodings for locations not seen in training.
- The app supports both scikit-learn-style models and neural networks (via a `use_nn` flag stored in `model_info.pkl`, which also switches on feature scaling).

## Dataset

| Item | Details |
|---|---|
| Raw data | `dataSetFull.csv` |
| Cleaned data | `dataset_clean.csv` |
| Source | `[TODO: confirm data source, e.g. scraped listings site, date collected]` |
| Rows / columns | `[TODO: confirm row and column counts]` |
| Target | Property price in TND, modeled as `log1p(price)` |

**Input features used by the app** (derived from `app.py`):

| Feature | Type | Allowed input range / values |
|---|---|---|
| Governorate, City/Location | Categorical | 16 governorates, 49 locations hardcoded in the app |
| `Area` | Numeric | 50–5000 m² |
| `pieces` | Numeric | 1–30 |
| `room` | Numeric | 1–25 |
| `bathroom` | Numeric | 1–15 |
| Age | Ordinal | `0`, `1-5`, `5-10`, `10-20`, `20-30`, `30-50`, `50-70`, `70-100`, `Plus de 100` |
| `state` | Ordinal | 1 = new/excellent, 2 = good, 3 = needs renovation |
| Amenities (binary) | Boolean | garage, garden, pool, elevator, beach view, mountain view, furnished, equipped kitchen, central heating, air conditioning, concierge |

**Engineered features** (computed in `prepare_features` in `app.py`):

- `total_rooms`, `area_per_room`, `log_area`
- `luxury_score`, `comfort_score`, `outdoor_score`, `privacy_score` (sums of amenity flags)
- `distance_to_capital` and `proximity_capital` (`1 / (distance + 1)`), from a hardcoded per-governorate distance table
- `age_encoded`, `governorate_rank` (loaded from `age_order.pkl`, `governorate_rank.pkl`)
- `city_encoded`, `location_encoded`, `city_price_per_sqm`, `location_price_per_sqm` (loaded from `city_means.pkl`, `location_means.pkl`, `city_price_per_sqm.pkl`, `loc_price_per_sqm.pkl`)

The authoritative feature list and order are stored in `model_info.pkl` (`features` key). `[TODO: list final feature set from model_info.pkl]`

**Preprocessing:** `[TODO: document cleaning steps from Notebook.ipynb (missing values, outlier handling, deduplication)]`

## Methodology

1. **Cleaning and exploration** in `Notebook.ipynb`, producing `dataset_clean.csv`. `[TODO: confirm steps]`
2. **Feature engineering** as listed above. Location encodings are stored as dictionaries and applied at inference time.
3. **Target transform:** price is modeled on the log scale (`log1p`) and inverted at prediction time (`expm1`).
4. **Model selection:** `requirements.txt` includes scikit-learn, XGBoost, and TensorFlow, and the app can load either a tree/linear-style model or a neural network. `[TODO: confirm candidate models compared, validation scheme, hyperparameter tuning, and which model was selected]`
5. **Serialization:** the selected model and its metadata (`model_name`, `best_r2`, `features`, `use_nn`) are saved with `joblib`. When `use_nn` is true, inputs are standardized with `scaler.pkl` before prediction.

At inference, unseen cities fall back to the mean of the stored encodings, and unseen governorates fall back to a default distance of 100 km and the median governorate rank.

## Evaluation

The app reads `model_name` and `best_r2` from `model_info.pkl` and shows them in the UI header. The values are not stated anywhere I could read, so none are reproduced here.

| Metric | Value |
|---|---|
| Selected model | `[TODO: from model_info.pkl]` |
| R² | `[TODO: from model_info.pkl / notebook]` |
| RMSE / MAE (TND) | `[TODO: confirm]` |
| Train/test split or CV scheme | `[TODO: confirm]` |

`[TODO: confirm the location encodings (city/location means, price per m²) were fitted on the training split only, to rule out target leakage in the reported R².]`

## Installation

**Prerequisites**

- Python 3.8–3.10. This range is inferred from the pinned `tensorflow==2.10.0` and `numpy==1.24.4`; `[TODO: confirm tested Python version]`.
- `git`

```bash
git clone https://github.com/BadisAbid/Tunisia_house_price_ai_model.git
cd Tunisia_house_price_ai_model

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install streamlit            # imported by app.py, not yet listed in requirements.txt
pip install jupyter              # only needed to open Notebook.ipynb
```

**Pinned dependencies** (`requirements.txt`):

| Package | Version |
|---|---|
| pandas | 2.0.3 |
| numpy | 1.24.4 |
| scikit-learn | 1.3.2 |
| xgboost | 2.0.3 |
| tensorflow | 2.10.0 |
| flask | 3.0.0 |
| joblib | 1.3.2 |
| matplotlib | 3.7.5 |
| seaborn | 0.13.2 |

## Usage

### Train

Training and export are done interactively in the notebook:

```bash
jupyter notebook Notebook.ipynb
```

Running the notebook is expected to produce the `.pkl` artifacts listed in [Project Structure](#project-structure). `[TODO: confirm the notebook's output paths, including whether it writes to models/]`

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`), choose a governorate and city, fill in the property details, and click **Predict Price**.

### Programmatic prediction

The inference logic lives in `prepare_features` in `app.py`. The equivalent standalone flow is:

```python
import joblib
import numpy as np

model      = joblib.load("models/best_model.pkl")   # see Known Issues
model_info = joblib.load("models/model_info.pkl")
scaler     = joblib.load("scaler.pkl")

# features_df: one-row DataFrame with columns in model_info["features"],
# built as in prepare_features() in app.py
if model_info["use_nn"]:
    pred_log = model.predict(scaler.transform(features_df), verbose=0)[0][0]
else:
    pred_log = model.predict(features_df)[0]

price_tnd = float(np.expm1(pred_log))
```

## Project Structure

```text
Tunisia_house_price_ai_model/
├── Notebook.ipynb            # EDA, feature engineering, training
├── app.py                    # Streamlit app (earlier Flask version kept as comments)
├── requirements.txt
├── dataSetFull.csv           # raw dataset
├── dataset_clean.csv         # cleaned dataset
├── logo_2.png                # app header image
├── model_info.pkl            # model name, R², feature list, use_nn flag
├── scaler.pkl                # input scaler (used when use_nn is true)
├── age_order.pkl             # age bracket → ordinal encoding
├── governorate_rank.pkl      # governorate → rank
├── city_means.pkl            # city → mean-price encoding
├── location_means.pkl        # location → mean-price encoding
├── city_price_per_sqm.pkl    # city → price per m²
├── loc_price_per_sqm.pkl     # location → price per m²
└── README.md
```

`app.py` also expects `models/best_model.pkl` and `models/model_info.pkl`, which are not in the repository. See [Known Issues](#known-issues).

## Technologies

- **Language:** Python
- **Data and ML:** pandas, NumPy, scikit-learn, XGBoost, TensorFlow
- **App:** Streamlit
- **Persistence:** joblib
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook

## Known Issues

- `streamlit` is imported by `app.py` but missing from `requirements.txt`.
- `app.py` loads `models/best_model.pkl` and `models/model_info.pkl`. The repository has no `models/` directory or `best_model.pkl`, and `model_info.pkl` is at the root. The app will not start from a fresh clone until the paths or files are aligned.
- `flask` is listed in `requirements.txt`, but the Flask version of the app is commented out in `app.py`.
- `requirements.txt` carries a commented-out set of minimum versions alongside the pinned set; only the pinned set is active.

## Limitations and Future Work

**Current limitations**

- Coverage is limited to the 16 governorates and 49 locations hardcoded in `app.py`; other locations fall back to average encodings.
- `distance_to_capital` is a hardcoded per-governorate approximation, not a per-property distance.
- The EUR figure uses a fixed rate (`price_tnd / 3.23`) and is not updated.
- Prices are point estimates with no confidence interval.
- Location encodings are static snapshots from the training data and do not reflect market changes.

**Future work**

- Report cross-validated metrics and prediction intervals.
- Move training from the notebook into a reproducible script and add a `models/` export step.
- Replace hardcoded location and distance tables with data-derived ones.
- Fetch the EUR/TND exchange rate instead of using a constant.
- Add a `requirements.txt` that matches the app's actual imports and a basic test for `prepare_features`.

## License

No license file was found in the repository. `[TODO: add a LICENSE file (e.g. MIT) and update this section]`

## Contributing

Issues and pull requests are welcome.

1. Fork the repository and create a feature branch.
2. Make your changes and verify the app runs with `streamlit run app.py`.
3. Open a pull request describing the change.

## Contact

Maintainer: [BadisAbid](https://github.com/BadisAbid) · `[TODO: add email or LinkedIn if desired]`
