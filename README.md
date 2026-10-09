
![Tunisia House Price AI Model](logo_2.png)
 
# 🏠 Tunisia House Price Prediction
 
**A machine learning model and web app that estimates residential property prices across Tunisia.**
 
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-337AB7)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
 
[Live Demo](https://badisabid.github.io/Tunisia_house_price_ai_model/) · [Report a Bug](https://github.com/BadisAbid/Tunisia_house_price_ai_model/issues) · [Request a Feature](https://github.com/BadisAbid/Tunisia_house_price_ai_model/issues)
 
---
 
## 📖 Table of Contents
 
- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
---
 
## 📌 About the Project
 
Property pricing in Tunisia is often opaque and varies widely between governorates, cities, and neighbourhoods. This project uses supervised machine learning to predict the market price of a house or apartment from its location, size, age, condition, and amenities.
 
The pipeline covers the full workflow: data cleaning, exploratory analysis, feature engineering, model training and comparison, and deployment through an interactive **Streamlit** web application where users enter property details and receive an instant estimate in both **Tunisian Dinar (TND)** and **Euro (EUR)**.
 
## ✨ Features
 
- **Instant price estimates** in TND and EUR from a simple form
- **Wide geographic coverage**: 16 governorates and 50+ cities and neighbourhoods, including Tunis, Ariana, Ben Arous, Nabeul, Sousse, Sfax, Djerba, and more
- **Rich property inputs**: area, number of pieces, rooms, bathrooms, age bracket, and condition
- **11 amenity flags**: garage, garden, pool, elevator, beach view, mountain view, furnished, equipped kitchen, central heating, air conditioning, concierge
- **Advanced feature engineering**: target-encoded locations, price-per-m² signals, composite luxury/comfort scores, and distance to the capital
- **Cascading location selectors**: city options update automatically with the chosen governorate
- **Reproducible research**: the full analysis and training process is documented in a Jupyter notebook
## 🛠 Tech Stack
 
| Category | Tools |
| --- | --- |
| Language | Python |
| Data processing | pandas, NumPy |
| Modeling | scikit-learn, XGBoost, TensorFlow / Keras |
| Visualization | Matplotlib, Seaborn |
| Serialization | joblib |
| Web app | Streamlit |
 
## 📁 Project Structure
 
```
Tunisia_house_price_ai_model/
├── app.py                    # Streamlit web application
├── Notebook.ipynb            # Data exploration, feature engineering & model training
├── dataSetFull.csv           # Raw dataset
├── dataset_clean.csv         # Cleaned dataset used for training
├── logo_2.png                # Application logo
├── requirements.txt          # Python dependencies
├── scaler.pkl                # Feature scaler (used by the neural network)
├── model_info.pkl            # Model metadata (name, features, R², model type)
├── city_means.pkl            # City-level target encoding
├── location_means.pkl        # Location-level target encoding
├── city_price_per_sqm.pkl    # Average price per m² by city
├── loc_price_per_sqm.pkl     # Average price per m² by location
├── governorate_rank.pkl      # Governorate price ranking
├── age_order.pkl             # Ordinal encoding of property age brackets
└── models/
    └── best_model.pkl        # Trained best-performing model
```
 
## 🔬 Methodology
 
### 1. Data
Property listings from across Tunisia (`dataSetFull.csv`) were cleaned and standardized into `dataset_clean.csv`.
 
### 2. Feature Engineering
 
| Feature group | Description |
| --- | --- |
| **Base features** | Area, pieces, rooms, bathrooms, state (condition), encoded age |
| **Amenity flags** | 11 binary indicators (garage, pool, elevator, etc.) |
| **Aggregates** | `total_rooms`, `area_per_room`, `log_area` |
| **Composite scores** | `luxury_score`, `comfort_score`, `outdoor_score`, `privacy_score` |
| **Geographic** | `distance_to_capital`, `proximity_capital`, `governorate_rank` |
| **Location encodings** | `city_encoded`, `location_encoded`, `city_price_per_sqm`, `location_price_per_sqm` |
 
Unseen cities fall back to the global mean of the corresponding encoding.
 
### 3. Modeling
Several algorithms (tree-based models, XGBoost, and a neural network) were trained and compared. The target is **log-transformed** (`log1p`) to reduce skew and converted back with `expm1` at prediction time. The best model is saved with its metadata and loaded by the app. When the selected model is a neural network, inputs are scaled with the saved `scaler.pkl`.
 
## 🚀 Getting Started
 
### Prerequisites
 
- Python 3.9 or higher
- `pip`
### Installation
 
```bash
# 1. Clone the repository
git clone https://github.com/BadisAbid/Tunisia_house_price_ai_model.git
cd Tunisia_house_price_ai_model
 
# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
 
# 3. Install dependencies
pip install -r requirements.txt
pip install streamlit
```
 
> **Note:** the app expects the trained model at `models/best_model.pkl` and metadata at `models/model_info.pkl`. If these files are missing, run `Notebook.ipynb` end to end to regenerate them.
 
### Run the app
 
```bash
streamlit run app.py
```
 
Then open **http://localhost:8501** in your browser.
 
## 💡 Usage
 
1. Select a **governorate** and **city / location**.
2. Enter the property's **area**, **number of pieces**, **rooms**, and **bathrooms**.
3. Choose the **age** bracket and **condition** (New, Good, or Needs Renovation).
4. Select any available **amenities**.
5. Click **Predict Price** to see the estimate in **TND** and **EUR**.
### Programmatic example
 
```python
from app import prepare_features  # or copy the function into your own script
 
sample = {
    "governorate": "Ariana",
    "city": "La Soukra",
    "area": 220,
    "pieces": 6,
    "rooms": 4,
    "bathrooms": 2,
    "age": "5-10",
    "state": 2,
    "amenities": ["garage", "garden", "air_conditioning"],
}
 
X = prepare_features(sample)
# prediction = np.expm1(model.predict(X)[0])
```
 
## 📊 Model Performance
 
| Metric | Value |
| --- | --- |
| Best model | _add model name (e.g. XGBoost / Random Forest / Neural Network)_ |
| R² score | _add value_ |
| MAE | _add value_ |
| RMSE | _add value_ |
 
> The in-app R² badge is read from `model_info.pkl`. Update this table with the results from your notebook.
 
## ⚠️ Limitations
 
- Predictions are **estimates** based on historical listing data, not official valuations.
- Asking prices in listings may differ from final transaction prices.
- Accuracy is lower for locations that are rare or absent in the training data.
- The EUR conversion uses a fixed rate (1 EUR ≈ 3.23 TND) and does not reflect live exchange rates.
- Pinned dependency versions (e.g. `tensorflow==2.10.0`) may require an older Python version.
## 🗺 Roadmap
 
- [ ] Add model evaluation plots and SHAP feature-importance explanations
- [ ] Support more cities and rural areas
- [ ] Fetch live TND/EUR exchange rates
- [ ] Add prediction confidence intervals
- [ ] Containerize with Docker
- [ ] Deploy to Streamlit Community Cloud
## 🤝 Contributing
 
Contributions, issues, and feature requests are welcome.
 
1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
## 📄 License
 
No license has been specified yet. Consider adding one (for example [MIT](https://choosealicense.com/licenses/mit/)) by creating a `LICENSE` file.
 
## 👤 Author
 
**Badis Abid**
 
- GitHub: [@BadisAbid](https://github.com/BadisAbid)
---
 
If you found this project useful, please consider giving it a ⭐
