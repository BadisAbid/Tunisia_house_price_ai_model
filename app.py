# """
# House Price Prediction Web Application - Tunisia
# Updated for Advanced Feature Engineering
# """

# from flask import Flask, render_template_string, request, jsonify
# import numpy as np
# import pandas as pd
# import joblib
# import os

# app = Flask(__name__)

# # Load models and new encodings
# print("Loading model and encoders...")
# model = joblib.load('models/best_model.pkl')
# model_info = joblib.load('models/model_info.pkl')
# scaler = joblib.load('scaler.pkl')

# # Load the new dictionary-based encodings
# city_means = joblib.load('city_means.pkl')
# location_means = joblib.load('location_means.pkl')
# loc_price_per_sqm = joblib.load('loc_price_per_sqm.pkl')
# city_price_per_sqm = joblib.load('city_price_per_sqm.pkl')
# governorate_rank = joblib.load('governorate_rank.pkl')
# age_order = joblib.load('age_order.pkl')

# FEATURES = model_info['features']
# USE_NN = model_info['use_nn']

# print(f"Loaded model: {model_info['model_name']}")
# print(f"Model R²: {model_info['best_r2']:.4f}")

# # Tunisia governorates and cities mapping
# GOVERNORATES = {
#     'tunis': ['Tunis', 'El Menzah', 'La Marsa', 'Carthage', 'Le Bardo', 'Le Kram', 
#               'El Ouardia', 'Ettahrir', 'El Omrane', 'El Omrane Superieur', 'Cité El Khadra'],
#     'Ariana': ['Ariana Ville', 'La Soukra', 'Raoued', 'Mnihla', 'Sidi Thabet', 'Chotrana'],
#     'Ben Arous': ['Boumhel Bassatine', 'El Mourouj', 'Ezzahra', 'Rades', 'Mornag', 
#                   'Hammam Lif', 'Mégrine', 'Hammam Chatt', 'Mohamadia'],
#     'Nabeul': ['Hammamet', 'Nabeul', 'Yasmine Hammamet', 'Beni Khiar', 'Kélibia'],
#     ' Sousse': ['Sousse Ville', 'Hammam Sousse', 'Sousse Riadh', 'Sahloul', 'Akouda'],
#     'Sfax': ['Sfax Ville', 'Sfax Sud'],
#     'Mahdia': ['Mahdia Ville'],
#     'Djerba': ['Djerba', 'Houmt Souk', 'Midoun'],
#     'La Manouba': ['La Manouba', 'Mornaguia', 'Denden', 'Oued Ellil'],
#     'Jendouba': ['Tabarka'],
#     'Bizerte': ['Bizerte'],
#     'Gabès': ['Gabès Sud'],
#     'Gafsa': ['Gafsa Sud'],
#     'Béja': ['Medjez El Bab'],
#     'Monastir': ['Monastir Ville'],
#     'Medenine': ['Zarzis']
# }

# AGE_OPTIONS = ['0', '1-5', '5-10', '10-20', '20-30', '30-50', '50-70', '70-100', 'Plus de 100']

# HTML_TEMPLATE = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>House Price Prediction - Tunisia</title>
#     <style>
#         * { margin: 0; padding: 0; box-sizing: border-box; }
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #1a5f7a 0%, #159895 100%);
#             min-height: 100vh; padding: 20px;
#         }
#         .container {
#             max-width: 900px; margin: 0 auto; background: white;
#             border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
#         }
#         .header {
#             background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
#             color: white; padding: 30px; text-align: center;
#         }
#         .header h1 { font-size: 2.5em; margin-bottom: 10px; }
#         .header p { font-size: 1.1em; opacity: 0.9; }
#         .model-info {
#             background: #f8f9fa; padding: 15px 30px;
#             display: flex; justify-content: space-around; border-bottom: 2px solid #eee;
#         }
#         .model-info div { text-align: center; }
#         .model-info span { display: block; font-size: 0.9em; color: #666; }
#         .model-info strong { font-size: 1.2em; color: #1a5f7a; }
#         .form-container { padding: 30px; }
#         .form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
#         .form-group { display: flex; flex-direction: column; }
#         .form-group.full-width { grid-column: span 2; }
#         label { font-weight: 600; color: #333; margin-bottom: 8px; font-size: 0.95em; }
#         input, select {
#             padding: 12px 15px; border: 2px solid #ddd; border-radius: 10px;
#             font-size: 1em; transition: all 0.3s;
#         }
#         input:focus, select:focus {
#             outline: none; border-color: #1a5f7a; box-shadow: 0 0 0 3px rgba(26, 95, 122, 0.2);
#         }
#         .checkbox-group {
#             display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
#             padding: 15px; background: #f8f9fa; border-radius: 10px;
#         }
#         .checkbox-item { display: flex; align-items: center; gap: 8px; }
#         .checkbox-item input[type="checkbox"] { width: 20px; height: 20px; cursor: pointer; }
#         .checkbox-item label { font-weight: normal; margin: 0; cursor: pointer; }
#         .predict-btn {
#             grid-column: span 2; padding: 18px;
#             background: linear-gradient(135deg, #1a5f7a 0%, #159895 100%);
#             color: white; border: none; border-radius: 12px; font-size: 1.2em;
#             font-weight: 600; cursor: pointer; transition: all 0.3s; margin-top: 10px;
#         }
#         .predict-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(26, 95, 122, 0.4); }
#         .result-container {
#             display: none; padding: 30px; text-align: center;
#             background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
#             color: white; animation: slideIn 0.5s ease;
#         }
#         @keyframes slideIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
#         .result-price { font-size: 3em; font-weight: bold; margin: 15px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
#         .result-price-eur { font-size: 1.5em; opacity: 0.9; }
#         .loading { display: none; text-align: center; padding: 30px; }
#         .spinner {
#             border: 4px solid #f3f3f3; border-top: 4px solid #1a5f7a;
#             border-radius: 50%; width: 50px; height: 50px;
#             animation: spin 1s linear infinite; margin: 0 auto 15px;
#         }
#         @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#         .footer { padding: 20px; text-align: center; color: #666; font-size: 0.9em; }
#         .tunisia-flag { font-size: 2em; }
#         @media (max-width: 768px) {
#             .form-grid { grid-template-columns: 1fr; }
#             .form-group.full-width, .predict-btn { grid-column: span 1; }
#             .checkbox-group { grid-template-columns: repeat(2, 1fr); }
#             .header h1 { font-size: 1.8em; }
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <div class="tunisia-flag">🇹🇳</div>
#             <h1>House Price Prediction</h1>
#             <p>Estimate property values across Tunisia</p>
#         </div>
#         <div class="model-info">
#             <div><span>Model</span><strong>{{ model_name }}</strong></div>
#             <div><span>Accuracy (R²)</span><strong>{{ r2_score }}%</strong></div>
#             <div><span>Status</span><strong style="color: #27ae60;">✓ Ready</strong></div>
#         </div>
#         <form class="form-container" id="predictionForm">
#             <div class="form-grid">
#                 <div class="form-group">
#                     <label for="governorate">Governorate *</label>
#                     <select id="governorate" name="governorate" required onchange="updateCities()">
#                         <option value="">Select Governorate</option>
#                         {% for gov in governorates %}<option value="{{ gov }}">{{ gov }}</option>{% endfor %}
#                     </select>
#                 </div>
#                 <div class="form-group">
#                     <label for="city">City/Location *</label>
#                     <select id="city" name="city" required><option value="">Select Governorate first</option></select>
#                 </div>
#                 <div class="form-group">
#                     <label for="area">Area (m²) *</label>
#                     <input type="number" id="area" name="area" min="50" max="5000" placeholder="e.g., 200" required>
#                 </div>
#                 <div class="form-group">
#                     <label for="pieces">Number of Pieces *</label>
#                     <input type="number" id="pieces" name="pieces" min="1" max="30" placeholder="e.g., 5" required>
#                 </div>
#                 <div class="form-group">
#                     <label for="rooms">Number of Rooms *</label>
#                     <input type="number" id="rooms" name="rooms" min="1" max="25" placeholder="e.g., 4" required>
#                 </div>
#                 <div class="form-group">
#                     <label for="bathrooms">Number of Bathrooms *</label>
#                     <input type="number" id="bathrooms" name="bathrooms" min="1" max="15" placeholder="e.g., 2" required>
#                 </div>
#                 <div class="form-group">
#                     <label for="age">Property Age *</label>
#                     <select id="age" name="age" required>
#                         <option value="">Select Age</option>
#                         {% for age in age_options %}<option value="{{ age }}">{{ age }} years</option>{% endfor %}
#                     </select>
#                 </div>
#                 <div class="form-group">
#                     <label for="state">State (1=New, 2=Good, 3=Needs Work)</label>
#                     <select id="state" name="state">
#                         <option value="1">New / Excellent</option>
#                         <option value="2" selected>Good Condition</option>
#                         <option value="3">Needs Renovation</option>
#                     </select>
#                 </div>
#                 <div class="form-group full-width">
#                     <label>Features & Amenities</label>
#                     <div class="checkbox-group">
#                         <div class="checkbox-item"><input type="checkbox" id="garage" name="garage" value="1"><label for="garage">Garage</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="garden" name="garden" value="1"><label for="garden">Garden</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="pool" name="pool" value="1"><label for="pool">Pool</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="elevator" name="elevator" value="1"><label for="elevator">Elevator</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="beach_view" name="beach_view" value="1"><label for="beach_view">Beach View</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="mountain_view" name="mountain_view" value="1"><label for="mountain_view">Mountain View</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="furnished" name="furnished" value="1"><label for="furnished">Furnished</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="equipped_kitchen" name="equipped_kitchen" value="1"><label for="equipped_kitchen">Equipped Kitchen</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="central_heating" name="central_heating" value="1"><label for="central_heating">Central Heating</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="air_conditioning" name="air_conditioning" value="1"><label for="air_conditioning">Air Conditioning</label></div>
#                         <div class="checkbox-item"><input type="checkbox" id="concierge" name="concierge" value="1"><label for="concierge">Concierge</label></div>
#                     </div>
#                 </div>
#                 <button type="submit" class="predict-btn">🏠 Predict Price</button>
#             </div>
#         </form>
#         <div class="loading" id="loading"><div class="spinner"></div><p>Analyzing property details...</p></div>
#         <div class="result-container" id="result">
#             <h2>Estimated Price</h2>
#             <div class="result-price" id="priceTND"></div>
#             <div class="result-price-eur" id="priceEUR"></div>
#             <p style="margin-top: 15px; opacity: 0.8;">* This is an estimate based on market data</p>
#         </div>
#         <div class="footer"><p>House Price Prediction Model - Tunisia 🇹🇳</p></div>
#     </div>
#     <script>
#         const governoratesCities = {{ governorates_json | safe }};
#         function updateCities() {
#             const gov = document.getElementById('governorate').value;
#             const citySelect = document.getElementById('city');
#             citySelect.innerHTML = '<option value="">Select City</option>';
#             if (gov && governoratesCities[gov]) {
#                 governoratesCities[gov].forEach(city => {
#                     const option = document.createElement('option');
#                     option.value = city; option.textContent = city;
#                     citySelect.appendChild(option);
#                 });
#             }
#         }
#         document.getElementById('predictionForm').addEventListener('submit', async function(e) {
#             e.preventDefault();
#             document.getElementById('loading').style.display = 'block';
#             document.getElementById('result').style.display = 'none';
#             const formData = {
#                 governorate: document.getElementById('governorate').value,
#                 city: document.getElementById('city').value,
#                 area: parseFloat(document.getElementById('area').value),
#                 pieces: parseInt(document.getElementById('pieces').value),
#                 rooms: parseInt(document.getElementById('rooms').value),
#                 bathrooms: parseInt(document.getElementById('bathrooms').value),
#                 age: document.getElementById('age').value,
#                 state: parseInt(document.getElementById('state').value),
#                 garage: document.getElementById('garage').checked ? 1 : 0,
#                 garden: document.getElementById('garden').checked ? 1 : 0,
#                 pool: document.getElementById('pool').checked ? 1 : 0,
#                 elevator: document.getElementById('elevator').checked ? 1 : 0,
#                 beach_view: document.getElementById('beach_view').checked ? 1 : 0,
#                 mountain_view: document.getElementById('mountain_view').checked ? 1 : 0,
#                 furnished: document.getElementById('furnished').checked ? 1 : 0,
#                 equipped_kitchen: document.getElementById('equipped_kitchen').checked ? 1 : 0,
#                 central_heating: document.getElementById('central_heating').checked ? 1 : 0,
#                 air_conditioning: document.getElementById('air_conditioning').checked ? 1 : 0,
#                 concierge: document.getElementById('concierge').checked ? 1 : 0
#             };
#             try {
#                 const response = await fetch('/predict', {
#                     method: 'POST', headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify(formData)
#                 });
#                 const data = await response.json();
#                 document.getElementById('loading').style.display = 'none';
#                 document.getElementById('result').style.display = 'block';
#                 document.getElementById('priceTND').textContent = data.price_tnd.toLocaleString('en-US') + ' TND';
#                 document.getElementById('priceEUR').textContent = '≈ ' + data.price_eur.toLocaleString('en-US') + ' EUR';
#             } catch (error) {
#                 document.getElementById('loading').style.display = 'none';
#                 alert('Error predicting price. Please try again.');
#             }
#         });
#     </script>
# </body>
# </html>
# '''

# DISTANCE_TO_CAPITAL = {
#     'tunis': 5, 'Ariana': 10, 'Ben Arous': 15, 'Nabeul': 65, ' Sousse': 120,
#     'Sfax': 260, 'Mahdia': 170, 'Djerba': 340, 'La Manouba': 15, 'Jendouba': 160,
#     'Bizerte': 60, 'Gabès': 320, 'Gafsa': 300, 'Béja': 100, 'Monastir': 140, 'Medenine': 360
# }

# def prepare_features(data):
#     features = {}
#     features['Area'] = data['area']
#     features['pieces'] = data['pieces']
#     features['room'] = data['rooms']
#     features['bathroom'] = data['bathrooms']
#     features['distance_to_capital'] = DISTANCE_TO_CAPITAL.get(data['governorate'], 100)
#     features['age_encoded'] = age_order.get(data['age'], 4)
#     features['state'] = data['state']
    
#     binary_cols = ['garage', 'garden', 'concierge', 'beach_view', 'mountain_view',
#                    'pool', 'elevator', 'furnished', 'equipped_kitchen',
#                    'central_heating', 'air_conditioning']
#     for col in binary_cols:
#         features[col] = data.get(col, 0)
        
#     features['total_rooms'] = features['room'] + features['bathroom']
#     features['area_per_room'] = features['Area'] / (features['room'] + 1)
#     features['luxury_score'] = sum(features[f] for f in binary_cols)
#     features['comfort_score'] = features['air_conditioning'] + features['central_heating'] + features['equipped_kitchen']
#     features['outdoor_score'] = features['garden'] + features['pool'] + features['beach_view'] + features['mountain_view']
#     features['privacy_score'] = features['garage'] + features['concierge'] + features['elevator']
#     features['proximity_capital'] = 1 / (features['distance_to_capital'] + 1)
#     features['governorate_rank'] = governorate_rank.get(data['governorate'], len(governorate_rank)//2)
    
#     # Use dictionary .get() with fallback means
#     global_city_mean = np.mean(list(city_means.values()))
#     global_loc_mean = np.mean(list(location_means.values()))
#     global_loc_ppsqm = np.mean(list(loc_price_per_sqm.values()))
#     global_city_ppsqm = np.mean(list(city_price_per_sqm.values()))
    
#     features['city_encoded'] = city_means.get(data['city'], global_city_mean)
#     features['location_encoded'] = location_means.get(data['city'], global_loc_mean)
    
#     # Map the magic features
#     features['location_price_per_sqm'] = loc_price_per_sqm.get(data['city'], global_loc_ppsqm)
#     features['city_price_per_sqm'] = city_price_per_sqm.get(data['city'], global_city_ppsqm)
    
#     features['log_area'] = np.log1p(features['Area'])
    
#     df = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
#     return df

# @app.route('/')
# def index():
#     return render_template_string(
#         HTML_TEMPLATE, model_name=model_info['model_name'],
#         r2_score=f"{model_info['best_r2']*100:.1f}",
#         governorates=list(GOVERNORATES.keys()),
#         governorates_json=str(GOVERNORATES).replace("'", '"'),
#         age_options=AGE_OPTIONS
#     )

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         required = ['governorate', 'city', 'area', 'pieces', 'rooms', 'bathrooms', 'age']
#         for field in required:
#             if not data.get(field): return jsonify({'error': f'Missing: {field}'}), 400
            
#         features_df = prepare_features(data)
        
#         if USE_NN:
#             features_scaled = scaler.transform(features_df)
#             prediction_log = model.predict(features_scaled, verbose=0)[0][0]
#         else:
#             prediction_log = model.predict(features_df)[0]
            
#         price_tnd = float(np.expm1(prediction_log))
#         price_eur = price_tnd / 3.23
        
#         return jsonify({'price_tnd': round(price_tnd, 0), 'price_eur': round(price_eur, 0)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     print("\nStarting House Price Prediction App")
# #     print("Open http://localhost:5000 in your browser\n")
# #     app.run(debug=True, host='0.0.0.0', port=5000)
# if __name__ == '__main__':
#     print("\nStarting House Price Prediction App")
#     print("Open http://localhost:5000 in your browser\n")
#     app.run(debug=False, host='0.0.0.0', port=5000)




















# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================
# st.set_page_config(
#     page_title="Tunisia House Price Prediction 🇹🇳",
#     page_icon="🏠",
#     layout="wide"
# )

# # ============================================================================
# # LOAD MODELS (Cached so it only loads once)
# # ============================================================================
# @st.cache_resource
# def load_model_assets():
#     model = joblib.load('models/best_model.pkl')
#     model_info = joblib.load('models/model_info.pkl')
#     scaler = joblib.load('scaler.pkl')
#     city_means = joblib.load('city_means.pkl')
#     location_means = joblib.load('location_means.pkl')
#     loc_price_per_sqm = joblib.load('loc_price_per_sqm.pkl')
#     city_price_per_sqm = joblib.load('city_price_per_sqm.pkl')
#     governorate_rank = joblib.load('governorate_rank.pkl')
#     age_order = joblib.load('age_order.pkl')
#     return model, model_info, scaler, city_means, location_means, loc_price_per_sqm, city_price_per_sqm, governorate_rank, age_order

# model, model_info, scaler, city_means, location_means, loc_price_per_sqm, city_price_per_sqm, governorate_rank, age_order = load_model_assets()
# FEATURES = model_info['features']
# USE_NN = model_info['use_nn']

# # ============================================================================
# # DATA DICTIONARIES
# # ============================================================================
# GOVERNORATES = {
#     'tunis': ['Tunis', 'El Menzah', 'La Marsa', 'Carthage', 'Le Bardo', 'Le Kram', 
#               'El Ouardia', 'Ettahrir', 'El Omrane', 'El Omrane Superieur', 'Cité El Khadra'],
#     'Ariana': ['Ariana Ville', 'La Soukra', 'Raoued', 'Mnihla', 'Sidi Thabet', 'Chotrana'],
#     'Ben Arous': ['Boumhel Bassatine', 'El Mourouj', 'Ezzahra', 'Rades', 'Mornag', 
#                   'Hammam Lif', 'Mégrine', 'Hammam Chatt', 'Mohamadia'],
#     'Nabeul': ['Hammamet', 'Nabeul', 'Yasmine Hammamet', 'Beni Khiar', 'Kélibia'],
#     ' Sousse': ['Sousse Ville', 'Hammam Sousse', 'Sousse Riadh', 'Sahloul', 'Akouda'],
#     'Sfax': ['Sfax Ville', 'Sfax Sud'],
#     'Mahdia': ['Mahdia Ville'],
#     'Djerba': ['Djerba', 'Houmt Souk', 'Midoun'],
#     'La Manouba': ['La Manouba', 'Mornaguia', 'Denden', 'Oued Ellil'],
#     'Jendouba': ['Tabarka'],
#     'Bizerte': ['Bizerte'],
#     'Gabès': ['Gabès Sud'],
#     'Gafsa': ['Gafsa Sud'],
#     'Béja': ['Medjez El Bab'],
#     'Monastir': ['Monastir Ville'],
#     'Medenine': ['Zarzis']
# }

# DISTANCE_TO_CAPITAL = {
#     'tunis': 5, 'Ariana': 10, 'Ben Arous': 15, 'Nabeul': 65, ' Sousse': 120,
#     'Sfax': 260, 'Mahdia': 170, 'Djerba': 340, 'La Manouba': 15, 'Jendouba': 160,
#     'Bizerte': 60, 'Gabès': 320, 'Gafsa': 300, 'Béja': 100, 'Monastir': 140, 'Medenine': 360
# }

# # ============================================================================
# # PREDICTION FUNCTION
# # ============================================================================
# def prepare_features(data):
#     features = {}
#     features['Area'] = data['area']
#     features['pieces'] = data['pieces']
#     features['room'] = data['rooms']
#     features['bathroom'] = data['bathrooms']
#     features['distance_to_capital'] = DISTANCE_TO_CAPITAL.get(data['governorate'], 100)
#     features['age_encoded'] = age_order.get(data['age'], 4)
#     features['state'] = data['state']
    
#     binary_cols = ['garage', 'garden', 'concierge', 'beach_view', 'mountain_view',
#                    'pool', 'elevator', 'furnished', 'equipped_kitchen',
#                    'central_heating', 'air_conditioning']
#     for col in binary_cols:
#         features[col] = 1 if col in data['amenities'] else 0
        
#     features['total_rooms'] = features['room'] + features['bathroom']
#     features['area_per_room'] = features['Area'] / (features['room'] + 1)
#     features['luxury_score'] = sum(features[f] for f in binary_cols)
#     features['comfort_score'] = features['air_conditioning'] + features['central_heating'] + features['equipped_kitchen']
#     features['outdoor_score'] = features['garden'] + features['pool'] + features['beach_view'] + features['mountain_view']
#     features['privacy_score'] = features['garage'] + features['concierge'] + features['elevator']
#     features['proximity_capital'] = 1 / (features['distance_to_capital'] + 1)
#     features['governorate_rank'] = governorate_rank.get(data['governorate'], len(governorate_rank)//2)
    
#     global_city_mean = np.mean(list(city_means.values()))
#     global_loc_mean = np.mean(list(location_means.values()))
#     global_loc_ppsqm = np.mean(list(loc_price_per_sqm.values()))
#     global_city_ppsqm = np.mean(list(city_price_per_sqm.values()))
    
#     features['city_encoded'] = city_means.get(data['city'], global_city_mean)
#     features['location_encoded'] = location_means.get(data['city'], global_loc_mean)
#     features['location_price_per_sqm'] = loc_price_per_sqm.get(data['city'], global_loc_ppsqm)
#     features['city_price_per_sqm'] = city_price_per_sqm.get(data['city'], global_city_ppsqm)
#     features['log_area'] = np.log1p(features['Area'])
    
#     df = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
#     return df

# # ============================================================================
# # USER INTERFACE
# # ============================================================================
# st.markdown("<h1 style='text-align: center; color: #c0392b;'>🏠 House Price Prediction - Tunisia 🇹🇳</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #666;'>Estimate property values across Tunisia using Machine Learning</h3>", unsafe_allow_html=True)

# # Model Info Bar
# col_m1, col_m2, col_m3 = st.columns(3)
# col_m1.metric(label="ML Model", value=model_info['model_name'])
# col_m2.metric(label="Accuracy (R²)", value=f"{model_info['best_r2']*100:.1f}%")
# col_m3.metric(label="Status", value="✅ Ready")

# st.markdown("---")

# # Input Form
# with st.form("prediction_form"):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📍 Location")
#         gov = st.selectbox("Governorate", options=list(GOVERNORATES.keys()), index=0)
#         city = st.selectbox("City / Location", options=GOVERNORATES[gov])
        
#         st.subheader("📐 Property Size")
#         # area = st.number_input("Area (m²)", min_value=50, max=5000, value=200)
#         # pieces = st.number_input("Number of Pieces", min_value=1, max=30, value=5)
#         area = st.number_input("Area (m2)", min_value=50, max_value=5000, value=200)
#         pieces = st.number_input("Number of Pieces", min_value=1, max_value=30, value=5)
        
#     with col2:
#         st.subheader("🛏️ Rooms")
#         # rooms = st.number_input("Number of Rooms", min_value=1, max=25, value=4)
#         # bathrooms = st.number_input("Number of Bathrooms", min_value=1, max=15, value=2)
#         rooms = st.number_input("Number of Rooms", min_value=1, max_value=25, value=4)
#         bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=15, value=2)
        
#         st.subheader("📅 Age & Condition")
#         age = st.selectbox("Property Age", options=['0', '1-5', '5-10', '10-20', '20-30', '30-50', '50-70', '70-100', 'Plus de 100'], index=1)
#         state = st.selectbox("Condition", options=[1, 2, 3], format_func=lambda x: ["New / Excellent", "Good Condition", "Needs Renovation"][x-1], index=1)

#     st.subheader("✨ Features & Amenities")
#     amenities = st.multiselect(
#         "Select available features:",
#         options=['garage', 'garden', 'pool', 'elevator', 'beach_view', 'mountain_view', 
#                  'furnished', 'equipped_kitchen', 'central_heating', 'air_conditioning', 'concierge'],
#         format_func=lambda x: x.replace('_', ' ').title()
#     )

#     submitted = st.form_submit_button("🏠 Predict Price", use_container_width=True, type="primary")

# # ============================================================================
# # PREDICTION LOGIC & RESULTS
# # ============================================================================
# if submitted:
#     with st.spinner('Analyzing property details...'):
#         form_data = {
#             'governorate': gov,
#             'city': city,
#             'area': area,
#             'pieces': pieces,
#             'rooms': rooms,
#             'bathrooms': bathrooms,
#             'age': age,
#             'state': state,
#             'amenities': amenities
#         }
        
#         features_df = prepare_features(form_data)
        
#         if USE_NN:
#             features_scaled = scaler.transform(features_df)
#             prediction_log = model.predict(features_scaled, verbose=0)[0][0]
#         else:
#             prediction_log = model.predict(features_df)[0]
            
#         price_tnd = float(np.expm1(prediction_log))
#         price_eur = price_tnd / 3.23

#     # Display Results
#     st.markdown("---")
#     st.success("### 🎯 Estimated Property Price")
    
#     col_res1, col_res2 = st.columns(2)
#     col_res1.metric(label="Price (TND)", value=f"{price_tnd:,.0f} TND")
#     col_res2.metric(label="Price (EUR)", value=f"{price_eur:,.0f} EUR")
    
#     st.caption("*This is an estimate based on historical market data across Tunisia.")











import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64 

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Tunisia House Price Prediction 🇹🇳",
    page_icon="🏠",
    layout="wide"
)


# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_model_assets():
    model = joblib.load('models/best_model.pkl')
    model_info = joblib.load('models/model_info.pkl')
    scaler = joblib.load('scaler.pkl')
    city_means = joblib.load('city_means.pkl')
    location_means = joblib.load('location_means.pkl')
    loc_price_per_sqm = joblib.load('loc_price_per_sqm.pkl')
    city_price_per_sqm = joblib.load('city_price_per_sqm.pkl')
    governorate_rank = joblib.load('governorate_rank.pkl')
    age_order = joblib.load('age_order.pkl')
    return model, model_info, scaler, city_means, location_means, loc_price_per_sqm, city_price_per_sqm, governorate_rank, age_order


model, model_info, scaler, city_means, location_means, loc_price_per_sqm, city_price_per_sqm, governorate_rank, age_order = load_model_assets()
FEATURES = model_info['features']
USE_NN = model_info['use_nn']


# ============================================================================
# DATA DICTIONARIES
# ============================================================================
GOVERNORATES = {
    'tunis': ['Tunis', 'El Menzah', 'La Marsa', 'Carthage', 'Le Bardo', 'Le Kram', 
              'El Ouardia', 'Ettahrir', 'El Omrane', 'El Omrane Superieur', 'Cité El Khadra'],
    'Ariana': ['Ariana Ville', 'La Soukra', 'Raoued', 'Mnihla', 'Sidi Thabet', 'Chotrana'],
    'Ben Arous': ['Boumhel Bassatine', 'El Mourouj', 'Ezzahra', 'Rades', 'Mornag', 
                  'Hammam Lif', 'Mégrine', 'Hammam Chatt', 'Mohamadia'],
    'Nabeul': ['Hammamet', 'Nabeul', 'Yasmine Hammamet', 'Beni Khiar', 'Kélibia'],
    'Sousse': ['Sousse Ville', 'Hammam Sousse', 'Sousse Riadh', 'Sahloul', 'Akouda'],
    'Sfax': ['Sfax Ville', 'Sfax Sud'],
    'Mahdia': ['Mahdia Ville'],
    'Djerba': ['Djerba', 'Houmt Souk', 'Midoun'],
    'La Manouba': ['La Manouba', 'Mornaguia', 'Denden', 'Oued Ellil'],
    'Jendouba': ['Tabarka'],
    'Bizerte': ['Bizerte'],
    'Gabès': ['Gabès Sud'],
    'Gafsa': ['Gafsa Sud'],
    'Béja': ['Medjez El Bab'],
    'Monastir': ['Monastir Ville'],
    'Medenine': ['Zarzis']
}

# Clean keys (remove leading/trailing spaces)
GOVERNORATES = {k.strip(): v for k, v in GOVERNORATES.items()}

DISTANCE_TO_CAPITAL = {
    'tunis': 5, 'Ariana': 10, 'Ben Arous': 15, 'Nabeul': 65, 'Sousse': 120,
    'Sfax': 260, 'Mahdia': 170, 'Djerba': 340, 'La Manouba': 15, 'Jendouba': 160,
    'Bizerte': 60, 'Gabès': 320, 'Gafsa': 300, 'Béja': 100, 'Monastir': 140, 'Medenine': 360
}


# ============================================================================
# SESSION STATE & CALLBACK
# ============================================================================
if "gov" not in st.session_state:
    st.session_state.gov = "tunis"
if "city" not in st.session_state:
    st.session_state.city = GOVERNORATES[st.session_state.gov][0]


def on_governorate_change():
    gov = st.session_state.gov
    cities = GOVERNORATES.get(gov, [])
    if cities:
        st.session_state.city = cities[0]


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def prepare_features(data):
    features = {}
    features['Area'] = data['area']
    features['pieces'] = data['pieces']
    features['room'] = data['rooms']
    features['bathroom'] = data['bathrooms']
    features['distance_to_capital'] = DISTANCE_TO_CAPITAL.get(data['governorate'], 100)
    features['age_encoded'] = age_order.get(data['age'], 4)
    features['state'] = data['state']
    
    binary_cols = ['garage', 'garden', 'concierge', 'beach_view', 'mountain_view',
                   'pool', 'elevator', 'furnished', 'equipped_kitchen',
                   'central_heating', 'air_conditioning']
    for col in binary_cols:
        features[col] = 1 if col in data['amenities'] else 0
        
    features['total_rooms'] = features['room'] + features['bathroom']
    features['area_per_room'] = features['Area'] / (features['room'] + 1)
    features['luxury_score'] = sum(features[f] for f in binary_cols)
    features['comfort_score'] = features['air_conditioning'] + features['central_heating'] + features['equipped_kitchen']
    features['outdoor_score'] = features['garden'] + features['pool'] + features['beach_view'] + features['mountain_view']
    features['privacy_score'] = features['garage'] + features['concierge'] + features['elevator']
    features['proximity_capital'] = 1 / (features['distance_to_capital'] + 1)
    features['governorate_rank'] = governorate_rank.get(data['governorate'], len(governorate_rank)//2)
    
    global_city_mean = np.mean(list(city_means.values()))
    global_loc_mean = np.mean(list(location_means.values()))
    global_loc_ppsqm = np.mean(list(loc_price_per_sqm.values()))
    global_city_ppsqm = np.mean(list(city_price_per_sqm.values()))
    
    features['city_encoded'] = city_means.get(data['city'], global_city_mean)
    features['location_encoded'] = location_means.get(data['city'], global_loc_mean)
    features['location_price_per_sqm'] = loc_price_per_sqm.get(data['city'], global_loc_ppsqm)
    features['city_price_per_sqm'] = city_price_per_sqm.get(data['city'], global_city_ppsqm)
    features['log_area'] = np.log1p(features['Area'])
    
    df = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    return df


# ============================================================================
# UI: HEADER & INFO
# ============================================================================
st.markdown("""
<div style="text-align:center;">
    <img src="data:image/png;base64,{}" width="320">
  
</div>
""".format(base64.b64encode(open("logo_2.png", "rb").read()).decode()),
unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: #c0392b;'>🏠 House Price Prediction - Tunisia 🇹🇳</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #666;'>Estimate property values across Tunisia using Machine Learning</h3>", unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(label="ML Model", value=model_info['model_name'])
col_m2.metric(label="Accuracy (R²)", value=f"{model_info['best_r2']*100:.1f}%")
col_m3.metric(label="Status", value="✅ Ready")

st.markdown("---")

# ============================================================================
# LOCATION SELECTORS (OUTSIDE FORM, with on_change)
# ============================================================================
st.subheader("📍 Location")
gov = st.selectbox(
    "Governorate",
    options=list(GOVERNORATES.keys()),
    index=list(GOVERNORATES.keys()).index(st.session_state.gov),
    key="gov",
    on_change=on_governorate_change
)

city_options = GOVERNORATES.get(gov, [])
city = st.selectbox(
    "City / Location",
    options=city_options,
    index=0 if st.session_state.city not in city_options else city_options.index(st.session_state.city),
    key="city"
)

# ============================================================================
# FORM FOR OTHER INPUTS
# ============================================================================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Property Size")
        area = st.number_input("Area (m2)", min_value=50, max_value=5000, value=200)
        pieces = st.number_input("Number of Pieces", min_value=1, max_value=30, value=5)
        
    with col2:
        st.subheader("🛏️ Rooms")
        rooms = st.number_input("Number of Rooms", min_value=1, max_value=25, value=4)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=15, value=2)
        
        st.subheader("📅 Age & Condition")
        age = st.selectbox(
            "Property Age",
            options=['0', '1-5', '5-10', '10-20', '20-30', '30-50', '50-70', '70-100', 'Plus de 100'],
            index=1
        )
        state = st.selectbox(
            "Condition",
            options=[1, 2, 3],
            format_func=lambda x: ["New / Excellent", "Good Condition", "Needs Renovation"][x-1],
            index=1
        )

    st.subheader("✨ Features & Amenities")
    amenities = st.multiselect(
        "Select available features:",
        options=['garage', 'garden', 'pool', 'elevator', 'beach_view', 'mountain_view', 
                 'furnished', 'equipped_kitchen', 'central_heating', 'air_conditioning', 'concierge'],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    submitted = st.form_submit_button("🏠 Predict Price", use_container_width=True, type="primary")


# ============================================================================
# PREDICTION LOGIC & RESULTS
# ============================================================================
if submitted:
    with st.spinner('Analyzing property details...'):
        form_data = {
            'governorate': gov,
            'city': city,
            'area': area,
            'pieces': pieces,
            'rooms': rooms,
            'bathrooms': bathrooms,
            'age': age,
            'state': state,
            'amenities': amenities
        }
        
        features_df = prepare_features(form_data)
        
        if USE_NN:
            features_scaled = scaler.transform(features_df)
            prediction_log = model.predict(features_scaled, verbose=0)[0][0]
        else:
            prediction_log = model.predict(features_df)[0]
            
        price_tnd = float(np.expm1(prediction_log))
        price_eur = price_tnd / 3.23

    st.markdown("---")
    st.success("### 🎯 Estimated Property Price")
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label="Price (TND)", value=f"{price_tnd:,.0f} TND")
    col_res2.metric(label="Price (EUR)", value=f"{price_eur:,.0f} EUR")
    
    st.caption("*This is an estimate based on historical market data across Tunisia.")