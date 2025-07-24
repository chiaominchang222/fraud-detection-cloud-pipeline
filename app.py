from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd  
import traceback
import time

app = Flask(__name__)

scaler = joblib.load('scaler/scaler.pkl')
voting_model = joblib.load('models/best_voting_model.pkl')

# The order of input features expected by the scaler/model
feature_order = list(scaler.feature_names_in_)

# Preprocess raw input by converting to float and ordering according to feature_order
def preprocess(raw):
    processed = {}
    for col in feature_order:
        processed[col] = float(raw.get(col, 0.0))  # If missing, default to 0.0
    return processed

# API endpoint for batch prediction
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        batch_start_time = time.time()
        print(">>> [Flask] Received /predict_batch request")

        data = request.json
        if not data or 'features_list' not in data:
            return jsonify({'error': 'Missing "features_list" field in request'}), 400

        features_list = data['features_list']
        if not isinstance(features_list, list):
            return jsonify({'error': '"features_list" must be a list of feature dicts'}), 400

        results = []
        for idx, raw_input in enumerate(features_list):
            input_features = preprocess(raw_input)
            ordered_values = [input_features[col] for col in feature_order]

            features_df = pd.DataFrame([ordered_values], columns=feature_order)
            features_scaled = scaler.transform(features_df)

            proba = voting_model.predict_proba(features_scaled)[0][1]
            prediction = voting_model.predict(features_scaled)[0]

            result = {
                'fraud_probability': round(float(proba), 6),
                'prediction': int(prediction)
            }

            # ⬇️ Add traceable fields such as cc_num, name, merchant, amt if present
            for field in ['cc_num', 'name', 'merchant', 'amt']:
                if field in raw_input:
                    result[field] = raw_input[field]

            results.append(result)

        print(f">>> [Flask] Completed batch, total = {len(results)} records, time used = {round(time.time() - batch_start_time, 2)} sec")

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# API endpoint to expose expected feature names
@app.route('/features', methods=['GET'])
def get_feature_names():
    return jsonify({'expected_feature_order': feature_order})

# Run the Flask app on all interfaces so it's accessible externally
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
