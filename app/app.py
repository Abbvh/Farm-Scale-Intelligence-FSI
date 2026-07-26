# ============================================
# FSI v1.0 — Flask Web Application
# Farm Scale Intelligence
# ============================================

from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os
import gdown

app = Flask(__name__)

# ── Model download from Google Drive if not present ──
MODEL_V1_ID  = "1pJjmLeB44luRqOaW3wdYMlhcYHabbwPB"
MODEL_V1B_ID = "1d3se1uUwnE9-R90QzWqzLx9mPjSnNuA8"

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/FSI_v1_mastitis_model.pkl"):
    print("Downloading FSI v1.0 model...")
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_V1_ID}",
        "models/FSI_v1_mastitis_model.pkl", quiet=False)

if not os.path.exists("models/FSI_v1b_precalving_model.pkl"):
    print("Downloading FSI v1.0b model...")
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_V1B_ID}",
        "models/FSI_v1b_precalving_model.pkl", quiet=False)

# Load models
print("Loading FSI models...")
model_v1  = joblib.load("models/FSI_v1_mastitis_model.pkl")
model_v1b = joblib.load("models/FSI_v1b_precalving_model.pkl")

with open("models/FSI_v1_features.json") as f:
    features_v1 = json.load(f)

# SHAP explainers
explainer_v1  = shap.TreeExplainer(model_v1)
explainer_v1b = shap.TreeExplainer(model_v1b)

print("FSI models loaded successfully")
print("Starting FSI web app...")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    try:
        data = request.json
        input_data = {
            'parity':                  data.get('parity', 2),
            'days_in_milk':            data.get('days_in_milk', 60),
            'breed':                   data.get('breed', 0),
            'milk_yield_kg':           data.get('milk_yield_kg', 28),
            'electrical_conductivity': data.get('electrical_conductivity', 4.8),
            'activity_steps':          data.get('activity_steps', 3800),
            'rumination_time':         data.get('rumination_time', 480),
            'body_temp':               data.get('body_temp', 38.7),
            'milk_fat_pct':            data.get('milk_fat_pct', 3.9),
            'milk_protein_pct':        data.get('milk_protein_pct', 3.2),
            'yield_deviation':         data.get('milk_yield_kg', 28) - 28.0,
            'early_lactation':         1 if data.get('days_in_milk', 60) < 100 else 0,
            'stress_score': (
                (1 if data.get('activity_steps', 3800) < 2800 else 0) +
                (1 if data.get('rumination_time', 480) < 400 else 0) +
                (1 if data.get('body_temp', 38.7) > 39.2 else 0)
            )
        }

        df_input = pd.DataFrame([input_data])
        prob     = float(model_v1.predict_proba(df_input)[0][1])
        prediction = int(model_v1.predict(df_input)[0])
        shap_vals  = explainer_v1.shap_values(df_input)[0]
        shap_dict  = {
            feat: round(float(val), 3)
            for feat, val in zip(features_v1, shap_vals)
        }
        shap_sorted = dict(sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]), reverse=True))

        return jsonify({
            'probability':     round(prob, 4),
            'probability_pct': round(prob * 100, 1),
            'prediction':      prediction,
            'risk_level':      'high' if prob >= 0.65 else 'medium' if prob >= 0.35 else 'low',
            'shap_values':     shap_sorted,
            'input_features':  input_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_precalving', methods=['POST'])
def predict_precalving():
    try:
        data = request.json
        input_data = {
            'parity_prior_lactation':              data.get('parity', 2),
            'n_scc_records_previous_lactation':    data.get('n_scc_records', 8),
            'dry_off_scc_missing':                 data.get('dry_off_scc_missing', 0),
            'dry_off_clinical_mastitis_within_7d': data.get('dry_off_clinical_mastitis', 0),
            'dry_off_imi_status_paper_rule':       data.get('dry_off_imi_status', 0),
            'calving_interval_days':               data.get('calving_interval', 380),
            'lactation_length_days':               data.get('lactation_length', 305),
            'month_of_dry_off':                    data.get('month_dry_off', 6),
            'month_of_calving':                    data.get('month_calving', 9),
            'first_scc_prev_lactation_cells_ml':   data.get('first_scc', 50000),
            'median_scc_prev_lactation_cells_ml':  data.get('median_scc', 80000),
            'mean_scc_prev_lactation_cells_ml':    data.get('mean_scc', 90000),
            'min_scc_prev_lactation_cells_ml':     data.get('min_scc', 20000),
            'max_scc_prev_lactation_cells_ml':     data.get('max_scc', 350000),
            'mean_first3_scc_prev_lactation_cells_ml': data.get('mean_first3_scc', 60000),
            'mean_last3_scc_prev_lactation_cells_ml':  data.get('mean_last3_scc', 120000),
            'ratio_last3_to_first3_scc':           data.get('ratio_last3_first3', 1.5),
            'last_scc_prev_lactation_cells_ml':    data.get('last_scc', 150000),
            'consecutive_3_scc_gt_200k_prev_lactation': data.get('consecutive_high_scc', 0),
            'pct_scc_lt_50k_prev_lactation':       data.get('pct_lt_50k', 40),
            'pct_scc_gt_100k_prev_lactation':      data.get('pct_gt_100k', 25),
            'pct_scc_gt_200k_prev_lactation':      data.get('pct_gt_200k', 15),
            'pct_scc_gt_400k_prev_lactation':      data.get('pct_gt_400k', 5),
            'pct_scc_gt_1000k_prev_lactation':     data.get('pct_gt_1000k', 1),
            'peak_yield_prev_lactation_kg':        data.get('peak_yield', 40),
            'yield_at_last_recording_kg':          data.get('last_yield', 25),
            'herd_new_dry_period_imi_6mo_rate':    data.get('herd_imi_6mo', 0.15),
            'herd_new_dry_period_imi_year_rate':   data.get('herd_imi_year', 0.14),
            'herd_mean_scc_k_cells_ml':            data.get('herd_mean_scc', 200),
            'herd_predicted_305d_yield_l':         data.get('herd_305d_yield', 9000),
        }

        df_input   = pd.DataFrame([input_data])
        prob       = float(model_v1b.predict_proba(df_input)[0][1])
        prediction = int(model_v1b.predict(df_input)[0])
        shap_vals  = explainer_v1b.shap_values(df_input)[0]
        shap_dict  = {
            feat: round(float(val), 3)
            for feat, val in zip(input_data.keys(), shap_vals)
        }
        shap_sorted = dict(sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]), reverse=True)[:10])

        return jsonify({
            'probability':     round(prob, 4),
            'probability_pct': round(prob * 100, 1),
            'prediction':      prediction,
            'risk_level':      'high' if prob >= 0.65 else 'medium' if prob >= 0.35 else 'low',
            'shap_values':     shap_sorted,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
