"""
Capstone Web Application
========================
Integrates all 5 models into a single web interface using Streamlit.

Run locally:  streamlit run webapp/app.py
Deploy:       Push to GitHub, then connect to Streamlit Community Cloud
              https://streamlit.io/cloud (free hosting)
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
# Avoid OMP/OpenBLAS fork-safety issues in Streamlit's multi-thread runner
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import streamlit as st
from pathlib import Path
import sys
import json
from PIL import Image
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# predict.py uses `from train import ...` so model5_innovation must be on sys.path
_MODEL5_DIR = str(ROOT_DIR / "models" / "model5_innovation")
if _MODEL5_DIR not in sys.path:
    sys.path.insert(0, _MODEL5_DIR)

traditional_feature_controls = { 
    'Distance(mi)': { 'label': 'Distance (mi)', 'min': 0, 'max': 100, 'control': 'slider' },
    'Timezone': { 'label': 'Timezone', 'options': {"UTC+5": 5, "UTC+6": 6,"UTC+7": 7,"UTC+8": 8}, 'control': 'selectbox' },
    'Temperature(F)': { 'label': 'Temperature (F)', 'min': -50, 'max': 150, 'control': 'slider' },
    'Wind_Chill(F)': { 'label': 'Wind Chill (F)', 'min': -50, 'max': 150, 'control': 'slider' },
    'Humidity(%)': { 'label': 'Humidity (%)', 'min': 0, 'max': 100, 'control': 'slider' },
    'Pressure(in)': { 'label': 'Pressure (in)', 'min': 28, 'max': 32, 'control': 'slider' },
    'Visibility(mi)': { 'label': 'Visibility (mi)', 'min': 0, 'max': 10, 'control': 'slider' },
    'Wind_Speed(mph)': { 'label': 'Wind Speed (mph)', 'min': 0, 'max': 100, 'control': 'slider' },
    'Precipitation(in)': { 'label': 'Precipitation (in)', 'min': 0, 'max': 10, 'control': 'slider' },
    'Amenity': { 'label': 'Amenity', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Bump': { 'label': 'Bump', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Crossing': { 'label': 'Crossing', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Give_Way': { 'label': 'Give Way', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Junction': { 'label': 'Junction', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'No_Exit': { 'label': 'No Exit', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Railway': { 'label': 'Railway', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Roundabout': { 'label': 'Roundabout', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Station': { 'label': 'Station', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Stop': { 'label': 'Stop', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Traffic_Calming': { 'label': 'Traffic Calming', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Traffic_Signal': { 'label': 'Traffic Signal', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Turning_Loop': { 'label': 'Turning Loop', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Sunrise_Sunset': { 'label': 'Sunrise Sunset', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Civil_Twilight': { 'label': 'Civil Twilight', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Nautical_Twilight': { 'label': 'Nautical Twilight', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Astronomical_Twilight': { 'label': 'Astronomical Twilight', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'hour': { 'label': 'Hour', 'min': 0, 'max': 23, 'control': 'slider' },
    'day_of_week': { 'label': 'Day of Week', 'options': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 'control': 'selectbox' },
    'month': { 'label': 'Month', 'options': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], 'control': 'selectbox' },
    'is_weekend': { 'label': 'Is Weekend', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_morning_rush': { 'label': 'Is Morning Rush', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_evening_rush': { 'label': 'Is Evening Rush', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_rush_hour': { 'label': 'Is Rush Hour', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'duration_min': { 'label': 'Duration (min)', 'min': 0, 'max': 120, 'control': 'slider' },
    'wind_dir_deg': { 'label': 'Wind Direction (deg)', 'min': 0, 'max': 360, 'control': 'slider' },
    'weather_cond_num': { 'label': 'Weather Condition', 'min': 0, 'max': 10, 'control': 'slider' },
    'weather_data_available': { 'label': 'Weather Data Available', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'is_freezing': { 'label': 'Is Freezing', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'low_visibility': { 'label': 'Low Visibility', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'accident_dir': { 'label': 'Accident Direction', 'options': ['North', 'South', 'East', 'West'], 'control': 'selectbox' },
    'lat_bin': { 'label': 'Latitude Bin', 'min': 0, 'max': 10, 'control': 'slider' },
    'n_road_features': { 'label': 'Number of Road Features', 'min': 0, 'max': 10, 'control': 'slider' },
    'has_traffic_control': { 'label': 'Has Traffic Control', 'options': ['Yes', 'No'], 'control': 'selectbox' },
    'Severity_Binary': { 'label': 'Severity Binary', 'options': ['Yes', 'No'], 'control': 'selectbox' }
}


def convert_feature_value(feature_name, value):
    control_config = traditional_feature_controls[feature_name]

    # Dictionary-backed selectbox, e.g. Timezone
    if isinstance(control_config.get("options"), dict):
        return control_config["options"][value]

    # Yes/No selectboxes
    if value == "Yes":
        return 1
    if value == "No":
        return 0

    # Day names
    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    if feature_name == "day_of_week":
        return day_map[value]

    # Month names
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    if feature_name == "month":
        return month_map[value]

    # Direction labels
    accident_dir_map = {
        "North": 0,
        "East": 1,
        "South": 2,
        "West": 3,
    }

    if feature_name == "accident_dir":
        return accident_dir_map[value]

    return value

# Page config
st.set_page_config(
    page_title="SmartCity FSA | AI Dashboard",
    page_icon="🏙️",
    layout="wide",
)

# Header
st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <h1 style='font-size:2.6rem; margin-bottom:0;'>🏙️ SmartCity FSA</h1>
        <p style='font-size:1.1rem; color:#90CAF9; margin-top:0.3rem;'>
            AI-Powered Urban Intelligence Platform &nbsp;|&nbsp; vibetastrophe.com
        </p>
    </div>
    <hr style='border:1px solid #1E88E5; margin-bottom:1.5rem;'>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/fluency/96/city.png", width=60)
st.sidebar.markdown("## SmartCity FSA")
st.sidebar.markdown("AI Urban Intelligence")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Home",
        "Model 1: Traditional ML",
        "Model 2: Deep Learning",
        "Model 3: CNN (Image Classification)",
        "Model 4: NLP (Text Classification)",
        "Model 5: Innovation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("vibetastrophe.com")

# ---------------------------------------------------------------------------
# Model pages
# ---------------------------------------------------------------------------

if model_choice == "Home":
    st.markdown("### Welcome to SmartCity FSA AI Platform")
    st.markdown("Use the sidebar to navigate between AI models. Each model makes real-time predictions on urban data.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Model 1: Traditional ML**\n\nPredicts traffic accident severity using Random Forest on weather & road features.")
        st.info("**Model 2: Deep Learning**\n\nDeep neural network for accident severity prediction using the same feature set.")
    with col2:
        st.info("**Model 3: CNN**\n\nEfficientNetB0 image classifier — detects potholes from road images.")
        st.info("**Model 4: NLP**\n\nDistilBERT transformer — classifies 311 complaint descriptions into 6 categories.")
    with col3:
        st.info("**Model 5: Innovation**\n\nXGBoost road deterioration predictor using NYC 311 data. Rates severity: Low → Critical.")
        st.success("**Live Data**\n\n434,722 NYC 311 complaints processed for batch predictions.")

    st.markdown("---")
    st.markdown("""
        <div style='text-align:center; color:#90CAF9; font-size:0.9rem;'>
            SmartCity FSA &nbsp;|&nbsp; AI Capstone Project &nbsp;|&nbsp; www.smartcityfsa.com
        </div>
    """, unsafe_allow_html=True)

elif model_choice == "Model 1: Traditional ML":
    st.header("Model 1: Traditional ML")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    @st.cache_resource
    def load_model1():
        return joblib.load(str(ROOT_DIR / "models/model1_traditional_ml/saved_model/model.joblib"))

    @st.cache_resource
    def load_scaler1():
        return joblib.load(str(ROOT_DIR / "models/model1_traditional_ml/saved_model/scaler.joblib"))

    @st.cache_resource
    def load_feature_cols():
        return joblib.load(str(ROOT_DIR / "models/model1_traditional_ml/saved_model/feature_columns.joblib"))
    
    loaded_feature_cols = load_feature_cols()
    feature_cols = [
        col for col in loaded_feature_cols if col not in {"Severity"}
    ]
    model = load_model1()
    scaler = load_scaler1()
    total_columns = len(feature_cols)
    # Create input fields for your features
    col1, col2 = st.columns(2)
    with col1:
        for i in range(total_columns // 2):
            if traditional_feature_controls[feature_cols[i]]['control'] == 'number_input':
                st.number_input(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'slider':
                st.slider(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'selectbox':
                st.selectbox(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    options=traditional_feature_controls[feature_cols[i]]['options'],
                    key=feature_cols[i]
                )
        # feature_1 = st.number_input("Feature 1", value=0.0)
        # feature_2 = st.selectbox("Feature 2", ["Option A", "Option B"])
    with col2:
        for i in range(total_columns // 2, total_columns):
            if traditional_feature_controls[feature_cols[i]]['control'] == 'number_input':
                st.number_input(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'selectbox':
                st.selectbox(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    options=traditional_feature_controls[feature_cols[i]]['options'],
                    key=feature_cols[i]
                )
            elif traditional_feature_controls[feature_cols[i]]['control'] == 'slider':
                st.slider(
                    traditional_feature_controls[feature_cols[i]]['label'],
                    min_value=traditional_feature_controls[feature_cols[i]]['min'],
                    max_value=traditional_feature_controls[feature_cols[i]]['max'],
                    key=feature_cols[i]
                )

    if st.button("Predict"):
        
        input_data = {}

        for feature in feature_cols:
            raw_value = st.session_state[feature]
            input_data[feature] = convert_feature_value(feature, raw_value)

        # input_data.pop("Severity_Binary", None)  # Remove target if accidentally included 
        input_df = pd.DataFrame([input_data])
        input_df[feature_cols] = scaler.transform(input_df[feature_cols])

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        st.success(f"Prediction: {prediction[0]}")
        st.write(f"Confidence: {probability.max():.2%}")
    # ---- END PATTERN ----

    st.info("Enter values above and click **Predict** to classify traffic accident severity.")

    st.markdown("---")
    st.subheader("Feature Importance")
    st.caption("Top 15 features by Random Forest Gini importance.")
    _importance_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(15)
        .set_index("Feature")
    )
    st.bar_chart(_importance_df)

elif model_choice == "Model 2: Deep Learning":
    st.header("Model 2: Deep Learning")
    # TODO: Load your DNN model and add prediction interface
    # Same pattern as Model 1, but load with:
    import tensorflow as tf

    @st.cache_resource
    def load_model2():
        return tf.keras.models.load_model(str(ROOT_DIR / "models/model2_deep_learning/saved_model/model.keras"))

    @st.cache_resource
    def load_scaler2():
        return joblib.load(str(ROOT_DIR / "models/model2_deep_learning/saved_model/scaler.joblib"))
    
        # The 29-column list used at training time (includes intentional duplicates)
    model2_features = [
        'Distance(mi)', 'Timezone',
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'wind_dir_deg', 'weather_cond_num', 'accident_dir',
        'hour', 'day_of_week', 'month', 'is_weekend',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'duration_min', 'wind_dir_deg', 'weather_cond_num', 'weather_data_available',
        'is_freezing', 'low_visibility', 'accident_dir', 'lat_bin',
        'n_road_features', 'has_traffic_control'
    ]
    # Unique features for UI (order-preserved, no duplicates)
    features = list(dict.fromkeys(model2_features))

    model = load_model2()
    scaler = load_scaler2()

    total_columns = len(features)
    col1, col2 = st.columns(2)
    with col1:
        for i in range(total_columns // 2):
            feature = features[i]
            control_config = traditional_feature_controls[feature]
            if control_config['control'] == 'number_input':
                st.number_input(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'slider':
                st.slider(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'selectbox':
                st.selectbox(
                    control_config['label'],
                    options=control_config['options'],
                    key=feature
                )
    with col2:
        for i in range(total_columns // 2, total_columns):
            feature = features[i]
            control_config = traditional_feature_controls[feature]
            if control_config['control'] == 'number_input':
                st.number_input(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'slider':
                st.slider(
                    control_config['label'],
                    min_value=control_config['min'],
                    max_value=control_config['max'],
                    key=feature
                )
            elif control_config['control'] == 'selectbox':
                st.selectbox(
                    control_config['label'],
                    options=control_config['options'],
                    key=feature
                )

    

    if st.button("Predict"):
        input_data = {}

        for feature in features:
            raw_value = st.session_state[feature]
            input_data[feature] = convert_feature_value(feature, raw_value)
            
        input_df = pd.DataFrame([input_data])[model2_features]  # 29 cols with duplicates
        scaled_input_df = scaler.transform(input_df)
        predictions = model.predict(scaled_input_df)
        label = "High Severity" if predictions[0][0] >= 0.5 else "Low Severity"
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {predictions[0][0]:.2%}")

    st.markdown("---")
    st.subheader("Model 1 vs Model 2 — Architecture Comparison")
    _cmp = pd.DataFrame({
        "": [
            "Architecture",
            "Output",
            "Imbalance handling",
            "Interpretability",
            "Best for",
        ],
        "Model 1 — Random Forest": [
            "Ensemble of decision trees",
            "Severity 1 / 2 / 3 / 4",
            "Class weights + balanced sampling",
            "High — feature importance",
            "Resource prioritisation",
        ],
        "Model 2 — Deep Neural Network": [
            "3-layer fully connected DNN",
            "High (Sev 3–4) or Low (Sev 1–2)",
            "Class weights in binary cross-entropy",
            "Moderate",
            "Fast operational triage",
        ],
    }).set_index("")
    st.table(_cmp)
    st.caption(
        "Model 2 uses binary output by design — it isolates the DNN vs. ensemble comparison "
        "from task complexity. Model 1 handles the full 1–4 breakdown."
    )

elif model_choice == "Model 3: CNN (Image Classification)":
    from models.model3_cnn.predict import THRESHOLD, predict as predict_model3
    st.header("Model 3: CNN — Pothole Image Classification")
    st.markdown(
        "Upload a road image (or load a sample below) to classify it as "
        "**Pothole (Positive)** or **No Pothole (Negative)**."
    )

    @st.cache_resource
    def load_model3():
        import tensorflow as tf
        model_path = ROOT_DIR / "models" / "model3_cnn" / "saved_model" / "efficientnet_model.keras"
        return tf.keras.models.load_model(str(model_path))

    model = load_model3()

    # --- Sample image buttons ---
    _img_root = ROOT_DIR / "data" / "raw" / "pothole_images"
    _samples = {
        "Positive 1": _img_root / "positive" / "G0010033.JPG",
        "Positive 2": _img_root / "positive" / "G0010117.JPG",
        "Negative 1": _img_root / "negative" / "G0015965.JPG",
        "Negative 2": _img_root / "negative" / "G0016163.JPG",
    }

    if 'model3_sample_path' not in st.session_state:
        st.session_state['model3_sample_path'] = None

    st.markdown("**Load a sample image:**")
    _scols = st.columns(len(_samples))
    for col, (label, path) in zip(_scols, _samples.items()):
        if col.button(label, use_container_width=True):
            st.session_state['model3_sample_path'] = str(path)

    st.markdown("---")

    uploaded_file = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"])

    # Resolve image source: upload takes priority over sample
    _image_source = None
    _caption = ""
    if uploaded_file is not None:
        uploaded_file.seek(0)
        _image_source = uploaded_file
        _caption = uploaded_file.name
    elif st.session_state.get('model3_sample_path'):
        _image_source = st.session_state['model3_sample_path']
        _caption = Path(_image_source).name

    if _image_source is not None:
        if hasattr(_image_source, 'seek'):
            _image_source.seek(0)
        st.image(Image.open(_image_source), caption=_caption, use_container_width=True)

        if st.button("Classify"):
            if hasattr(_image_source, 'seek'):
                _image_source.seek(0)

            # Mirror predict.py preprocessing: RGB + 224x224 + raw pixel array.
            img = Image.open(_image_source).convert("RGB").resize((224, 224))
            img_array = np.array(img)
            predictions = predict_model3(model, np.array([img_array]), [_caption or "uploaded_image"])
            pred_class = int(predictions.loc[0, "predicted_class"])
            confidence = float(predictions.loc[0, "confidence"])

            if pred_class == 1:
                st.success(f"Pothole detected — {confidence:.2%} confidence")
            else:
                st.error(f"No pothole — {confidence:.2%} confidence")
            st.caption(f"Decision threshold: {THRESHOLD:.2f}")

elif model_choice == "Model 4: NLP (Text Classification)":
    st.header("Model 4: NLP — 311 Complaint Classification")
    st.markdown(
        "Paste or load a **resolution description** from a 311 complaint to classify "
        "it into one of 6 complaint categories."
    )

    _M4_DIR = ROOT_DIR / "models" / "model4_nlp_classification" / "saved_model"
    _distilbert_weights = _M4_DIR / "model.safetensors"

    if _distilbert_weights.exists():
        st.success("Active model: DistilBERT (fine-tuned transformer)")
    else:
        st.error("DistilBERT weights not found. Expected: models/model4_nlp_classification/saved_model/model.safetensors")

    def _m4_get_device():
        import torch

        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @st.cache_resource
    def load_model4_distilbert():
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        device = _m4_get_device()
        model = DistilBertForSequenceClassification.from_pretrained(_M4_DIR)
        tokenizer = DistilBertTokenizerFast.from_pretrained(_M4_DIR)
        model.to(device)
        model.eval()

        with open(_M4_DIR / "id2label.json", "r") as f:
            id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

        return model, tokenizer, id2label, device

    _m4_model, _m4_tokenizer, _m4_id2label, _m4_device = load_model4_distilbert()

    _M4_SAMPLES = {
        "Blocked Driveway": (
            "The New York City Police Department responded to the complaint and their "
            "investigation determined that police action was not necessary. If the problem "
            "persists, please contact 311 to create another complaint."
        ),
        "Heat/Hot Water": (
            "This complaint is a duplicate of a building-wide condition already reported by "
            "another tenant. The original complaint is still open, and HPD may only need to "
            "confirm that the condition exists by inspecting one apartment."
        ),
        "Illegal Parking": (
            "The New York City Police Department responded to the complaint and their "
            "investigation determined that no criminal violation existed. The condition was "
            "corrected without the need to issue a summons or effect an arrest."
        ),
        "Noise - Residential": (
            "The New York City Police Department responded to the complaint and their "
            "investigation determined that no criminal violation existed. The condition was "
            "corrected without the need to issue a summons or effect an arrest. "
            "If the problem persists, please contact 311 to create another complaint."
        ),
        "Snow or Ice": (
            "Your report was submitted and will be used to monitor snow conditions around "
            "the City. The Department of Sanitation has a winter storm operation currently "
            "underway and cannot respond to individual requests at this time."
        ),
    }

    if 'model4_text_input' not in st.session_state:
        st.session_state['model4_text_input'] = ''

    st.markdown("**Load a sample resolution description:**")
    _s4cols = st.columns(len(_M4_SAMPLES))
    for col, (lbl, txt) in zip(_s4cols, _M4_SAMPLES.items()):
        if col.button(lbl, use_container_width=True, key=f"m4_sample_{lbl}"):
            st.session_state['model4_text_input'] = txt
            st.rerun()

    st.markdown("---")
    user_text = st.text_area(
        "Resolution description text:",
        height=160,
        key='model4_text_input',
        placeholder="Paste a 311 resolution description here…",
    )

    if st.button("Classify", key="m4_classify") and user_text.strip():
        import torch

        inputs = _m4_tokenizer(
            [user_text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(_m4_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _m4_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = _m4_id2label[pred_idx]

        st.success(f"Predicted Category: **{label}**")
        st.write(f"Confidence: {confidence:.2%}")
        st.markdown("**All class probabilities:**")
        class_indices = sorted(_m4_id2label.keys())
        prob_df = pd.DataFrame({
            "Category": [_m4_id2label[i] for i in class_indices],
            "Probability": [float(probs[i]) for i in class_indices],
        }).sort_values("Probability", ascending=False)
        prob_df["Probability"] = prob_df["Probability"].map(lambda p: f"{p:.2%}")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

elif model_choice == "Model 5: Innovation":
    st.header("🛣️ Model 5: Innovation — Road Deterioration Prediction")
    st.write("Predict NYC 311 road complaint deterioration severity using XGBoost.")
    
    # ---- Load Model 5 (XGBoost Road Deterioration) ----
    @st.cache_resource
    def load_model5_xgb():
        base_path = Path(__file__).resolve().parent.parent / "models" / "model5_innovation" / "saved_model"
        
        model = joblib.load(str(base_path / "road_xgb_model.joblib"))
        scaler = joblib.load(str(base_path / "road_xgb_scaler.joblib"))
        
        with open(str(base_path / "road_xgb_features.json"), 'r') as f:
            features = json.load(f)
        
        return {"model": model, "scaler": scaler, "features": features}
    
    try:
        model5_bundle = load_model5_xgb()
        model = model5_bundle["model"]
        scaler = model5_bundle["scaler"]
        feature_names = model5_bundle["features"]
        model5_loaded = True
    except Exception as e:
        st.error(f"Failed to load Model 5: {e}")
        model5_loaded = False
    
    if not model5_loaded:
        st.stop()
    
    # ---- Constants ----
    ROAD_COMPLAINT_TYPES = {
        'Street Condition', 'Snow or Ice', 'Traffic Signal Condition',
        'Blocked Driveway', 'Street Light Condition',
        'Highway Sign - Damaged', 'Highway Sign - Missing',
        'Sidewalk Condition', 'Curb Condition', 'Pothole',
    }
    
    COMPLAINT_SEVERITY = {
        'Street Condition': 3, 'Pothole': 3,
        'Sidewalk Condition': 2, 'Curb Condition': 2,
        'Snow or Ice': 2, 'Traffic Signal Condition': 2,
        'Blocked Driveway': 1, 'Street Light Condition': 1,
        'Highway Sign - Damaged': 2, 'Highway Sign - Missing': 1,
    }
    
    HIGH_SEVERITY_DESCRIPTOR_KWS = [
        'POTHOLE', 'CAVE-IN', 'STRUCTURAL', 'BROKEN', 'FAILED',
        'COLLAPSED', 'SINK', 'ROUGH', 'PITTED', 'CRACK',
    ]
    
    LEVEL_NAMES = ['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Critical']
    LEVEL_COLORS = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}
    NUMERIC_FEATURES = [
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'severity_weight', 'resolution_hours',
    ]
    
    BOROUGHS = ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']
    STATUSES = ['Closed', 'Open', 'In Progress', 'Pending', 'Started']
    CHANNELS = ['ONLINE', 'MOBILE', 'PHONE', 'OTHER']
    DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def _map_descriptor(desc):
        """Map descriptor text to category."""
        if not isinstance(desc, str):
            return 'Other'
        d = desc.upper()
        street_keywords = ['STREET', 'SIDEWALK', 'CURB', 'POTHOLE', 'ASPHALT', 'PAVED']
        if any(kw in d for kw in street_keywords):
            return 'Street_Sidewalk'
        return 'Other'
    
    def predict_single(complaint_type, descriptor, borough, status, channel,
                       hour, day_of_week_int, month, resolution_hours):
        descriptor_cat = _map_descriptor(descriptor)
        severity_weight = COMPLAINT_SEVERITY.get(complaint_type, 1)
        
        # Build feature row
        row = {str(col): 0 for col in feature_names}
        row.update({
            'hour_of_day': hour, 'day_of_week': day_of_week_int,
            'month': month, 'is_weekend': int(day_of_week_int >= 5),
            'severity_weight': severity_weight, 'resolution_hours': resolution_hours,
        })
        
        # One-hot encoding
        for prefix, value in [('borough', borough), ('open_data_channel_type', channel),
                               ('status', status), ('descriptor_cat', descriptor_cat)]:
            col = f'{prefix}_{value}'
            if col in row:
                row[col] = 1
        
        # Create DataFrame and scale
        df_row = pd.DataFrame([row])[feature_names]
        num_cols = [c for c in NUMERIC_FEATURES if c in df_row.columns]
        df_row[num_cols] = scaler.transform(df_row[num_cols])
        
        # Predict
        pred = model.predict(df_row)[0]
        proba = model.predict_proba(df_row)[0]
        return int(pred), LEVEL_NAMES[int(pred)], proba
    
    # ---- Example presets ----
    EXAMPLES = {
        'Low':      {'complaint_type': 'Blocked Driveway', 'descriptor': 'DRIVEWAY', 'borough': 'BRONX', 'status': 'Closed', 'channel': 'ONLINE', 'hour': 9, 'day': 'Tuesday', 'month': 2, 'resolution_hours': 1.0},
        'Medium':   {'complaint_type': 'Street Light Condition', 'descriptor': 'STREET LIGHT', 'borough': 'QUEENS', 'status': 'Closed', 'channel': 'ONLINE', 'hour': 14, 'day': 'Wednesday', 'month': 2, 'resolution_hours': 6.0},
        'High':     {'complaint_type': 'Curb Condition', 'descriptor': 'BROKEN CURB', 'borough': 'MANHATTAN', 'status': 'Closed', 'channel': 'ONLINE', 'hour': 10, 'day': 'Tuesday', 'month': 3, 'resolution_hours': 24.0},
        'Critical': {'complaint_type': 'Street Condition', 'descriptor': 'POTHOLE', 'borough': 'BROOKLYN', 'status': 'Open', 'channel': 'ONLINE', 'hour': 10, 'day': 'Monday', 'month': 2, 'resolution_hours': 300.0},
    }
    
    # Initialize session state with defaults
    for k, v in EXAMPLES['Low'].items():
        if f'model5_{k}' not in st.session_state:
            st.session_state[f'model5_{k}'] = v
    
    # Load examples
    st.markdown('**Load example by severity level:**')
    ex_cols = st.columns(4)
    for col, (lvl, ex) in zip(ex_cols, EXAMPLES.items()):
        if col.button(f'📊 {lvl}', use_container_width=True, key=f'btn_m5_{lvl}'):
            for k, v in ex.items():
                st.session_state[f'model5_{k}'] = v
                st.session_state[f'model5_{k}_input'] = v  # keep widget keys in sync
            st.rerun()

    st.markdown('---')

    # Compute selectbox indices from session state so example buttons update dropdowns
    _complaint_opts = sorted(ROAD_COMPLAINT_TYPES)
    _ct_default      = st.session_state.get('model5_complaint_type', _complaint_opts[0])
    _borough_default = st.session_state.get('model5_borough', BOROUGHS[0])
    _status_default  = st.session_state.get('model5_status', STATUSES[0])
    _channel_default = st.session_state.get('model5_channel', CHANNELS[0])
    _day_default     = st.session_state.get('model5_day', DAY_NAMES[1])

    # Pre-initialize text_input key so value= and session_state don't conflict
    if 'model5_descriptor_input' not in st.session_state:
        st.session_state['model5_descriptor_input'] = st.session_state.get('model5_descriptor', 'DRIVEWAY')

    # Input form
    with st.form('model5_form'):
        col1, col2 = st.columns(2)
        with col1:
            input_complaint = st.selectbox(
                "Complaint Type",
                options=_complaint_opts,
                index=_complaint_opts.index(_ct_default) if _ct_default in _complaint_opts else 0,
                key='model5_complaint_type_input',
            )
            input_descriptor = st.text_input(
                "Descriptor (keyword)",
                key='model5_descriptor_input',
            )
            input_borough = st.selectbox(
                "Borough",
                options=BOROUGHS,
                index=BOROUGHS.index(_borough_default) if _borough_default in BOROUGHS else 0,
                key='model5_borough_input',
            )
            input_status = st.selectbox(
                "Status",
                options=STATUSES,
                index=STATUSES.index(_status_default) if _status_default in STATUSES else 0,
                key='model5_status_input',
            )

        with col2:
            input_channel = st.selectbox(
                "Channel",
                options=CHANNELS,
                index=CHANNELS.index(_channel_default) if _channel_default in CHANNELS else 0,
                key='model5_channel_input',
            )
            input_hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=st.session_state.get('model5_hour', 12),
                key='model5_hour_input',
            )
            input_day = st.selectbox(
                "Day of Week",
                options=DAY_NAMES,
                index=DAY_NAMES.index(_day_default) if _day_default in DAY_NAMES else 1,
                key='model5_day_input',
            )
            input_month = st.slider(
                "Month",
                min_value=1,
                max_value=12,
                value=st.session_state.get('model5_month', 6),
                key='model5_month_input',
            )
            input_resolution = st.number_input(
                "Resolution Hours",
                min_value=0.0,
                max_value=1000.0,
                value=float(st.session_state.get('model5_resolution_hours', 24.0)),
                step=1.0,
                key='model5_resolution_input',
            )
        
        submit = st.form_submit_button("🔍 Predict Deterioration Level", use_container_width=True)
        
        if submit:
            day_idx = DAY_NAMES.index(input_day)
            pred_level, pred_name, pred_proba = predict_single(
                complaint_type=input_complaint,
                descriptor=input_descriptor,
                borough=input_borough,
                status=input_status,
                channel=input_channel,
                hour=input_hour,
                day_of_week_int=day_idx,
                month=input_month,
                resolution_hours=input_resolution,
            )
            
            # Display results
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"**Predicted Level:** {pred_name}")
                st.write(f"**Confidence:** {pred_proba[pred_level]:.1%}")
            
            with col2:
                # Show all class probabilities
                st.write("**All Levels:**")
                for i, prob in enumerate(pred_proba):
                    level_name = LEVEL_NAMES[i].split()[1]
                    st.write(f"{LEVEL_NAMES[i]}: {prob:.1%}")

    st.markdown('---')
    st.subheader("Dataset Insights: NYC 311 Road Complaints")

    DATA_PATH_311 = ROOT_DIR / "data" / "raw" / "urbanpulse_311_complaints.csv"

    @st.cache_data
    def _compute_batch_preds(_model, _scaler, feat_names, data_path):
        from train import clean_data
        from predict import (
            filter_road_complaints,
            create_deterioration_label,
            build_road_features,
            NUMERIC_FEATURES as NF,
        )
        df = pd.read_csv(data_path)
        df_clean = clean_data(df)
        df_road = filter_road_complaints(df_clean)
        if df_road.empty:
            return pd.DataFrame()
        df_road = create_deterioration_label(df_road)
        X, _ = build_road_features(df_road)
        for col in feat_names:
            if col not in X.columns:
                X[col] = 0
        X = X[list(feat_names)].copy()
        num_cols = [c for c in NF if c in X.columns]
        X[num_cols] = _scaler.transform(X[num_cols])
        preds = _model.predict(X)
        proba = _model.predict_proba(X)
        df_road = df_road.reset_index(drop=True)
        level_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
        out = df_road[['complaint_type', 'borough']].copy()
        out['level_name'] = pd.Series(preds).map(level_map).values
        out['confidence'] = proba.max(axis=1).round(4)
        return out

    with st.expander("📊 Batch Predictions on Full Dataset", expanded=False):
        st.caption("Source: `data/raw/urbanpulse_311_complaints.csv`")
        with st.spinner("Processing dataset (first run may take a moment) …"):
            try:
                batch_df = _compute_batch_preds(
                    model, scaler, tuple(feature_names), str(DATA_PATH_311)
                )
                if batch_df.empty:
                    st.warning("No road-related complaints found in the dataset.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Road Complaints", f"{len(batch_df):,}")
                    c2.metric("Avg Confidence", f"{batch_df['confidence'].mean():.1%}")
                    c3.metric("Most Common Level", batch_df['level_name'].mode()[0])

                    st.write("**Prediction Distribution**")
                    dist = (
                        batch_df['level_name']
                        .value_counts()
                        .reindex(['Low', 'Medium', 'High', 'Critical'], fill_value=0)
                    )
                    st.bar_chart(dist)

                    st.write("**By Borough**")
                    borough_pivot = (
                        batch_df.groupby(['borough', 'level_name'])
                        .size()
                        .unstack(fill_value=0)
                        .reindex(columns=['Low', 'Medium', 'High', 'Critical'], fill_value=0)
                    )
                    st.dataframe(borough_pivot)

                    st.write("**By Complaint Type**")
                    type_pivot = (
                        batch_df.groupby(['complaint_type', 'level_name'])
                        .size()
                        .unstack(fill_value=0)
                        .reindex(columns=['Low', 'Medium', 'High', 'Critical'], fill_value=0)
                    )
                    st.dataframe(type_pivot)

                    st.write("**Sample Predictions (first 100 rows)**")
                    st.dataframe(batch_df.head(100))
            except Exception as e:
                st.error(f"Failed to run batch predictions: {e}")

    st.info("💡 Tip: Use the example buttons above to load pre-configured scenarios!")
