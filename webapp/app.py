"""
Capstone Web Application
========================
Integrates all 5 models into a single web interface using Streamlit.

Run locally:  streamlit run webapp/app.py
Deploy:       Push to GitHub, then connect to Streamlit Community Cloud
              https://streamlit.io/cloud (free hosting)
"""
import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Capstone Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("AI Capstone Dashboard")
st.write("Select a model from the sidebar to make predictions.")

# Sidebar navigation
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

# ---------------------------------------------------------------------------
# Helper: Cache model loading so it only happens once
# ---------------------------------------------------------------------------
# Use @st.cache_resource for models — they load once and stay in memory.
#
# Example:
#     @st.cache_resource
#     def load_model1():
#         import joblib
#         return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
#
#     @st.cache_resource
#     def load_model3():
#         import tensorflow as tf
#         return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")

# ---------------------------------------------------------------------------
# Model pages — fill these in with your model loading and prediction logic
# ---------------------------------------------------------------------------

if model_choice == "Home":
    st.write("Welcome! Use the sidebar to navigate between models.")
    st.write("Each model page lets you input data and see predictions in real time.")

elif model_choice == "Model 1: Traditional ML":
    st.header("Model 1: Traditional ML")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    # @st.cache_resource
    # def load_model1():
    #     import joblib
    #     return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
    #
    # model = load_model1()
    #
    # # Create input fields for your features
    # col1, col2 = st.columns(2)
    # with col1:
    #     feature_1 = st.number_input("Feature 1", value=0.0)
    #     feature_2 = st.selectbox("Feature 2", ["Option A", "Option B"])
    # with col2:
    #     feature_3 = st.slider("Feature 3", 0, 100, 50)
    #
    # if st.button("Predict"):
    #     import pandas as pd
    #     input_df = pd.DataFrame([{"feature_1": feature_1, ...}])
    #     prediction = model.predict(input_df)
    #     probability = model.predict_proba(input_df)
    #     st.success(f"Prediction: {prediction[0]}")
    #     st.write(f"Confidence: {probability.max():.2%}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — load your model and add input fields here.")

elif model_choice == "Model 2: Deep Learning":
    st.header("Model 2: Deep Learning")
    # TODO: Load your DNN model and add prediction interface
    # Same pattern as Model 1, but load with:
    #     import tensorflow as tf
    #     model = tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")
    st.info("Not yet implemented — load your model and add input fields here.")

elif model_choice == "Model 3: CNN (Image Classification)":
    st.header("Model 3: CNN — Image Classification")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    @st.cache_resource
    def load_model3():
        import tensorflow as tf
        return tf.keras.models.load_model("models/model3_cnn/saved_model/efficientnet_model.keras")
    
    model = load_model3()
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        from PIL import Image
        import numpy as np
    
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
        # Preprocess — must match your training preprocessing
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
    
        if st.button("Classify"):
            prediction = model.predict(img_batch)
            print(prediction)
            confidence = float(prediction.max())
            predicted_class = "Positive" if prediction[0][0] > 0.5 else "Negative"
            st.success(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    # st.info("Not yet implemented — add image upload and classification here.")

elif model_choice == "Model 4: NLP (Text Classification)":
    st.header("Model 4: NLP — Text Classification")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    # @st.cache_resource
    # def load_model4():
    #     import joblib
    #     model = joblib.load("models/model4_nlp_classification/saved_model/model.joblib")
    #     vectorizer = joblib.load("models/model4_nlp_classification/saved_model/vectorizer.joblib")
    #     return model, vectorizer
    #
    # model, vectorizer = load_model4()
    #
    # user_text = st.text_area("Enter text to classify:", height=150)
    # if st.button("Classify") and user_text:
    #     text_vectorized = vectorizer.transform([user_text])
    #     prediction = model.predict(text_vectorized)[0]
    #     confidence = model.predict_proba(text_vectorized).max()
    #     st.success(f"Predicted Category: {prediction}")
    #     st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — add text input and classification here.")

elif model_choice == "Model 5: Innovation":
    st.header("Model 5: Innovation")
    # TODO: Add your custom model interface
    st.info("Not yet implemented — add your innovation model interface here.")
