import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# --- LOAD SAVED ARTIFACTS ---
# Use st.cache_resource to load the model and preprocessing objects only once
@st.cache_resource
def load_model_and_preprocessors():
    """Loads the trained model and all preprocessing objects."""
    try:
        model = tf.keras.models.load_model('Ann_ChunPrediction.h5')
        
        with open('Encoder_Contract.pkl', 'rb') as file:
            encoder_contract = pickle.load(file)
            
        with open('Encoder_Internet.pkl', 'rb') as file:
            encoder_internet = pickle.load(file)
            
        with open('Label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)

        with open('Sclar.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return model, encoder_contract, encoder_internet, label_encoder, scaler
    except FileNotFoundError:
        st.error("Error: Model or preprocessor files not found. Please make sure the following files are in the same directory: 'Ann_ChunPrediction.h5', 'Encoder_Contract.pkl', 'Encoder_Internet.pkl', 'Label_encoder.pkl', 'Sclar.pkl'")
        return None, None, None, None, None

# Load all the necessary files
model, encoder_contract, encoder_internet, label_encoder, scaler = load_model_and_preprocessors()

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem 10rem;
    }
</style>
""", unsafe_allow_html=True)


st.title(' भविष्यवाणी 🔮')
st.markdown("This app predicts whether a customer is likely to churn based on their account information. Please provide the customer's details below.")


# --- USER INPUT SECTION ---

col1, col2 = st.columns(2)

with col1:
    st.header("Customer Details")
    tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100, value=1)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=30.0, format="%.2f")
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

with col2:
    st.header("Services Information")
    partner = st.selectbox('Partner', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])


# --- PREDICTION LOGIC ---
if st.button('Predict Churn'):
    if model is None:
        st.warning("Prediction cannot proceed as essential files are missing.")
    else:
        # 1. Create a DataFrame from user input
        input_data = pd.DataFrame({
            'Partner': [partner],
            'tenure': [tenure],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'Contract': [contract],
            'TotalCharges': [total_charges]
        })
        
        # --- PREPROCESSING PIPELINE ---
        
        # 2. Handle 'No internet service' and 'No phone service'
        cols_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV']
        for col in cols_to_replace:
            input_data[col] = input_data[col].replace('No internet service', 'No')
        input_data['MultipleLines'] = input_data['MultipleLines'].replace('No phone service', 'No')

        # 3. Apply Label Encoding to binary columns
        binary_cols = ['Partner', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV']
        for col in binary_cols:
            # Use a try-except block to handle unseen labels gracefully
            try:
                input_data[col] = label_encoder.transform(input_data[col])
            except ValueError:
                # If a value was never seen during training, you might default it
                # For this dataset, 'Yes'/'No' covers all cases after replacement
                st.warning(f"Unexpected value in '{col}'. Check your input.")
                # Fallback to a default (e.g., 0 for 'No') if needed
                input_data[col] = 0

        # 4. Apply One-Hot Encoding for Contract and InternetService
        # Internet Service
        internet_encoded = encoder_internet.transform(input_data[['InternetService']])
        internet_df = pd.DataFrame(internet_encoded, columns=encoder_internet.get_feature_names_out(['InternetService']))
        
        # Contract
        contract_encoded = encoder_contract.transform(input_data[['Contract']])
        contract_df = pd.DataFrame(contract_encoded, columns=encoder_contract.get_feature_names_out(['Contract']))
        
        # 5. Combine all data and drop original categorical columns
        processed_data = pd.concat([input_data.drop(columns=['InternetService', 'Contract']), internet_df, contract_df], axis=1)

        # 6. Ensure column order matches the model's training data
        final_column_order = [
            'Partner', 'tenure', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'TotalCharges',
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year'
        ]
        processed_data = processed_data[final_column_order]

        # 7. Scale the final data
        scaled_data = scaler.transform(processed_data)

        # 8. Make Prediction
        prediction_proba = model.predict(scaled_data)[0][0]
        prediction = 1 if prediction_proba > 0.5 else 0

        # --- DISPLAY RESULT ---
        st.write("---")
        st.header('Prediction Result')
        if prediction == 1:
            st.error(f'**Churn: Yes** (Prediction Probability: {prediction_proba:.2%})')
            st.warning('This customer is at a high risk of churning.')
        else:
            st.success(f'**Churn: No** (Prediction Probability: {prediction_proba:.2%})')
            st.balloons()
            st.info('This customer is likely to stay.')