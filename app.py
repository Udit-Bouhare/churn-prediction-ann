import streamlit as st 
import numpy as np 
import torch 
import pandas as pd 
import pickle
from model import ANN

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer will **leave the bank or not**")

# -------------------- LOAD MODEL --------------------
checkpoint = torch.load("model.pth")
model = ANN(num_features=checkpoint["num_features"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------------------- LOAD PREPROCESSORS --------------------
with open('label_encoder_gender.pkl','rb') as file: 
    label_encoder_gender = pickle.load(file) 

with open('onehot_encoder_geo.pkl','rb') as file: 
    onehot_encoder_geo = pickle.load(file) 

with open('scaler.pkl', 'rb') as file: 
    scaler = pickle.load(file) 


# -------------------- INPUT UI --------------------
st.subheader("🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_) 
    age = st.slider('🎂 Age',18,92) 
    tenure = st.slider('📆 Tenure (years)',0,10)

with col2:
    balance = st.number_input('💰 Balance', min_value=0.0)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0)
    num_of_products = st.slider('📦 Number of Products',1,4)

has_cr_card = st.selectbox('💳 Has Credit Card', [0,1]) 
is_active_member = st.selectbox('⚡ Is Active Member', [0,1])



# -------------------- PREDICT BUTTON --------------------
if st.button("🚀 Predict"):

    # -------------------- DATA PREPARATION --------------------
    input_data = pd.DataFrame({
        'CreditScore' : [credit_score], 
        'Gender' : [label_encoder_gender.transform([gender])[0]], 
        'Age' : [age], 
        'Tenure' : [tenure], 
        'Balance' : [balance], 
        'NumOfProducts' : [num_of_products], 
        'HasCrCard' : [has_cr_card], 
        'IsActiveMember' : [is_active_member], 
        'EstimatedSalary' : [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine all features
    input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # -------------------- FIX: COLUMN ORDER --------------------
    input_df = input_df[scaler.feature_names_in_]   # ✅ CRITICAL FIX

    # Scale
    input_scaled = scaler.transform(input_df)

    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # -------------------- PREDICTION --------------------
    with torch.no_grad():
        prob = model(input_tensor)
    
    prob_value = prob.item()

    # -------------------- RESULT UI --------------------
    st.subheader("📈 Prediction Result")

    st.metric(label="Churn Probability",value=f"{prob_value*100:.2f}%")

    if prob_value > 0.5:
        st.error("⚠️ High Risk: Customer is likely to churn!")
    else:
        st.success("✅ Safe: Customer is not likely to churn")