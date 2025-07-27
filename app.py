# Py file to build a web app for churn prediction using Streamlit

# imports
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pickle

# MODEL & PREPROCESSING SETUP

# load the trained ANN model
model = tf.keras.models.load_model(r"models\churn_model.h5")

# load the encoders and scaler
with open(r"models\gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)
    gender_encoder = gender_encoder.get("Gender") ## file was initially stored as a dict

# load the geography encoder
with open(r"models\geo_encoder.pkl", "rb") as file:
    geo_encoder = pickle.load(file)

# load the scaler
with open(r"models\scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# STREAMLIT APP SETUP

# main heading
st.title("Customer CHurn Prediction")

# getting user input
geography = st.selectbox("Geography", options=geo_encoder.categories_[0])
gender = st.selectbox("Gender", options=gender_encoder.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=1000.0)
credit_score = st.number_input("Credit Score", min_value=50, max_value=1000, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=10000000.0, value=50000.0)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=2)
number_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_credit_card = st.selectbox("Has Credit Card", options=["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", options=["Yes", "No"])



# encode geography input
geo_encoded = np.array(geo_encoder.transform([[geography]]))
# create a dataframe for the encoded geography
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(["Geography"]))

# create dataframe from user input
input_data = pd.DataFrame(
    {
        "CreditScore":[credit_score],
        "Gender":[gender_encoder.transform([[gender]])[0]],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[number_of_products],
        "HasCrCard":[1 if has_credit_card == "Yes" else 0],
        "IsActiveMember":[1 if is_active_member == "Yes" else 0],
        "EstimatedSalary":[estimated_salary],
    }
)

# concatenate the input data with the encoded geography
input_data_final = pd.concat([input_data, geo_encoded_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data_final)

# make prediction
prediction = model.predict(input_data_scaled)

# churn probability = prediction[0][0]

# add a button to make prediction
if st.button("Predict Churn"):
    # display the prediction
    if prediction[0][0] > 0.5:
        st.write(f"The churn probability is {prediction[0][0]:.2f}.")
        st.write("The customer is likely to churn.")
    else:
        st.write(f"The churn probability is {prediction[0][0]:.2f}.")
        st.write("The customer is likely to stay.")

