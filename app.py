import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo K-means, regresión logística y escalador desde archivos
with open('kmeans_model.pkl', 'rb') as kmeans_file:
    kmeans_model = pickle.load(kmeans_file)

with open('logistic_model.pkl', 'rb') as logistic_file:
    logistic_model = pickle.load(logistic_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title('Predicción del grupo y probabilidad de adquirir depósitos')

month_encoded = st.number_input('Momento del contacto de campaña (meses)', min_value=1, max_value=12, step=1)
age = st.number_input('Edad (años)', min_value=18, max_value=100, step=1)
balance = st.number_input('Balance (euros)', min_value=-5000.0, max_value=100000.0, step=100.0)
campaign = st.number_input('Número de campañas en que ha habido ocntacto', min_value=1, max_value=50, step=1)

user_data = pd.DataFrame({
    'month': [month_encoded],
    'age': [age],
    'balance': [balance],
    'campaign': [campaign]
})

user_data_kmeans = user_data_combined[['age', 'balance']] 

cluster_prediction = kmeans_model.predict(user_data_kmeans)[0]

try:
    user_data_to_scale = user_data.drop(columns=['month'])  
    user_data_standardized = scaler.transform(user_data_to_scale)

    user_data_combined = pd.DataFrame(user_data_standardized, columns=user_data_to_scale.columns)
    user_data_combined['month'] = user_data['month'].values 

    user_data_kmeans = user_data_combined[['age', 'balance']] 

    cluster_prediction = kmeans_model.predict(user_data_kmeans)[0]

    probability_prediction = logistic_model.predict_proba(user_data_combined)[0][1]

    st.write(f"El usuario pertenece al clúster: **{cluster_prediction}**")
    st.write(f"Probabilidad de adquirir un depósito: **{probability_prediction:.2f}**")

except Exception as e:
    st.error(f"Ha ocurrido un error al realizar la predicción: {e}")

