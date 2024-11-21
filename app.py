import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo y el escalador desde archivos
with open('kmeans_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('logisitc_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Título de la aplicación
st.title('Predicción del grupo-target más propenso a adquirir depósitos de venta anual')

# Entrada de datos del usuario
month_encoded = st.number_input('Momento del contacto de campaña (meses)', min_value=0)
balance = st.number_input('Sueldo (euros)', min_value=0)
age = st.number_input('Edad (años)', min_value=0)

# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
    'Momento del contacto de campaña': [month_encoded],
    'Sueldo ': [balance],
    'Edadh': [age]
})

# Estandarizar las entradas
user_data_standardized = scaler.transform(user_data)

# Realizar la predicción
prediction = model.predict(user_data_standardized)

# Mostrar la predicción
st.write(f'Predicción del grupo-target más propenso a adquirir depósitos de venta anual': {prediction[0]:.2f}')
