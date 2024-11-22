import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo y el escalador desde archivos
with open('kmeans_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Título de la aplicación
st.title('Predicción del grupo y probabilidad de adquirir depósitos')

# Entrada de datos del usuario
month_encoded = st.number_input('Momento del contacto de campaña (meses)', min_value=0)
balance = st.number_input('Sueldo (euros)', min_value=0)
age = st.number_input('Edad (años)', min_value=0)

# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
    'Momento del contacto de campaña': [month_encoded],
    'Sueldo ': [balance],
    'Edad': [age]
})

# Estandarizar las entradas (excepto 'month')
user_data_to_scale = user_data.drop(columns=['month'])  # Excluir 'month' de la estandarización
user_data_standardized = scaler.transform(user_data_to_scale)

# Combinar 'month' con los datos estandarizados
user_data_combined = pd.DataFrame(user_data_standardized, columns=user_data_to_scale.columns)
user_data_combined['month'] = user_data['month'].values  # Añadir de nuevo la columna 'month'

# Predicción del clúster con K-means
cluster_prediction = kmeans_model.predict(user_data_combined)[0]

# Predicción de la probabilidad con el modelo de regresión logística
probability_prediction = logistic_model.predict_proba(user_data_combined)[0][1]

# Mostrar las predicciones
st.write(f"El usuario pertenece al clúster: **{cluster_prediction}**")
st.write(f"Probabilidad de adquirir un depósito: **{probability_prediction:.2f}**")
