import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo y el escalador desde archivos
with open('kmeans_model.pkl', 'rb') as model_file:
    kmodel = pickle.load(model_file)

with open('logistic_model.pkl', 'rb') as model_file:
    lmodel = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Título de la aplicación
st.title('Predicción del grupo y probabilidad de adquirir depósitos')

# Entrada de datos del usuario
balance = st.number_input('Sueldo (euros)', min_value=0)
age = st.number_input('Edad (años)', min_value=0)

# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
    'balance ': [balance],
    'age': [age]
})

# Estandarizar las entradas
user_data_standardized = scaler.transform(user_data)

# Predicción del clúster con K-means
cluster_prediction = kmodel.predict(user_data)[0]

# Predicción de la probabilidad con el modelo de regresión logística
probability_prediction = lmodel.predict_proba(user_data)[0][1]

# Mostrar las predicciones
st.write(f"El usuario pertenece al clúster: **{cluster_prediction}**")
st.write(f"Probabilidad de adquirir un depósito: **{probability_prediction:.2f}**")
