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

# Título de la aplicación
st.title('Predicción del grupo y probabilidad de adquirir depósitos')

# Entrada de datos del usuario
month_encoded = st.number_input('Momento del contacto de campaña (meses)', min_value=0, max_value=11, step=1)
age = st.number_input('Edad (años)', min_value=18, max_value=100, step=1)
balance = st.number_input('Balance (euros)', min_value=-5000.0, max_value=100000.0, step=100.0)
campaign = st.number_input('Número de campañas en las que ha sido contactado', min_value=1, max_value=50, step=1)

# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
    'month': [month_encoded],  # Ajusta los nombres de columnas según tu modelo
    'age': [age],
    'balance': [balance],
    'campaign': [campaign]
})

try:
    # Estandarizar las entradas (excepto 'month')
    user_data_to_scale = user_data.drop(columns=['month_encoded'])  # Excluir 'month' de la estandarización
    user_data_standardized = scaler.transform(user_data_to_scale)

    # Combinar 'month' con los datos estandarizados
    user_data_combined = pd.DataFrame(user_data_standardized, columns=user_data_to_scale.columns)
    user_data_combined['month_encoded'] = user_data['month_encoded'].values  # Añadir de nuevo la columna 'month'

    # Filtrar las características utilizadas por KMeans
    user_data_kmeans = user_data_combined[['age', 'balance']]  # Cambia según las columnas utilizadas en KMeans

    # Predicción del clúster con K-means
    cluster_prediction = kmeans_model.predict(user_data_kmeans)[0]

    # Predicción de la probabilidad con el modelo de regresión logística
    probability_prediction = logistic_model.predict_proba(user_data_combined)[0][1]

    # Mostrar las predicciones
    st.write(f"El usuario pertenece al clúster: **{cluster_prediction}**")
    st.write(f"Probabilidad de adquirir un depósito: **{probability_prediction:.2f}**")

except Exception as e:
    st.error(f"Ha ocurrido un error al realizar la predicción: {e}")
