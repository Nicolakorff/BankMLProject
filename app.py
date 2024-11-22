try:
    # Estandarizar las entradas (excepto 'month')
    user_data_to_scale = user_data.drop(columns=['month'])  # Excluir 'month' de la estandarización
    user_data_standardized = scaler.transform(user_data_to_scale)

    # Combinar 'month' con los datos estandarizados
    user_data_combined = pd.DataFrame(user_data_standardized, columns=user_data_to_scale.columns)
    user_data_combined['month'] = user_data['month'].values  # Añadir de nuevo la columna 'month'

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

