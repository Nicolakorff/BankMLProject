import json
import pickle
from pathlib import Path

import streamlit as st

from bank_utils import MONTH_ORDER, predict_customer

MODELS = Path(__file__).parent / "models"


@st.cache_resource
def load_artifacts():
    def load(name):
        with open(MODELS / name, "rb") as f:
            return pickle.load(f)
    with open(MODELS / "cluster_profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)
    return (load("preprocessor.pkl"), load("segmenter.pkl"), load("kmeans_model.pkl"),
            load("logistic_model.pkl"), profiles)


prep, segmenter, kmeans_model, logistic_model, profiles = load_artifacts()

MONTHS_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
             "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
JOBS_ES = {"Administrativo/a": "admin.", "Obrero/a": "blue-collar", "Emprendedor/a": "entrepreneur",
           "Empleado/a del hogar": "housemaid", "Dirección/gestión": "management", "Jubilado/a": "retired",
           "Autónomo/a": "self-employed", "Servicios": "services", "Estudiante": "student",
           "Técnico/a": "technician", "Desempleado/a": "unemployed", "Desconocido": "unknown"}
MARITAL_ES = {"Casado/a": "married", "Soltero/a": "single", "Divorciado/a": "divorced"}
EDUCATION_ES = {"Primaria": "primary", "Secundaria": "secondary", "Terciaria": "tertiary", "Desconocida": "unknown"}
YES_NO = {"Sí": "yes", "No": "no"}

st.title("Predicción del grupo y probabilidad de adquirir depósitos")
st.caption("Introduce los datos del cliente. Los mismos 9 datos que usa el modelo.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Edad (años)", min_value=18, max_value=100, value=40)
    balance = st.number_input("Balance (euros)", min_value=-5000.0, max_value=100000.0, value=1000.0, step=100.0)
    campaign = st.number_input("Número de contactos en esta campaña", min_value=1, max_value=60, value=2)
    month = st.selectbox("Mes del contacto", MONTHS_ES, index=4)
with col2:
    job = st.selectbox("Profesión", list(JOBS_ES))
    marital = st.selectbox("Estado civil", list(MARITAL_ES))
    education = st.selectbox("Nivel de estudios", list(EDUCATION_ES), index=1)
    housing = st.radio("¿Tiene hipoteca?", list(YES_NO), index=1, horizontal=True)
    loan = st.radio("¿Tiene préstamo personal?", list(YES_NO), index=1, horizontal=True)

customer = {
    "age": age, "job": JOBS_ES[job], "marital": MARITAL_ES[marital],
    "education": EDUCATION_ES[education], "balance": balance,
    "housing": YES_NO[housing], "loan": YES_NO[loan],
    "month": MONTH_ORDER[MONTHS_ES.index(month)], "campaign": campaign,
}

try:
    cluster, probability = predict_customer(customer, prep, segmenter, kmeans_model, logistic_model)
    st.subheader("Resultado")
    st.metric("Probabilidad de adquirir un depósito", f"{probability:.0%}")
    st.write(f"El cliente pertenece al clúster: **{cluster}**")
    p = profiles[str(cluster)]
    st.markdown(
        f"**Perfil del clúster {cluster}** ({int(p['n_clients']):,} clientes de entrenamiento)\n"
        f"- Edad media: {p['edat_mitjana']:.0f} años · saldo medio: {p['saldo_mitja_€']:,.0f} €\n"
        f"- Casados: {p['% casats']:.0f}% · solteros: {p['% solters']:.0f}% · divorciados: {p['% divorciats']:.0f}%\n"
        f"- Con hipoteca: {p['% hipoteca']:.0f}% · con préstamo personal: {p['% préstec']:.0f}%\n"
        f"- Conversión histórica del clúster (test): {p['conversió test %']:.0f}%"
    )
    st.caption("Los clústers se definen sobre todo por el estado civil y el préstamo personal. "
               "La probabilidad es orientativa (AUC del modelo ≈ 0,68): sirve para priorizar contactos, "
               "no para decidir de forma automática.")
except Exception as e:  # noqa: BLE001
    st.error(f"No se ha podido calcular la predicción: {e}")
