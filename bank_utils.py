"""Utilitats compartides entre el notebook 05 i l'app Streamlit.

Aquí viuen les transformacions de dades perquè el notebook (entrenament) i
l'app (predicció) facin EXACTAMENT el mateix preprocessat.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

MONTH_ORDER = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
EDUCATION_MAP = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
JOB_GROUPS = {
    'admin.': 'qualified', 'blue-collar': 'qualified', 'technician': 'qualified',
    'management': 'qualified', 'retired': 'qualified',
    'services': 'semi-qualified', 'housemaid': 'semi-qualified',
    'entrepreneur': 'freelance', 'self-employed': 'freelance',
    'student': 'no qualified', 'unemployed': 'no qualified',
    'unknown': 'other',
}
JOB_GROUP_CODE = {'freelance': 1, 'qualified': 2, 'semi-qualified': 3,
                  'no qualified': 4, 'other': 5}
MARITAL_CATEGORIES = ['divorced', 'married', 'single', 'unknown']
NUMERIC_COLS = ['age', 'balance', 'campaign']
FEATURE_COLUMNS = ['age', 'balance', 'campaign', 'education_encoded',
                   'month_encoded', 'job_grouped_encoded', 'loan', 'housing',
                   'marital_divorced', 'marital_married', 'marital_single',
                   'marital_unknown']
# Variables de perfil de client per a la segmentació (sense mes ni campanya:
# descriuen el contacte, no el client).
SEGMENT_COLUMNS = ['age', 'balance', 'education_encoded', 'housing', 'loan',
                   'marital_divorced', 'marital_married', 'marital_single']
RAW_COLUMNS = ['age', 'job', 'marital', 'education', 'balance',
               'housing', 'loan', 'month', 'campaign']


def encode_features(df):
    """Codifica les variables categòriques (sense ajustar res a les dades)."""
    out = pd.DataFrame(index=df.index)
    out['age'] = df['age'].astype(int)
    out['balance'] = df['balance'].astype(float)
    out['campaign'] = df['campaign'].astype(int)
    out['education_encoded'] = df['education'].map(EDUCATION_MAP)
    out['month_encoded'] = df['month'].map({m: i for i, m in enumerate(MONTH_ORDER)})
    out['job_grouped_encoded'] = df['job'].map(JOB_GROUPS).map(JOB_GROUP_CODE)
    out['loan'] = (df['loan'] == 'yes').astype(int)
    out['housing'] = (df['housing'] == 'yes').astype(int)
    for cat in MARITAL_CATEGORIES:
        out[f'marital_{cat}'] = (df['marital'] == cat).astype(float)
    out = out[FEATURE_COLUMNS]
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Valors desconeguts a les columnes: {bad}")
    return out


def fit_preprocessor(encoded_train):
    """Aprèn els paràmetres NOMÉS amb el conjunt d'entrenament:
    límits IQR per winsoritzar 'balance' i RobustScaler per a age/balance/campaign."""
    q1, q3 = encoded_train['balance'].quantile([0.25, 0.75])
    iqr = q3 - q1
    limits = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    clipped = encoded_train.copy()
    clipped['balance'] = clipped['balance'].clip(*limits)
    scaler = RobustScaler().fit(clipped[NUMERIC_COLS])
    return {'balance_limits': limits, 'scaler': scaler}


def apply_preprocessor(encoded, prep):
    """Aplica (sense reajustar) els paràmetres apresos a train."""
    out = encoded.copy()
    out['balance'] = out['balance'].clip(*prep['balance_limits'])
    out[NUMERIC_COLS] = prep['scaler'].transform(out[NUMERIC_COLS])
    return out[FEATURE_COLUMNS]


def fit_segmenter(preprocessed_train):
    """StandardScaler per a les variables de segmentació (només train), perquè
    totes pesin igual en la distància euclidiana de K-means."""
    return StandardScaler().fit(preprocessed_train[SEGMENT_COLUMNS])


def segment_features(preprocessed, segmenter):
    """Matriu de variables de perfil, estandarditzada, per a K-means."""
    return pd.DataFrame(segmenter.transform(preprocessed[SEGMENT_COLUMNS]),
                        index=preprocessed.index, columns=SEGMENT_COLUMNS)


def predict_customer(raw_row, prep, segmenter, kmeans, logistic):
    """Retorna (clúster, probabilitat de contractar el dipòsit) per a un client.
    raw_row: dict amb les claus de RAW_COLUMNS."""
    df = pd.DataFrame([raw_row])[RAW_COLUMNS]
    X = apply_preprocessor(encode_features(df), prep)
    cluster = int(kmeans.predict(segment_features(X, segmenter))[0])
    proba = float(logistic.predict_proba(X)[0, 1])
    return cluster, proba
