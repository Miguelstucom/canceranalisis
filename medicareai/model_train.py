import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

print("\n=== Cargando Datos ===")
df_historial = pd.read_csv("medicareai/static/csv/historial_medico.csv")
print("\nHistorial Médico:")
print(df_historial.head())

df_sangre = pd.read_csv("medicareai/static/csv/Analisis_sangre_dataset.csv")
print("\nAnálisis de Sangre:")
print(df_sangre.head())

df_cancer = pd.read_csv("medicareai/static/csv/Analisis_cancer.csv")
print("\nAnálisis de Cáncer:")
print(df_cancer.head())

df_total = df_historial.merge(df_sangre, on="id", how="inner") \
                       .merge(df_cancer, on="id", how="inner")

print(f"\n✅ Datos combinados correctamente: {df_total.shape[0]} pacientes")

if 'Survival_Prediction' not in df_total.columns:
    print("\n❌ ERROR: No se encuentra la columna 'Survival_Prediction'.")
    exit()

# Separar X e y
X = df_total.drop(columns=["id", "Survival_Prediction"])
y = df_total["Survival_Prediction"]

# Convertir etiquetas 'Yes'/'No' o similares a valores numéricos
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("\nEtiquetas convertidas a numéricas:", list(le.classes_))

# Codificar variables categóricas
X = pd.get_dummies(X)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Definir modelos
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "Red Neuronal": MLPClassifier(max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Función para entrenar y evaluar modelos
def evaluar_modelo(nombre, modelo):
    print(f"\n🔍 === Entrenando modelo: {nombre} ===")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"📊 Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("📉 Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

    return modelo

# Entrenar y evaluar todos los modelos
modelos_entrenados = {}
for nombre, modelo in modelos.items():
    modelos_entrenados[nombre] = evaluar_modelo(nombre, modelo)

# Opcional: Guardar el modelo con mejor desempeño (por ejemplo, Random Forest)
modelo_final = modelos_entrenados["Random Forest"]
joblib.dump({
    'model': modelo_final,
    'feature_names': X.columns.tolist()
}, "modelo_diagnostico_final.pkl")

print("\n✅ Modelo final guardado como 'modelo_diagnostico_final.pkl'")
