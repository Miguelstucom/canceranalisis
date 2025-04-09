import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
import seaborn as sns

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

# Select only the specified variables
selected_vars = ['Age', 'tumor_size', 'relapse', 'Family history', 'inflammatory_bowel_disease', 'cancer_stage']
X = df_total[selected_vars].copy()
y = df_total["Survival_Prediction"]

# Convert Yes/No columns to 1/0
boolean_columns = ['relapse', 'Family history', 'inflammatory_bowel_disease']
for col in boolean_columns:
    if col in X.columns:
        # Convert to boolean (1/0)
        X[col] = X[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0})
        # Ensure numeric type
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

# Convertir etiquetas 'Yes'/'No' o similares a valores numéricos
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("\nEtiquetas convertidas a numéricas:", list(le.classes_))

# Create correlation matrix
print("\n=== Matriz de Correlación ===")
# Combine X and y for correlation analysis
df_corr = X.copy()
df_corr['Survival_Prediction'] = y

# Calculate correlation matrix
correlation_matrix = df_corr.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('medicareai/static/metrics/correlation_matrix.png')
plt.close()

# Print correlation with Survival_Prediction
print("\nCorrelación con Survival_Prediction:")
survival_corr = correlation_matrix['Survival_Prediction'].sort_values(ascending=False)
print(survival_corr.to_string())

# No need for dummy variables anymore
# X = pd.get_dummies(X)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define models and their parameter grids
modelos = {
    "Random Forest": {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },

    "Red Neuronal": {
        'model': MLPClassifier(random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [500]
        }
    },
    "XGBoost": {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3, 5]
        }
    }
}

# Train and evaluate each model
results = {}
for name, model_info in modelos.items():
    print(f"\n=== Entrenando {name} ===")
    
    # Grid search for best parameters
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    print(f"\nMejores parámetros para {name}:")
    print(grid_search.best_params_)
    
    print(f"\nMétricas para {name}:")
    print(classification_report(y_test, y_pred))
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(f'medicareai/static/metrics/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Save the best model (Random Forest)
best_rf_model = grid_search.best_estimator_  # Get the best model from grid search
joblib.dump({
    'model': best_rf_model,
    'feature_names': X.columns.tolist(),
    'feature_importances': best_rf_model.feature_importances_
}, 'modelo_diagnostico_final.pkl')

print("\n=== Importancia de Variables ===")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.to_string())

print("\n=== Resultados Finales ===")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
