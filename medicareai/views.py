from django.shortcuts import render
from django.conf import settings
import os
import pandas as pd
import joblib
from .ml_model import ColonCancerModel
import tensorflow as tf
import base64
from sklearn.preprocessing import LabelEncoder

model = None
rf_model = None

def load_models():
    global model, rf_model
    # Load CNN model for images
    if model is None:
        print("Initializing CNN model...")
        model = ColonCancerModel()
        model_path = os.path.join(settings.BASE_DIR, 'medicareai', 'trained_model.keras')
        if os.path.exists(model_path):
            print("Loading existing CNN model...")
            model.build_model()
            model.model = tf.keras.models.load_model(model_path)
    
    # Load Random Forest model for CSV data
    if rf_model is None:
        rf_model_path = os.path.join(settings.BASE_DIR, 'modelo_diagnostico_final.pkl')
        print(f"Looking for Random Forest model at: {rf_model_path}")
        if os.path.exists(rf_model_path):
            print("Loading Random Forest model...")
            model_data = joblib.load(rf_model_path)
            if isinstance(model_data, dict):
                rf_model = model_data['model']
                # Store feature names in a separate attribute if the model doesn't support feature_names_in_
                if not hasattr(rf_model, 'feature_names_in_'):
                    rf_model._feature_names = model_data['feature_names']
            else:
                rf_model = model_data  # For backward compatibility
        else:
            print(f"❌ Random Forest model not found at {rf_model_path}")
            raise FileNotFoundError(f"Random Forest model not found at {rf_model_path}")

def process_csv_files(historial_file, sangre_file, cancer_file):
    # Read CSV files
    df_historial = pd.read_csv(historial_file)
    df_sangre = pd.read_csv(sangre_file)
    df_cancer = pd.read_csv(cancer_file)
    
    print("\n=== Datos Cargados ===")
    print("\nHistorial Médico:")
    print(df_historial)
    
    # Remove Survival_Prediction from historial_medico if it exists
    if 'Survival_Prediction' in df_historial.columns:
        df_historial = df_historial.drop(columns=['Survival_Prediction'])
        print("\nColumna 'Survival_Prediction' eliminada del historial médico")
    
    print("\nAnálisis de Sangre:")
    print(df_sangre)
    print("\nAnálisis de Cáncer:")
    print(df_cancer)
    
    # Merge all dataframes
    df_total = df_historial \
        .merge(df_sangre, on="id", how="inner") \
        .merge(df_cancer, on="id", how="inner")
    
    print("\n=== Datos Combinados ===")
    print(df_total)
    
    if len(df_total) != 1:
        raise ValueError(f"Expected 1 patient, found {len(df_total)} after merging CSVs")
    
    # Select only the specified variables
    selected_vars = [
        'Age', 'tumor_size', 'relapse', 'Family history', 'inflammatory_bowel_disease', 'cancer_stage', 'obesity'
    ]
    
    # Check if all required variables are present
    missing_vars = [var for var in selected_vars if var not in df_total.columns]
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    X = df_total[selected_vars].copy()
    
    # Convert categorical variables to numeric with specific mappings
    categorical_mappings = {
        'Sexo': {'F': 0, 'M': 1},
        'Family history': {'No': 0, 'Yes': 1},
        'smoke': {'No': 0, 'Yes': 1},
        'alcohol': {'No': 0, 'Yes': 1},
        'obesity': {'Normal': 0, 'Overweight': 1, 'Obese': 2},
        'diet': {'Low': 0, 'Moderate': 1, 'High': 2},
        'Screening_History': {'Never': 0, 'Irregular': 1, 'Regular': 2},
        'Healthcare_Access': {'Low': 0, 'Moderate': 1, 'High': 2},
        'inflammatory_bowel_disease': {'No': 0, 'Yes': 1},
        'relapse': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)
            print(f"\nMapping for {col}: {mapping}")
    
    # Ensure all numeric columns are properly typed
    numeric_columns = ['Age', 'Hemoglobina', 'Plaquetas', 'Globulos blancos', 
                      'Glucosa', 'HDL', 'tumor_size']
    
    for col in numeric_columns:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print("\n=== Datos para Predicción (después de conversión) ===")
    print(X)

    # Get expected columns from the model
    expected_columns = getattr(rf_model, 'feature_names_in_', getattr(rf_model, '_feature_names', None))
    if expected_columns is not None:
        print("\n=== Columnas Esperadas por el Modelo ===")
        print(expected_columns)
        
        # Add missing columns with zeros
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
                print(f"Agregando columna faltante: {col}")
        
        # Reorder columns to match training data
        X = X[expected_columns]
    
    print("\n=== Datos Finales para Predicción ===")
    print(X)

    # Show feature importances
    if hasattr(rf_model, 'feature_importances_'):
        print("\n=== Importancia de Variables en la Predicción ===")
        feature_importance = pd.DataFrame({
            'feature': expected_columns if expected_columns is not None else X.columns,
            'importance': rf_model.feature_importances_,
            'value': X.iloc[0].values  # Current values for prediction
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nImportancia de cada variable:")
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f} (Valor actual: {row['value']})")

    return X

def upload_image(request):
    prediction = None
    confidence = None
    survival_pred = None
    survival_prob = None
    error_message = None
    image_data = None
    combined_prediction = None
    combined_confidence = None

    if request.method == 'POST':
        try:
            # Process image if provided
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', image_file.name)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                with open(temp_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)

                # Load models and make image prediction
                load_models()
                prediction, confidence = model.predict_image(temp_path)

                # Convert image to base64 for display
                with open(temp_path, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')

                os.remove(temp_path)

            # Process CSV files if all are provided
            required_files = ['historial_medico', 'analisis_sangre', 'analisis_cancer']
            if all(file in request.FILES for file in required_files):
                # Load Random Forest model
                load_models()
                
                if rf_model is None:
                    raise ValueError("Random Forest model not found")

                # Process CSV files
                X = process_csv_files(
                    request.FILES['historial_medico'],
                    request.FILES['analisis_sangre'],
                    request.FILES['analisis_cancer']
                )

                # Make prediction
                survival_pred = rf_model.predict(X)[0]
                probs = rf_model.predict_proba(X)[0]
                survival_prob = probs[1] * 100 if survival_pred == 1 else probs[0] * 100
                
                # Convert numeric prediction to text
                survival_pred = "Benign" if survival_pred == 0 else "Malignant"
                
                print("\n=== Predicción de Supervivencia ===")
                print(f"Predicción: {survival_pred}")
                print(f"Probabilidad: {survival_prob:.2f}%")
                print(f"Probabilidades completas: {probs}")

            # Combine predictions if both are available
            if prediction and survival_pred:
                # Get weights from form, defaulting to 0.7 and 0.3 if not provided
                image_weight = float(request.POST.get('image_weight', 70)) / 100
                csv_weight = float(request.POST.get('csv_weight', 30)) / 100
                print(image_weight)
                print(csv_weight)
                
                # Ensure weights sum to 1
                total_weight = image_weight + csv_weight
                if total_weight != 1.0:
                    image_weight = image_weight / total_weight
                    csv_weight = csv_weight / total_weight
                
                # Convert predictions to numeric values (0 for benign, 1 for malignant)
                image_pred_numeric = 0 if prediction == 'imagenesColonBenigno' else 1
                csv_pred_numeric = 0 if survival_pred == 'Benign' else 1
                
                # Calculate weighted average using form weights
                combined_pred_numeric = (image_pred_numeric * image_weight) + (csv_pred_numeric * csv_weight)
                
                # Convert back to text prediction
                combined_prediction = "Benign" if combined_pred_numeric < 0.5 else "Malignant"
                
                # Calculate combined confidence
                image_confidence = confidence / 100  # Convert to decimal
                csv_confidence = survival_prob / 100  # Convert to decimal
                combined_confidence = ((image_confidence * image_weight) + (csv_confidence * csv_weight)) * 100

        except Exception as e:
            error_message = f"An error occurred during analysis: {str(e)}"
            print(f"Error in upload_image: {str(e)}")

    return render(request, 'medicareai/upload.html', {
        'prediction': prediction,
        'confidence': confidence,
        'survival_pred': survival_pred,
        'survival_prob': survival_prob,
        'error_message': error_message,
        'image_data': image_data,
        'combined_prediction': combined_prediction,
        'combined_confidence': combined_confidence
    })
