from django.shortcuts import render
from django.conf import settings
import os
from .ml_model import ColonCancerModel
import tensorflow as tf
import base64

model = None

def load_model():
    global model
    if model is None:
        print("Initializing new model...")
        model = ColonCancerModel()
        model_path = os.path.join(settings.BASE_DIR, 'medicareai', 'trained_model.keras')
        if os.path.exists(model_path):
            print("Loading existing model...")
            model.build_model()
            model.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        else:
            print("Training new model...")
            data_dir = os.path.join(settings.BASE_DIR, 'data')
            print(f"Looking for data in: {data_dir}")
            model.train_model(data_dir)
            print("Saving model...")
            model.model.save(model_path)
            print("Model saved successfully")

def upload_image(request):
    prediction = None
    confidence = None
    error_message = None
    image_data = None

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Save the uploaded image temporarily
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', image_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # Load the model and make prediction
            load_model()
            prediction, confidence = model.predict_image(temp_path)

            # Convert image to base64 for display
            with open(temp_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Clean up the temporary file
            os.remove(temp_path)

        except Exception as e:
            error_message = f"An error occurred during analysis. Please try again."
            print(f"Error in upload_image: {str(e)}")

    return render(request, 'medicareai/upload.html', {
        'prediction': prediction,
        'confidence': confidence,
        'error_message': error_message,
        'image_data': image_data
    })
