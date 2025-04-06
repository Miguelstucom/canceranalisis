import matplotlib
matplotlib.use('Agg')  # Backend sin GUI

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
import pathlib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils import class_weight

class ColonCancerModel:
    def __init__(self):
        self.model = None
        self.img_height = 180
        self.img_width = 180
        # Load class names from file
        self.class_names = self.load_class_names()

    def load_class_names(self):
        try:
            # Get the directory of the current file (ml_model.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            class_names_path = os.path.join(current_dir, 'class_names.txt')
            
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            return class_names
        except Exception as e:
            print(f"Error loading class names: {e}")
            # Return default class names if file cannot be read
            return ['imagenesColonBenigno', 'ImagenesColonCancerigeno']

    def build_model(self):
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),
        ])

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Transfer learning inicial

        self.model = models.Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def fine_tune_model(self, learning_rate=1e-5):
        self.model.layers[2].trainable = True  # base_model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def train_model(self, data_dir):
        data_dir = pathlib.Path(data_dir)

        subdirs = [f.name for f in data_dir.iterdir() if f.is_dir() and f.name not in ['plots', 'metrics']]
        if len(subdirs) != 2:
            raise ValueError(f"Expected 2 directories, found {len(subdirs)}: {subdirs}")
        subdirs.sort()

        for class_name in subdirs:
            path = os.path.join(data_dir, class_name)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
            print(f"{class_name}: {count} imágenes")

        self.class_names = subdirs

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=32,
            class_names=subdirs
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=32,
            class_names=subdirs
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        label_counts = Counter()
        for _, labels in val_ds.unbatch():
            label_counts[int(labels.numpy())] += 1
        print("Distribución validation:", label_counts)

        all_labels = []
        for _, labels in train_ds.unbatch():
            all_labels.append(labels.numpy())

        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        class_weights = {i: w for i, w in enumerate(weights)}
        print("Pesos de clase:", class_weights)

        self.build_model()

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            class_weight=class_weights
        )

        print("Iniciando fine-tuning...")

        self.fine_tune_model()
        history_fine = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            class_weight=class_weights
        )

        # Métricas
        y_true, y_pred = [], []
        for images, labels in val_ds:
            predictions = self.model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        metrics_dir = os.path.join(os.path.dirname(data_dir), 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'))
        plt.close()

        # Gráfico de entrenamiento
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'] + history_fine.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'training_history.png'))
        plt.close()

    def predict_image(self, image_path, cancer_threshold=0.5):
        # Ensure class names are loaded
        if not self.class_names:
            self.class_names = self.load_class_names()
            
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = predictions[0]

        try:
            if score[1] > cancer_threshold:
                return self.class_names[1], 100 * score[1]
            else:
                return self.class_names[0], 100 * score[0]
        except IndexError:
            print("Warning: Prediction failed. Using default class names.")
            # Return a default prediction if something goes wrong
            return 'imagenesColonBenigno', 100 * score[0]
