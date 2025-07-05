import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import json

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load and prepare dataset
def load_dataset():
    try:
        print("Loading PlantVillage dataset...")
        dataset, info = tfds.load('plant_village', with_info=True, as_supervised=True)
        train_ds = dataset['train']
        
        # Get class information
        num_classes = info.features['label'].num_classes
        class_names = info.features['label'].names
        
        # Save class names to file
        with open('output/class_names.json', 'w') as f:
            json.dump(class_names, f)
        
        print(f"Dataset loaded successfully with {num_classes} classes")
        return train_ds, num_classes, class_names
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def prepare_dataset(dataset):
    try:
        print("Preparing dataset...")
        
        def preprocess(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0
            return image, label
        
        # Apply preprocessing
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        
        # Split dataset
        total_size = tf.data.experimental.cardinality(dataset).numpy()
        train_size = int(0.8 * total_size)
        
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)
        
        # Batch datasets
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        print("Dataset preparation completed")
        return train_ds, val_ds
    
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise

def create_model(num_classes):
    try:
        print("Creating model...")
        model = models.Sequential([
            # Base
            layers.Conv2D(32, 3, padding='same', activation='relu', 
                         input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        print("Model created successfully")
        return model
    
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def train_model(model, train_ds, val_ds):
    try:
        print("Starting model training...")
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'output/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3
            )
        ]
        
        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        
        # Save final model
        model.save('output/final_model.h5')
        print("Model training completed")
        return history
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def plot_training_history(history):
    try:
        print("Plotting training history...")
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('output/training_history.png')
        print("Training history plot saved")
    
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

def main():
    try:
        # Load and prepare dataset
        train_ds, num_classes, class_names = load_dataset()
        train_ds, val_ds = prepare_dataset(train_ds)
        
        # Create and train model
        model = create_model(num_classes)
        history = train_model(model, train_ds, val_ds)
        
        # Plot training history
        plot_training_history(history)
        
        print("Training completed successfully!")
        print("Model saved as 'output/final_model.h5'")
        print("Best model saved as 'output/best_model.h5'")
        print("Class names saved as 'output/class_names.json'")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()