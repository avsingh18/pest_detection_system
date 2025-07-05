import PIL
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None

def load_model_and_classes():
    try:
        # Check if model exists
        if not os.path.exists('output/best_model.h5'):
            raise FileNotFoundError("Model file 'output/best_model.h5' not found. Please run training first.")
        
        # Check if class names exist
        if not os.path.exists('output/class_names.json'):
            raise FileNotFoundError("Class names file 'output/class_names.json' not found. Please run training first.")
        
        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model('output/best_model.h5')
        
        # Load class names
        print("Loading class names...")
        with open('output/class_names.json', 'r') as f:
            class_names = json.load(f)
        
        return model, class_names
    
    except Exception as e:
        print(f"Error loading model and classes: {str(e)}")
        raise

def predict_image(image_path, model, class_names):
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        
        # Load and preprocess image
        print("\nProcessing image...")
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(img_array, verbose=0)  # Set verbose=0 to reduce output
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"Error predicting image: {str(e)}")
        raise

def main():
    try:
        print("\n=== Plant Disease Detection System ===")
        print("Loading the model and resources...")
        
        # Load model and class names
        model, class_names = load_model_and_classes()
        
        while True:
            # Get image path from user
            print("\nEnter the path to your image file (or 'quit' to exit):")
            image_path = input("> ").strip()
            
            # Check if user wants to quit
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Plant Disease Detection System!")
                break
            
            # Make prediction
            try:
                predicted_class, confidence = predict_image(image_path, model, class_names)
                
                # Print results with formatting
                print("\n=== Prediction Results ===")
                print("┌" + "─" * 50 + "┐")
                print("│ Predicted Disease/Condition:".ljust(51) + "│")
                print("│ " + predicted_class.replace('_', ' ').ljust(49) + "│")
                print("│" + "─" * 50 + "│")
                print("│ Confidence Level:".ljust(51) + "│")
                print("│ {:.2%}".format(confidence).ljust(49) + "│")
                print("└" + "─" * 50 + "┘")
                
                # Provide interpretation
                if confidence >= 0.90:
                    print("\nInterpretation: Very high confidence in prediction")
                elif confidence >= 0.70:
                    print("\nInterpretation: Good confidence in prediction")
                else:
                    print("\nInterpretation: Prediction made with lower confidence,")
                    print("consider taking another photo with better lighting/angle")
                
            except Exception as e:
                print(f"\nError processing image: {str(e)}")
                print("Please make sure the image file exists and is a valid image format")
                print("Supported formats: JPG, JPEG, PNG, WEBP")
    
    except Exception as e:
        print(f"\nSystem Error: {str(e)}")
        print("Please make sure the model is trained and all files are in place")

if __name__ == "__main__":
    main()