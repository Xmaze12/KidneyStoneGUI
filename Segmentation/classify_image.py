# classify_image.py
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def classify_image(model_path, image_path):
    """
    Loads a trained Keras model, preprocesses an image, and returns the
    classification as a string ('Stone' or 'No Stone').
    """
    try:
        # Load the trained model
        model = load_model(model_path)

        # Define class names
        class_names = ['No Stone', 'Stone']

        # Load and preprocess the image for ResNet50
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)

        # Make prediction
        prediction = model.predict(img_preprocessed, verbose=0)

        # The output is a probability. Threshold at 0.5.
        if prediction[0][0] >= 0.5:
            return class_names[1]  # Stone
        else:
            return class_names[0]  # No Stone

    except Exception as e:
        # Print error to be captured by MATLAB if something goes wrong
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Ensure all required file paths are provided
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py <path_to_image>")
        sys.exit(1)

    model_file = 'best_model.keras'
    image_file = sys.argv[1]

    if not os.path.exists(model_file):
        print(f"Error: Model file not found at '{model_file}'")
        sys.exit(1)
        
    if not os.path.exists(image_file):
        print(f"Error: Image file not found at '{image_file}'")
        sys.exit(1)

    # Get the classification result and print it to standard output
    result = classify_image(model_file, image_file)
    print(result)