import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model = tf.keras.models.load_model('F_animal_classification_model.keras')

# List of class names
class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer", 
    "Dog", "Dolphin", "Elephant", "Giraffe", 
    "Horse", "Kangaroo", "Lion", "Panda", 
    "Tiger", "Zebra"
]

# Function to preprocess an image and make predictions
def predict_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(128, 128))  # Use the same size as your training
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the predicted class
    confidence = predictions[0][predicted_class_index]  # Get the confidence of the predicted class

    return predicted_class_index, confidence, predictions

# Example usage
if __name__ == "__main__":
    # Path to the image you want to predict
    img_path = 'test_data/mk.jpg'  # Replace with your image path

    predicted_class_index, confidence, predictions = predict_image(img_path)
    predicted_class_name = class_names[predicted_class_index]  # Map index to class name

    # Print the predicted class and confidence percentage
    print(f'Animal is: {predicted_class_name}')
    print(f'Confidence: {confidence * 100:.2f}%')  # Print confidence as a percentage
    print(f'Predictions: {predictions}')
