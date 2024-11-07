import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('F_animal_classification_model.keras')

# Define class names
class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer", 
    "Dog", "Dolphin", "Elephant", "Giraffe", 
    "Horse", "Kangaroo", "Lion", "Panda", 
    "Tiger", "Zebra"
]

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess and predict an image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_index, confidence

# Route for the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Predict using the model
    predicted_class_index, confidence = predict_image(file_path)
    predicted_class_name = class_names[predicted_class_index]

    # Remove the saved file after prediction
    os.remove(file_path)

    # Return result
    return jsonify({
        'animal': predicted_class_name,
        'confidence': f'{confidence * 100:.2f}'
    })

if __name__ == '__main__':
    app.run(debug=True)
