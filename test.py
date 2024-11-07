import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Parameters
img_height, img_width = 128, 128

# Load test images and labels
def load_test_images(test_dir, class_names):
    test_images = []
    test_labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(class_dir, filename)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
                    test_images.append(img_array)
                    test_labels.append(class_names.index(class_name))  # Assign label based on class name index

    return np.array(test_images), np.array(test_labels)

# Path to your saved model and test dataset
model_path = 'F_animal_classification_model.keras'  # Update this to your saved model path
test_data_dir = 'dataset'  # Update this to your test data folder

# Load class names (these should match your training classes)
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 
               'Dolphin', 'Elephant', 'Giraffe', 'Horse', 
               'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Load test data
X_test, y_test = load_test_images(test_data_dir, class_names)
print(f'Loaded {len(X_test)} test images and {len(y_test)} labels.')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save confusion matrix as image

# Create a figure for the overall report
fig, ax = plt.subplots(figsize=(12, 8))

# Add the stats and plot to the image
ax.axis('off')
text = f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy*100:.2f}%\n\nClassification Report:\n{report}"
ax.text(0.1, 0.5, text, fontsize=12, wrap=True)

# Save overall report image
plt.savefig('model_report.png', bbox_inches='tight')

# Optionally, you can show the confusion matrix plot
plt.show()
