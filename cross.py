import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Path to your classes folder
data_dir = 'dataset'  # Update this to your classes folder

# Parameters
img_height, img_width = 128, 128
batch_size = 128  # Reduced batch size for better memory management
epochs = 15
n_splits = 5  # Number of folds for cross-validation

# Load images and labels
def load_images(data_dir):
    images = []
    labels = []
    class_names = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            class_names.append(class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(class_dir, filename)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(len(class_names) - 1)  # Assign label based on index of class_names

    return np.array(images), np.array(labels), class_names

# Load data
X, y, class_names = load_images(data_dir)
print(f'Loaded {len(X)} images and {len(y)} labels.')
print(f'Class names: {class_names}')

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=30,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# KFold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results
all_val_losses = []
all_val_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f'Fold {fold + 1}/{n_splits}')

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    validation_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size)

    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False  # Freeze base model layers

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout layer
    predictions = Dense(len(class_names), activation='softmax')(x)

    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(f'best_model_fold_{fold + 1}.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')

    # Store results
    all_val_losses.append(val_loss)
    all_val_accuracies.append(val_accuracy)

# Average results across all folds
print('Cross-Validation Results:')
print(f'Average Validation Loss: {np.mean(all_val_losses)}')
print(f'Average Validation Accuracy: {np.mean(all_val_accuracies)}')
