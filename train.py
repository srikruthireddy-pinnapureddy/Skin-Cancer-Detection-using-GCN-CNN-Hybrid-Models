import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

import warnings
warnings.filterwarnings("ignore")#supress warnings for cleaner output

# Load metadata
meta_data_path = "C:\\Users\\srikr\\OneDrive\\Desktop\\SkinCancer\\HAM10000_metadata.csv"
meta_data = pd.read_csv(meta_data_path)

# Handle categorical data to numerical form
encoder = LabelEncoder()
meta_data["dx_label"] = encoder.fit_transform(meta_data["dx"])

# Set paths
images_dir = "C:\\Users\\srikr\\OneDrive\\Desktop\\SkinCancer\\Dataset"
train_images_dir = "C:\\Users\\srikr\\OneDrive\\Desktop\\SkinCancer\\train\\"
validation_images_dir = "C:\\Users\\srikr\\OneDrive\\Desktop\\SkinCancer\\validation\\"

# Ensure directories exist, no error
def create_dirs(dir_path, dir_names):
    for dir_name in dir_names:
        os.makedirs(os.path.join(dir_path, str(dir_name)), exist_ok=True)

# Create training directories
dir_names = encoder.transform(encoder.classes_)  # Encoded class labels(types of skin cancer are mapped to unique integer)
create_dirs(train_images_dir, dir_names)

# Assign images to the appropriate folder
for image in os.scandir(images_dir):
    try:
        img_name = image.name.split(".")[0]
        # Match the image name with metadata
        img_cancer_type = str(meta_data.dx_label[meta_data.image_id == img_name].item())
        dest_path = os.path.join(train_images_dir, img_cancer_type, image.name)
        shutil.copy(image.path, dest_path)
    except Exception as e:
        print(f"Error processing {image.name}: {e}")

# Check processed images
print("\nImage assignment completed.\n")
for sub_dir in os.scandir(train_images_dir):
    print(f"Class {sub_dir.name}: {len(list(os.scandir(sub_dir)))} images.")

# Calculate 5% of each type for validation
five_percent_content = {}
for sub_dir in os.scandir(train_images_dir):
    total_count = len(list(os.scandir(sub_dir)))
    five_percent_content[sub_dir.name] = round(total_count * 0.05)

# Create validation directories
create_dirs(validation_images_dir, dir_names)

# Move 5% of images to validation directories
for sub_dir in os.scandir(train_images_dir):
    images_paths = [image.path for image in os.scandir(sub_dir)]
    for image_path in images_paths[:five_percent_content[sub_dir.name]]:
        shutil.move(image_path, os.path.join(validation_images_dir, sub_dir.name, os.path.split(image_path)[-1]))

print("\nValidation set creation completed.\n")

# Check the counts in validation directories
for sub_dir in os.scandir(validation_images_dir):
    print(f"Class {sub_dir.name}: {len(list(os.scandir(sub_dir)))} images.")

# Set image size and batch size
img_size = 250
batch_size = 32

# Data augmentation
generator = ImageDataGenerator(
    zoom_range=0.3,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)

augmented_train_data = generator.flow_from_directory(
    train_images_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset="training"
)

unaugmented_test_data = generator.flow_from_directory(
    train_images_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset="validation"
)

unaugmented_dev_data = image_dataset_from_directory(
    validation_images_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

# Model architecture
model = Sequential([
    BatchNormalization(input_shape=(img_size, img_size, 3)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    augmented_train_data,
    validation_data=unaugmented_test_data,
    epochs=15,
    verbose=1
)

# Save the trained model
model_dir = "C:\\Users\\srikr\\OneDrive\\Desktop\\SkinCancer\\"
model.save(os.path.join(model_dir, 'skincancer_model.h5'))
print("\nModel saved successfully.")

# Load the model
loaded_model = tf.keras.models.load_model(os.path.join(model_dir, 'skincancer_model.h5'))

# Extract training metrics
metrics = history.history
train_loss = metrics["loss"]
train_accuracy = metrics["accuracy"]
test_loss = metrics["val_loss"]
test_accuracy = metrics["val_accuracy"]

# Visualize training results
plt.figure(figsize=(13, 4))
plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(train_loss, label="Train")
plt.plot(test_loss, label="Test")
plt.grid(True)
plt.legend(loc="best")

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(train_accuracy, label="Train")
plt.plot(test_accuracy, label="Test")
plt.grid(True)
plt.legend(loc="best")
plt.show()
"""
# Test prediction on a new image
new_image_path = "D:\\project\\SKIN_CANCER_DETECTION\\test_image.jpg"
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (img_size, img_size))
new_image = np.expand_dims(new_image, axis=0)
new_image = new_image / 255.0

prediction = loaded_model.predict(new_image)
predicted_class = np.argmax(prediction)

# Decode predicted class
predicted_class_label = encoder.inverse_transform([predicted_class])[0]
print(f"\nThe predicted class label for the new image is: {predicted_class_label}")
"""