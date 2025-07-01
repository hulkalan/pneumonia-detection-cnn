import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ---------------------------------------
# Step 1: Set paths and image dimensions
# ---------------------------------------

base_dir = "chest_xray"  # CHANGE this path
img_height, img_width = 150, 150
batch_size = 32

# ---------------------------------------
# Step 2: Preprocess the images
# ---------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

train_data = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_data = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_data = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# ---------------------------------------
# Step 3: Build the CNN model
# ---------------------------------------

model = Sequential([
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1), activation='relu'))
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

# ---------------------------------------
# Step 4: Compile the model
# ---------------------------------------

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------------------------------------
# Step 5: Train the model
# ---------------------------------------

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# ---------------------------------------
# Step 6: Evaluate the model
# ---------------------------------------

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.4f}")

# ---------------------------------------
# Step 7: Plot Accuracy and Loss
# ---------------------------------------

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------
# Step 8: Save the model
# ---------------------------------------

model.save("pneumonia_cnn_model.h5")
print("Model saved as pneumonia_cnn_model.h5")
