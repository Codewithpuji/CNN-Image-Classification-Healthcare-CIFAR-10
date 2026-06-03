# ============================================================
# CIFAR-10 Image Classification using CNN
# MSc Data Analytics — Poojitha Kalyanam (2024)
# University for the Creative Arts, Germany
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow version:", tf.__version__)

# ── 1. Load & Preprocess Data ────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Normalise pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten label arrays
y_train, y_test = y_train.flatten(), y_test.flatten()

# CIFAR-10 class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

K = len(labels)  # 10 classes
print(f"Number of classes: {K}")

# ── 2. Visualise Sample Images ───────────────────────────────
fig, ax = plt.subplots(5, 5, figsize=(8, 8))
k = 0
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k])
        ax[i][j].set_title(labels[y_train[k]], fontsize=8)
        ax[i][j].axis('off')
        k += 1
plt.suptitle('Sample Training Images', fontsize=12)
plt.tight_layout()
plt.show()

# ── 3. Build CNN Model (Functional API) ──────────────────────
i = Input(shape=x_train[0].shape)  # (32, 32, 3)

# Block 1
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Block 2
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Block 3
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Classifier head
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)  # 10-class output

model = Model(i, x)
model.summary()

# ── 4. Compile & Train (Phase 1 — No Augmentation) ──────────
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nPhase 1: Training without augmentation (50 epochs)...")
r1 = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50
)

# ── 5. Continue Training with Data Augmentation ──────────────
print("\nPhase 2: Fine-tuning with data augmentation (10 epochs)...")
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = data_generator.flow(x_train, y_train, batch_size=batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r2 = model.fit(
    train_generator,
    validation_data=(x_test, y_test),
    steps_per_epoch=steps_per_epoch,
    epochs=10
)

# ── 6. Plot Training Accuracy ────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(r2.history['accuracy'], label='Training Accuracy', color='red')
plt.plot(r2.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Model Accuracy (Augmentation Phase)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ── 7. Sample Prediction ─────────────────────────────────────
image_number = 0
plt.imshow(x_test[image_number])
plt.title('Test Sample')
plt.axis('off')
plt.show()

n = np.array(x_test[image_number]).reshape(1, 32, 32, 3)
predicted_label = labels[model.predict(n).argmax()]
original_label  = labels[y_test[image_number]]
print(f"Original: {original_label} | Predicted: {predicted_label}")

# ── 8. Full Evaluation ───────────────────────────────────────
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=labels))

# ── 9. Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix — CIFAR-10')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred_classes, labels)
