import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

print("Loading data...")

X = np.load("X.npy")
y = np.load("y.npy")

# convertir les mots en nombres
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# sauvegarder les classes pour le test temps réel
classes = encoder.classes_
np.save("classes.npy", classes)


print(f"Data loaded. X shape: {X.shape}")

# séparation des données
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

input_shape = (X.shape[1], X.shape[2])

# modèle GRU
model = models.Sequential([
    layers.Input(shape=input_shape),

    layers.BatchNormalization(),

    layers.GRU(128, return_sequences=False),

    layers.Dropout(0.3),

    layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Starting training...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# évaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# sauvegarde du modèle
model.save("speech_model.h5")
print("Model saved as speech_model.h5")
