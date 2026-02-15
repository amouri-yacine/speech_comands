import os

import librosa
import numpy as np

DATA_PATH = r"C:\Users\yacin\OneDrive\Bureau\DEEP_LEARNING\speech_comands"   # dossier principal des audios
SAMPLES_TO_CONSIDER = 16000    # 1 seconde à 16 kHz

mfccs = []
labels = []

# parcourir chaque dossier (chaque mot)
for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(label_path):
        continue

    # parcourir chaque fichier audio
    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            file_path = os.path.join(label_path, file)

            signal, sr = librosa.load(file_path, sr=16000)

            # garder seulement 1 seconde
            if len(signal) >= SAMPLES_TO_CONSIDER:
                signal = signal[:SAMPLES_TO_CONSIDER]

                # extraction MFCC
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

                mfccs.append(mfcc.T)   # temps en premier
                labels.append(label)

# conversion en tableau numpy
X = np.array(mfccs)
y = np.array(labels)

print("Dataset shapes:")
print("X:", X.shape)
print("y:", y.shape)
print("y:", y.shape)
# sauvegarder le dataset
np.save("X.npy", X)
np.save("y.npy", y)

print("Files saved: X.npy and y.npy")

