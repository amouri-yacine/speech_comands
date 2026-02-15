import time

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

CONFIDENCE_THRESHOLD = 0.80
SILENCE_THRESHOLD = 0.015
COOLDOWN_SECONDS = 0.8
NEW_SENTENCE_DELAY = 3.0

SAMPLE_RATE = 16000
BUFFER_SIZE = 16000
CHUNK_SIZE = 4000

print("Loading system...", end="")
model = tf.keras.models.load_model("speech_model.h5")
class_names = np.load("classes.npy")
print("\rSystem Ready. Speak naturally. (Ctrl+C to stop)\n")

def preprocess_live_input(audio_buffer):
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=SAMPLE_RATE, n_mfcc=13).T

    if mfcc.shape[0] < 32:
        pad_width = 32 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:32, :]

    return np.expand_dims(mfcc, axis=0)

audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
last_speech_time = time.time()
new_sentence_ready = True

try:
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE) as stream:

        print("Listening: ", end="", flush=True)

        while True:
            new_data, _ = stream.read(CHUNK_SIZE)
            new_data = new_data.flatten()

            audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = new_data

            current_time = time.time()

            if (current_time - last_speech_time) > NEW_SENTENCE_DELAY:
                if not new_sentence_ready:
                    print(".", end="\nListening: ", flush=True)
                    new_sentence_ready = True

            # silence
            if np.max(np.abs(new_data)) < SILENCE_THRESHOLD:
                continue

            input_tensor = preprocess_live_input(audio_buffer)
            predictions = model.predict(input_tensor, verbose=0)

            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                word = class_names[predicted_index]

                if word != "_background_noise_":
                    if new_sentence_ready:
                        word = word.capitalize()
                        new_sentence_ready = False

                    print(f"{word}", end=" ", flush=True)

                    last_speech_time = time.time()

                    # reset buffer
                    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

                    time.sleep(COOLDOWN_SECONDS)
                    stream.read(stream.read_available)

except KeyboardInterrupt:
    print("\n\nStopped.")
