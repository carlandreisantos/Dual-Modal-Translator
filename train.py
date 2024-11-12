import os
import cv2
import numpy as np
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.callbacks import TensorBoard
from keras import utils, models, layers, callbacks
# 1. Setup Paths and Action Labels
DATA_PATH = 'Training Data'  # Path where your training data is stored
actions = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]  # Get folder names as actions

# 2. Preprocess Data and Create Labels and Features
label_map = {label: num for num, label in enumerate(actions)}  # Mapping of action names to labels
sequences, labels = [], []
sequence_length = 30  # The length of each sequence (video frame count)

for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    for sequence in np.array(os.listdir(action_folder)).astype(int):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(action_folder, str(sequence), "{}.npy".format(frame_num))
            if os.path.exists(file_path):
                res = np.load(file_path)  # Load the keypoints
                window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)  # Feature data
y = utils.to_categorical(labels).astype(int)  # One-hot encoded labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3. Build and Train LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = callbacks.TensorBoard(log_dir=log_dir)

model = models.Sequential()
model.add(layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))  # 30 frames, 1662 keypoints per frame
model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(actions), activation='softmax'))  # Output layer for the number of actions
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Train the Model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

# 5. Save the Trained Model
model.save('model.keras')
