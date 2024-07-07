# Importing necessary libraries
# type: ignore
import numpy as np
import os
import mediapipe as mp
from utilities import get_unique_filename
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical 
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


# Initializing MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize the parameters
sequence_length = 30  # Number of frames in each sequence
DATA_PATH = './Sign_Language_Dataset'  # Path to save the keypoints data
os.listdir(DATA_PATH)
signs = np.array([i for i in os.listdir(DATA_PATH) if i[0] != '.']) # dynamic way to get the classes

# Preprocessing the data
label_map = {label:num for num, label in enumerate(signs)}

# Initializing the lists to store the sequences and labels
sequences, labels = [], []
for action in tqdm(signs, desc="Loading in the landmarks data....", ncols=100):
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
Y = to_categorical(labels).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=48)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='train_accuracy', patience=34, verbose=1, mode='max', restore_best_weights=True)

# Defining the LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))  
model.add(LSTM(64, return_sequences=False, activation='tanh')) # Last layer does not return sequences
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test), callbacks=[tb_callback, early_stopping_callback])


# Saving the model based on the timestamp
model_filename = get_unique_filename('saved_model/sign_language')
model.save(model_filename)
print("Model saved successfully at: ", model_filename)
