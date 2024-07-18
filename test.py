# Importing all the necessary libraries
import cv2
import os
import argparse
import numpy as np
import mediapipe as mp
from collections import Counter
from utilities import landmarks_detection, draw_styled_landmarks, extract_keypoints, load_most_recent_model
from tensorflow.keras.models import load_model

# Setting up argument parser
parser = argparse.ArgumentParser(description='Realtime LSTM Sign Language Detection')
parser.add_argument('--model', type=str, default=None, help='Specific model name to load')

# Parsing arguments
args = parser.parse_args()

# Intializing detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.8
smoothed_probabilities = None  # To store the smoothed probabilities
alpha = 0.2  # Smoothing factor

# Intializing the signs and the path to the saved keypoints data
DATA_PATH = './Sign_Language_Dataset'  
os.listdir(DATA_PATH)
signs = np.array([i for i in os.listdir(DATA_PATH) if i[0] != '.']) # dynamic way to get the classes

# Model loading logic
model_directory = 'saved_model'
if args.model:
    # If a specific model name is provided then load that model
    model_path = f"{model_directory}/{args.model}.keras"
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
else:
    # If no specific model name is provided, then load the most recent model
    model_directory = 'saved_model'
    base_model_name = 'sign_language'
    model = load_most_recent_model(model_directory, base_model_name)

# Initializing camera
cap = cv2.VideoCapture(0)

# Initializing the mediapipe model
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Displaying countdown before starting predictions
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, str(i), (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.imshow('Realtime LSTM Sign Language Detection', frame)
        cv2.waitKey(1000)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Performing detections
        image, results = landmarks_detection(frame, holistic)
        
        # Drawing landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic for the LSTM model (processing last 30 frames from the webcam feed continuously)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            # Applying exponential smoothing to the probabilities for better consistency
            if smoothed_probabilities is None:
                smoothed_probabilities = res
            else:
                smoothed_probabilities = alpha * res + (1 - alpha) * smoothed_probabilities
            
            current_prediction = np.argmax(smoothed_probabilities)
            predictions.append(current_prediction)
            
            # Visualization logic
            consistency_check_window = 10
            if len(predictions) >= consistency_check_window:
                most_common_pred, num_occurrences = Counter(predictions[-consistency_check_window:]).most_common(1)[0] # Fetching the most common prediction
                if num_occurrences > consistency_check_window / 2 and smoothed_probabilities[most_common_pred] > threshold: # Checking if the most common prediction occurs more than half the time
                                                                                                                            # and if the probability of the prediction is greater than the threshold
                    current_sign = signs[most_common_pred]
                    if len(sentence) > 0 and current_sign != sentence[-1]:
                        sentence.append(current_sign)
                    elif len(sentence) == 0:
                        sentence.append(current_sign)
                        
            if len(sentence) > 5:
                sentence = sentence[-5:]
            
            # Displaying the sentence
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    
        cv2.imshow('Realtime LSTM Sign Language Detection', image)
        
        # Breaking gracefully
        if cv2.waitKey(10) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()