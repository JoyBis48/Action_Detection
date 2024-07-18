# Importing the required libraries
import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
from tensorflow.keras.models import load_model # type: ignore

# Initializing MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to detect landmarks in the image
def landmarks_detection(image, holistic):
    # Conversion of the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    # Conversion of image back to BGR for displaying purpose
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# Function to draw landmarks with more distinct colors and styles
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(255, 20, 147), thickness=2, circle_radius=3), 
    mp_drawing.DrawingSpec(color=(255, 165, 0), thickness=2, circle_radius=2)
    ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(0, 191, 255), thickness=2, circle_radius=3), 
    mp_drawing.DrawingSpec(color=(147, 112, 219), thickness=2, circle_radius=2)
    )
    return None
    
 # Function to extract the keypoints from the results   
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def get_unique_filename(base_path):
    directory, base_filename = os.path.split(base_path)
    timestamp = datetime.now().strftime("%H%M%S")
    unique_filename = f"{base_filename}_{timestamp}.keras"
    unique_path = os.path.join(directory, unique_filename)
    return unique_path

def load_most_recent_model(directory, base_name):
    files = os.listdir(directory)
    # Filtering only keras saved models
    model_files = [file for file in files if file.startswith(base_name) and file.endswith('.keras')]
    # Sorting the saved models by modification time in descending order
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    # Loading the most recent model if any models were found
    if model_files:
        model_path = os.path.join(directory, model_files[0])
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        return model
    else:
        print("No model found.")
        return None

