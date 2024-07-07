# Importing necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp
from utilities import landmarks_detection, draw_styled_landmarks, extract_keypoints
import argparse

# Function to check and create directories
def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Collect keypoints for a specific sign.")
parser.add_argument("sign", type=str, help="The sign for which to collect data.")
args = parser.parse_args()

# Initialize the parameters
no_sequences = 30  # Number of sequences to capture for each sign
sequence_length = 30  # Number of frames in each sequence
DATA_PATH = "./Sign_Language_Dataset"  # Path to save the keypoints data
exit_flag = False  
sign = args.sign
sign_path = os.path.join(DATA_PATH, sign)
check_and_create_dir(sign_path)

# Setting mediapipe model
mp_holistic = mp.solutions.holistic

# Initializing VideoCapture
cap = cv2.VideoCapture(0)

# Displaying countdown before starting collection
for i in range(10, 0, -1):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame for countdown.")
        break
    cv2.putText(frame, f'Starting in {i} seconds...', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
    cv2.imshow('Webcam Feed', frame)
    cv2.waitKey(1000)  # Wait for 1 second

# Check if the countdown was interrupted
if not ret:
    cap.release()
    cv2.destroyAllWindows()

else:
    # Start collecting data
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(no_sequences):
            if exit_flag:
                break
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame.")
                    break
                
                image, results = landmarks_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {sign} Video Number {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Press Esc key to end collection', (15, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Webcam Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Collecting frames for {sign} Video Number {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Press Esc key to end collection', (15, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Webcam Feed', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(sign_path, str(sequence), str(frame_num))
                check_and_create_dir(os.path.dirname(npy_path))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == 27:
                    exit_flag = True
                    break

            if exit_flag: # Break out of the outer loop
                break

        cap.release()
        cv2.destroyAllWindows()