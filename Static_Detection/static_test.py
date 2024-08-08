import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from PIL import Image
import os
import tempfile

# Load the trained MLP Classifier model
model = joblib.load('saved_models/mlp_model.joblib')

# Loading the class dictionary
class_names = {i: str(i) for i in range(10)}  # For numbers 0-9
class_names.update({10 + i: chr(97 + i) for i in range(26)})  # For letters a-z

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


def skeletal_image(image_path, shape=(256, 256, 3)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb) 
    if not results.multi_hand_landmarks:
        return None
    white_background = np.ones(shape, dtype=np.uint8) * 255
    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(white_background, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    white_background_bgr = cv2.cvtColor(white_background, cv2.COLOR_RGB2BGR)
    return white_background_bgr


# Function to extract landmarks from an uploaded image
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        # Extract landmarks
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
        return landmarks
    return None



# Streamlit app
st.title('Hand Gesture Recognition')

# Option for the user to choose the input method
input_method = st.radio("Choose the input method:", ("Upload an Image", "Use Webcam"))

if input_method == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert('RGB')
elif input_method == "Use Webcam":
    uploaded_file = st.camera_input("Take a picture")
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert('RGB')

if uploaded_file is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    # Save the uploaded or captured image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
        uploaded_image.save(tmpfile, format="JPEG")
        tmpfile_path = tmpfile.name
    
    try:
        skeletal_img = skeletal_image(tmpfile_path)
        if skeletal_img is not None:
            st.image(skeletal_img, channels="BGR", caption='This processed image contains your hand landmarks')
        processed_image = extract_landmarks(tmpfile_path)
        
        if processed_image is not None:
            with st.spinner('Please wait, while the model predicts...'):
                # Reshape the processed_image for the model if necessary
                processed_image = processed_image.reshape(1, -1)  # Reshape if your model expects a specific input shape
                predictions = model.predict(processed_image)
                predicted_class_name = class_names[predictions[0]]
                
                # Display the prediction
                st.write(f"The predicted ASL sign seems to be {predicted_class_name.upper()}")
        else:
            st.write("No hand landmarks were detected.")
    finally:
        # Ensure the temporary file is deleted even if an error occurs
        os.remove(tmpfile_path)


