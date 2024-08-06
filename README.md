# Sign Language Detection

This project aims to detect sign language gestures using a trained LSTM model. It leverages the use of MediaPipe for landmark detection and a Sequential LSTM model for gesture recognition. The project includes scripts for data collection, model training, and real-time gesture detection. Currently only the words 'what', 'your', 'name', 'how', 'you' can be detected by the model to try to form meaningful sentences out of it. Addition of  more words can be done by following the steps mentioned below.

## Features

- **Data Collection**: Collects keypoints data for different sign language gestures using a webcam.
- **Model Training**: Trains an LSTM model on the collected keypoints data.
- **Real-time Detection**: Detects sign language gestures in real-time using a webcam feed.

## Installation

To run this code, you need to have Python installed on your system. The project has been tested on Python version 3.12. Follow these steps to set up the project:

1. **Clone the repository**:
    ```sh
    git clone [https://github.com/JoyBis48/Sign_Language_Detection.git](https://github.com/JoyBis48/Action_Detection.git)
    ```
2. **Navigate to the project directory**:
    ```sh
    cd Sign_Language_Detection
    ```
3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## How It Works

### Data Collection

The **collection.py** script is used to collect keypoints data for different sign language gestures. It uses MediaPipe to detect landmarks and saves the keypoints data as `.npy` files. Once the collection.py script is run, there will be a 10 second countdown before the webcam starts to capture the mediapipe holistic keypoints from the signs displayed by the user. For better quality of the dataset captured, the number of sequences has been set to 30 and the number of frames captured in each sequence is also set to 30.

1. **Run the data collection script**:
    ```sh
    python collection.py <sign>
    ```
    Replace `<sign>` with the name of the sign you want to collect data for.

### Model Training

The **train.py** script is used to train an LSTM model on the collected keypoints data.

1. **Run the training script**:
    ```sh
    python train.py
    ```
    This will train the model and save it in the **saved_model** directory.

### Real-time Detection

The **test.py** script is used to detect sign language gestures in real-time using a webcam feed.

1. **Run the detection script**:
    ```sh
    python test.py
    ```
    Optionally, you can specify a model to load using the `--model` argument:
    ```sh
    python test.py --model <model_name>
    ```

## Dependencies

- OpenCV
- NumPy
- MediaPipe
- TensorFlow
- scikit-learn
- tqdm

## File Descriptions

- **collection.py**: Script for collecting keypoints data for different sign language gestures.
- **train.py**: Script for training an LSTM model on the collected keypoints data.
- **test.py**: Script for detecting sign language gestures in real-time using a webcam feed.
- **utilities.py**: Contains utility functions for landmark detection, drawing landmarks, extracting keypoints, and loading models.

## Directory Structure

```
Sign_Language_Detection/
├── collection.py
├── train.py
├── test.py
├── utilities.py
├── requirements.txt
├── Sign_Language_Dataset/
│   └── <sign>/
│       └── <sequence>/
│           └── <frame>.npy
├── saved_model/
│   └── <model_name>.keras
└── Logs/
```
