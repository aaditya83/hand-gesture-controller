# Hand Gesture Controller

A Python application that uses MediaPipe to detect hand gestures and control your computer (e.g., simulating a steering wheel).

## Features
- **Gesture Recognition**: Detects 8+ gestures and hand states.
- **Driving Mode**: Maps gestures to keyboard inputs (Left/Right) for driving games.
- **High Sensitivity**: Optimized for fast detection.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `mediapipe`, `opencv-python`, and `pyautogui` installed)*

2. Download the Model:
   - The `hand_landmarker.task` file is included in this repository.

## Usage
Run the script:
```bash
python hand_gestures.py
```
- **Fist**: Brake (Left Arrow)
- **Open Hand (5 fingers)**: Gas (Right Arrow)
- **Press 'q'**: Quit
