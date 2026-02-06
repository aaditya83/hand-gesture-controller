# Face Identification System

This application identifies faces and detects simple emotions (Happy/Neutral) using your webcam.

## Features
*   **Identity**: Uses LBPH (OpenCV) to learn faces from images.
*   **Emotion**: Uses "Smile Detection" to differentiate between Neutral and Happy.

## Prerequisites
1.  **Python** installed.

## Setup
1.  Install dependencies:
    ```bash
    pip install opencv-contrib-python numpy
    ```

2.  **Add Faces** (if you haven't):
    *   Run `python capture_faces.py` and follow the instructions.
    *   Or drop photos into the `known_faces` folder.

## Run
```bash
python main.py
```
*Press 'q' to quit.*
