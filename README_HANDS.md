# Hand Gesture Recognition (MediaPipe)

This script uses the **MediaPipe Hand Landmarker** model (`hand_landmarker.task`) to detect hand landmarks with high precision.
Instead of simple color detection, it uses a pre-trained neural network to find the coordinates of 21 points on your hand, allowing for complex gesture recognition.

## Supported Gestures
The script manually classifies the following gestures based on finger positions:
- **Numbers**: 1, 2, 3, 4, 5
- **Special**: Fist, Rock, Spiderman, Call Me, Thumb Up, Point, Pinky

## Usage

1.  **Run the script**:
    ```bash
    python hand_gestures.py
    ```
    *(Or `& d:/flex/.venv/Scripts/python.exe hand_gestures.py` if needed)*

2.  **Controls**:
    - **Q**: Quit immediately.
    - **Thumb Up**: Hold a "Thumb Up" gesture to trigger the saving/quitting timer.

## How it works
1.  **Landmark Detection**: The `hand_landmarker.task` model analyzes the video frame.
2.  **Geometric Analysis**:
    - The script calculates which fingers are open/extended.
    - It compares these states to specific patterns (e.g., Index + Pinky = "Rock").
3.  **Visualization**: Draws the skeleton and gesture label on the screen.

