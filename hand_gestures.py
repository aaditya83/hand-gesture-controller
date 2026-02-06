import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import time
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    MODEL_PATH = r'd:\flex\python\face_recognition_app\hand_landmarker.task'
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # Load the model. Ensure the path is correct!
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=1, 
        # Fine-Tuning Tip: Increase this value (e.g., to 0.5) if you get too many false positives.
        # Lower values (like 0.1) are good for detecting hands in poor lighting or quick motion,
        # but might pick up noise.
        min_hand_detection_confidence=0.1, 
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )
    
    detector = vision.HandLandmarker.create_from_options(options)

    # Connections between landmarks to draw the skeleton
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),   
        (0, 5), (5, 6), (6, 7), (7, 8),   
        (5, 9), (9, 10), (10, 11), (11, 12), 
        (9, 13), (13, 14), (14, 15), (15, 16), 
        (13, 17), (17, 18), (18, 19), (19, 20), 
        (0, 17) 
    ]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    print("--- Hand Gesture Ultimate ---")
    print("Sensitivity: EXTREME (0.1)")
    print("Gestures: 1, 2, 3, 4, 5, Fist, Rock, Spiderman, Call Me")
    print("Commands: Fist -> Back Arrow | Open Hand -> Forward Arrow")
    print("Press 'q' in the window to QUIT.")
    
    last_timestamp_ms = 0
    prev_frame_time = 0
    
    # Track the state of our driving controls
    current_key_held = None
    last_action_text = "NEUTRAL"
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Calculate Frames Per Second (FPS) so we can see how smooth it's running
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        # Mirror the frame so it feels natural to the user
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Mediapip needs a monotonically increasing timestamp
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms: timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms
        
        result = detector.detect_for_video(mp_image, timestamp_ms)
        
        box_color = (0, 0, 0) # Default box color
        gesture = "Unknown"

        if result and result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            
            # 1. Visualize the landmarks
            points = {}
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
            for p1, p2 in HAND_CONNECTIONS:
                if p1 in points and p2 in points:
                    cv2.line(frame, points[p1], points[p2], (200, 255, 200), 2)
            
            # 2. Decode the gesture
            fingers = []
            
            # Check the Thumb
            # Fine-Tuning Tip: The multiplier '1.0' compares the thumb tip distance to the knuckle distance.
            # Adjust this if you have flexible thumbs or it's triggering too easily!
            p4, p3, p17 = hand_landmarks[4], hand_landmarks[3], hand_landmarks[17]
            d4_17 = ((p4.x - p17.x)**2 + (p4.y - p17.y)**2)
            d3_17 = ((p3.x - p17.x)**2 + (p3.y - p17.y)**2)
            thumb_open = d4_17 > d3_17 * 1.0
            fingers.append(1 if thumb_open else 0)
            
            # Check the other 4 Fingers (simple height check vs knuckle)
            # Note: Y coordinates increase downwards in images
            fingers.append(1 if hand_landmarks[8].y < hand_landmarks[6].y else 0)
            fingers.append(1 if hand_landmarks[12].y < hand_landmarks[10].y else 0)
            fingers.append(1 if hand_landmarks[16].y < hand_landmarks[14].y else 0)
            fingers.append(1 if hand_landmarks[20].y < hand_landmarks[18].y else 0)
            
            count = sum(fingers)
            
            # --- MAP GESTURES TO ACTIONS ---
            target_key = None
            
            if count == 0: 
                gesture = "Fist (BRAKE)"
                target_key = 'left'
                box_color = (255, 0, 0)
            elif count == 5: 
                gesture = "Five (GAS)"
                target_key = 'right'
                box_color = (0, 255, 0)
            elif count == 4: gesture = "Four (4)"
            
            # Identify other specific gestures
            elif count == 3:
                if fingers[0] and fingers[1] and fingers[4]: gesture = "Spiderman"
                else: gesture = "Three (3)"
            elif count == 2:
                if fingers[1] and fingers[2]: gesture = "Victory (2)"
                elif fingers[1] and fingers[4]: gesture = "Rock"
                elif fingers[0] and fingers[4]: gesture = "Call Me"
                else: gesture = "Two (2)"
            elif count == 1:
                if fingers[0]: gesture = "Thumb Up"
                elif fingers[1]: gesture = "Point (1)"
                elif fingers[4]: gesture = "Pinky (1)"
                else: gesture = "One (1)"
            
            # Manage Keyboard Inputs (State Machine)
            # We check if the state changed to avoid spamming the key press
            if target_key != current_key_held:
                if current_key_held:
                    pyautogui.keyUp(current_key_held)
                    print(f"Released: {current_key_held}")
                
                if target_key:
                    pyautogui.keyDown(target_key)
                    print(f"Holding: {target_key}")
                
                current_key_held = target_key
            
            # Update the on-screen status text
            if current_key_held == 'right': last_action_text = "GAS >>"
            elif current_key_held == 'left': last_action_text = "<< BRAKE"
            else: last_action_text = "NEUTRAL"
            
            # Draw the classification result
            top_y = min([p[1] for p in points.values()])
            cv2.putText(frame, gesture, (points[0][0]-40, top_y-20), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
            
            # Highlight the hand area
            xs = [p[0] for p in points.values()]
            ys = [p[1] for p in points.values()]
            cv2.rectangle(frame, (min(xs)-20, min(ys)-20), (max(xs)+20, max(ys)+20), box_color, 2)

        else:
             # If no hand is seen, warn the user
             cv2.putText(frame, "Waiting for Hand...", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             
             # SAFETY: If we lose the hand, stop driving immediately!
             if current_key_held:
                 pyautogui.keyUp(current_key_held)
                 print(f"Safety Release: {current_key_held}")
                 current_key_held = None
                 last_action_text = "NEUTRAL"

        # Show feedback in the UI (always useful for the user to know state)
        cv2.putText(frame, last_action_text, (w//2 - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)

        cv2.putText(frame, f"FPS: {int(fps)}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Hand Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # IMPORTANT: Release any held keys before killing the app
            if current_key_held: 
                pyautogui.keyUp(current_key_held)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
