import cv2
import os
import numpy as np

def get_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def get_smile_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
    smile_cascade = cv2.CascadeClassifier(cascade_path)
    return smile_cascade

def train_model(known_faces_dir):
    face_cascade = get_face_detector()
    if not face_cascade: return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    known_faces, ids, names, current_id = [], [], {}, 0

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Created '{known_faces_dir}'. Please add images.")
        return None, {}

    print("Training model on known faces...")
    image_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found for training.")
        return None, {}

    for filename in image_files:
        path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in faces:
            known_faces.append(img[y:y+h, x:x+w])
            ids.append(current_id)
            names[current_id] = name
            break
        current_id += 1

    if ids:
        recognizer.train(known_faces, np.array(ids))
        print("Training complete!")
        return recognizer, names
    return None, {}

def get_eye_detector():
    # 'haarcascade_eye_tree_eyeglasses.xml' is often more robust for open eyes 
    # (even without glasses) than the standard eye cascade.
    cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    return eye_cascade

def main():
    KNOWN_FACES_DIR = "known_faces"
    recognizer, name_map = train_model(KNOWN_FACES_DIR)
    
    face_cascade = get_face_detector()
    smile_cascade = get_smile_detector()
    eye_cascade = get_eye_detector()

    video_capture = cv2.VideoCapture(0)
    print("Starting... Press 'q' to quit.")

    blink_frames = 0
    # Increased threshold: Eyes must be "missing" for ~15 frames (approx 0.5s) to count.
    # This prevents flickering noise from quitting the app.
    BLINK_THRESHOLD = 15 

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Faces
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            blink_frames = 0
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            # --- Identity ---
            label = "Unknown"
            if recognizer:
                try:
                    id_, conf = recognizer.predict(roi_gray)
                    if conf < 75: label = name_map.get(id_, "Unknown")
                except: pass
            
            # --- Smile ---
            roi_gray_lower = roi_gray[int(h/2):h, :]
            smiles = smile_cascade.detectMultiScale(roi_gray_lower, scaleFactor=1.7, minNeighbors=15)
            expression = "Neutral"
            if len(smiles) > 0: expression = "Happy"

            # --- Blink Detection ---
            roi_gray_upper = roi_gray[0:int(h/1.7), :]
            
            # Use minNeighbors=3 with the eyeglasses cascade for maximum sensitivity to detecting an open eye.
            # This handles "winking" better (it should still find the one open eye).
            eyes = eye_cascade.detectMultiScale(roi_gray_upper, scaleFactor=1.1, minNeighbors=3)
            
            blink_status = ""
            # Logic: If 0 eyes found -> Maybe Blinking. 
            if len(eyes) == 0:
                blink_frames += 1
            else:
                blink_frames = 0

            # Debug Info on screen: Show how many frames the eyes have been missing
            debug_text = f"Eyes: {len(eyes)} | BlinkCount: {blink_frames}/{BLINK_THRESHOLD}"
            cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            if blink_frames >= BLINK_THRESHOLD:
                print(">>> BLINK DETECTED! Terminating App (simulating 'q')...")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "BLINK QUIT!", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Face ID & Emotion', frame)
                cv2.waitKey(200)
                video_capture.release()
                cv2.destroyAllWindows()
                return

            color = (0, 255, 0)
            if expression == "Happy": color = (0, 255, 255)
            if blink_frames > 2: color = (0, 0, 255) # Warning color if starting to blink

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
            # Don't show confusing blink text at bottom, just expression
            cv2.putText(frame, expression, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face ID & Emotion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
