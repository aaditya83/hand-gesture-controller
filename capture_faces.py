import cv2
import os

def main():
    # Ask for the user's name to label the image
    name = input("Enter the name for this face: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    # Ensure directory exists
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Load face detector for visual feedback (yellow box)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print(f"\n--- Capture Mode: {name} ---")
    print("1. Allow the camera to see your face.")
    print("2. Press 's' to SAVE the photo.")
    print("3. Press 'q' to QUIT.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        display_frame = frame.copy()

        # Draw box around face (visual guide)
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Instructions on screen
        cv2.putText(display_frame, "Press 's' to Save", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(display_frame, "Press 'q' to Quit", (10, 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow('Capture Face', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # 's' to Save
        # 's' to Save
        if key == ord('s'):
            if len(faces) == 0:
                print(" >> Error: No face detected.")
                cv2.rectangle(display_frame, (0,0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), cv2.FILLED)
                cv2.imshow('Capture Face', display_frame)
                cv2.waitKey(100)
                continue
            
            # Find the largest face (closest one)
            # Face format: (x, y, w, h) -> Area = w * h
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            # (Optional) We could crop to just the face, but usually saving the whole frame 
            # or a crop is fine. For better training, let's save the whole frame but 
            # maybe we only CARE that we found the largest one. 
            
            # Construct filename. 
            filename = f"{name}.jpg"
            filepath = os.path.join(known_faces_dir, filename)
            
            # Since we only want ONE face for this name, we can overwrite or warn.
            # But to be safe, if it exists, we just overwrite it to enforce "one face" per name entry effectively, 
            # Or simplified: just save and break.
            
            # Save the clean frame (without the yellow box)
            cv2.imwrite(filepath, frame)
            print(f" >> Success! Saved photo to: {filepath}")
            
            # Flash Green effect
            cv2.rectangle(display_frame, (0,0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), cv2.FILLED)
            cv2.imshow('Capture Face', display_frame)
            cv2.waitKey(500)
            
            print("Capture complete. Exiting...")
            break
            
        # 'q' to Quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
