import cv2
import os
import time

def main():
    cap = cv2.VideoCapture(0)
    
    # Setup directories
    base_dir = "hand_data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    classes = ["open", "closed", "victory", "one"]
    for c in classes:
        path = os.path.join(base_dir, c)
        if not os.path.exists(path):
            os.makedirs(path)

    print("--- Hand Gesture Data Collector ---")
    print(f"Classes: {classes}")
    print("Press '0' -> Save to 'open'")
    print("Press '1' -> Save to 'closed'")
    print("Press '2' -> Save to 'victory'")
    print("Press '3' -> Save to 'one'")
    print("Press 'q' -> Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # Display instructions
        cv2.putText(frame, "0: Open | 1: Closed | 2: Victory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        label = None
        if key == ord('0'): label = "open"
        elif key == ord('1'): label = "closed"
        elif key == ord('2'): label = "victory"
        elif key == ord('3'): label = "one"
        elif key == ord('q'): break
        
        if label:
            # Generate filename
            ts = int(time.time() * 1000)
            filename = f"{base_dir}/{label}/{ts}.jpg"
            cv2.imwrite(filename, frame)
            
            # Flash Effect
            print(f"Saved to {filename}")
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (255, 255, 255), cv2.FILLED)
            cv2.imshow("Data Collector", frame)
            cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
