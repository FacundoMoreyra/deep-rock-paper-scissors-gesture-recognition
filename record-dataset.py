import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Load a video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize dataset arrays
dataset = []
labels = []

# Save the current landmarks and label as a dataset entry
def save_data(label):
    global dataset, labels, hand_landmarks
    if hand_landmarks is not None:
        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        dataset.append(landmarks.flatten())
        labels.append(label)
        print(f"Saved {label} gesture")

hand_landmarks = None

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    # If a hand is detected, draw the landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw only the fingertip landmarks
            for i in [mp_hands.HandLandmark.THUMB_TIP,
                      mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp_hands.HandLandmark.RING_FINGER_TIP,
                      mp_hands.HandLandmark.PINKY_TIP]:
                landmark = hand_landmarks.landmark[i]
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    # Instructions
    cv2.putText(frame, "Press 'r' for Rock, 'p' for Paper, 's' for Scissors", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Rock Paper Scissors Dataset Collection', frame)

    # Check for key press
    key = cv2.waitKey(5) & 0xFF
    if key == ord('r'):
        save_data(0)
    elif key == ord('p'):
        save_data(1)
    elif key == ord('s'):
        save_data(2)
    elif key == 27: # Exit with 'Esc'
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save the dataset
np.save('rps_dataset.npy', np.array(dataset))
np.save('rps_labels.npy', np.array(labels))
