import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Function to recognize the player's hand gesture using the model
def recognize_gesture(hand_landmarks):
    # Extract the x, y coordinates of the landmarks
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)

    # Convert to numpy array and reshape for model input
    input_data = np.array(landmarks).reshape(1, 42)

    # Predict the gesture using the model
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted class to the corresponding gesture
    return choices[predicted_class]

# Load the gesture classifier model
model = tf.keras.models.load_model('rps_model.h5')
choices = ['Rock', 'Paper', 'Scissors']

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Load a video capture from webcam
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(frame_rgb)
    
    # If a hand is detected, recognize the gesture
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            player_choice = recognize_gesture(hand_landmarks)

            # Display the choice
            cv2.putText(frame, f"Choice: {player_choice}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    # Display the frame
    cv2.imshow('Rock-Paper-Scissors', frame)
    
    # Exit with 'Esc'
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
