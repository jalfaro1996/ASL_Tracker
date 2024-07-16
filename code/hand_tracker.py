import cv2
import mediapipe as mp

# Use mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Create video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Focus on the hands
    results = hands.process(image)
    
    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Figure out the hand position
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
