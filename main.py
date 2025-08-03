import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl and other logging
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

import cv2
import mediapipe as mp
import time
from directkeys import right_pressed,left_pressed
from directkeys import PressKey, ReleaseKey


break_key_pressed=left_pressed
accelerato_key_pressed=right_pressed

time.sleep(2.0)
current_key_pressed = set()
previous_gesture = None  # Track previous gesture to avoid repeated key presses
frame_skip = 0  # Process every nth frame for better performance

mp_draw=mp.solutions.drawing_utils
mp_hand=mp.solutions.hands


tipIds=[4,8,12,16,20]

video=cv2.VideoCapture(0)

# Optimize camera settings for better performance
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag

# Optimize MediaPipe for better performance
with mp_hand.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand for better performance
    min_detection_confidence=0.7,  # Higher confidence for more stable detection
    min_tracking_confidence=0.5,
    model_complexity=1  # Use lighter model for better performance
) as hands:
    while True:
        # Skip frames for better performance
        frame_skip += 1
        ret, image = video.read()
        if not ret:
            continue

        if frame_skip % 2 == 0:  # Process every 2nd frame
            # Resize image for faster processing
            processed_image = cv2.resize(image, (320, 240))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image.flags.writeable = False
            results = hands.process(processed_image)
            processed_image.flags.writeable = True
            
            lmList = []
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    # Draw landmarks on the original size image
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
                    
                    # Get landmarks from the processed image size
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, c = processed_image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

            fingers = []
            if len(lmList) != 0:
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                total = fingers.count(1)
                
                current_gesture = "NONE"
                if total == 0:
                    current_gesture = "BRAKE"
                elif total == 5:
                    current_gesture = "GAS"

                # Only update keys if gesture has changed
                if current_gesture != previous_gesture:
                    if current_gesture == "BRAKE":
                        for key in current_key_pressed: ReleaseKey(key)
                        current_key_pressed.clear()
                        PressKey(break_key_pressed)
                        current_key_pressed.add(break_key_pressed)
                    elif current_gesture == "GAS":
                        for key in current_key_pressed: ReleaseKey(key)
                        current_key_pressed.clear()
                        PressKey(accelerato_key_pressed)
                        current_key_pressed.add(accelerato_key_pressed)
                    else:  # "NONE"
                        for key in current_key_pressed: ReleaseKey(key)
                        current_key_pressed.clear()
                    previous_gesture = current_gesture
        
        # Update gesture display outside the processing loop to prevent flickering
        if previous_gesture == "BRAKE":
            cv2.rectangle(image, (20, 300), (270, 425), (0, 0, 255), cv2.FILLED)  # Red box for brake
            cv2.putText(image, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        elif previous_gesture == "GAS":
            cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, " GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

        # Clean up and display
        cv2.imshow("Frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release any remaining pressed keys before closing
for key in current_key_pressed:
    ReleaseKey(key)
    
video.release()
cv2.destroyAllWindows()

