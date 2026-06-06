import cv2
import mediapipe as mp
import rtmidi
import math
import time

midi_out = rtmidi.MidiOut()

print("creating virtual midi port: 'hand tracking midi'...")
midi_out.open_virtual_port("hand tracking midi")

# Dynamically grabbing solutions safely
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def map_val(val, in_min, in_max, out_min, out_max):
    mapped = out_min + ((val - in_min) / (in_max - in_min)) * (out_max - out_min)
    return int(max(out_min, min(out_max, mapped)))

last_cc1 = -1
last_cc2 = -1

print("system ready hold ur hand up to the camera. press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist_y = 1.0 - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            cc1_value = map_val(wrist_y, 0.2, 0.8, 0, 127)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = math.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2 +
                (thumb_tip.z - index_tip.z) ** 2
            )
            cc2_value = map_val(distance, 0.03, 0.25, 0, 127)

            if cc1_value != last_cc1:
                midi_out.send_message([176, 1, cc1_value])
                last_cc1 = cc1_value
            if cc2_value != last_cc2:
                midi_out.send_message([176, 2, cc2_value])
                last_cc2 = cc2_value

            cv2.putText(frame, f"height (CC1): {cc1_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"pinch (CC2): {cc2_value}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("hand midi controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
del midi_out
print("midi closed, bye")
