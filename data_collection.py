import cv2
import mediapipe as mp
import json
import os
from feature_extraction import normalize_landmarks, get_hand_shape, calculate_rotation_angle, get_hand_position

DATA_FILE = "gestures.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

data = load_data()
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1)

recording = False
gesture_name = ""
variation_count = 1
include_shape = False
include_angle = False
include_pos = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    frame_height, frame_width, _ = flipped_frame.shape

    shape = [0] * 30
    pos = [0] * 8
    angle = [0] * 2

    if hands_results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height) for lm in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks)
            hand_shape = get_hand_shape(normalized_landmarks)
            hand_angle = calculate_rotation_angle(landmarks)

            if hand_label.classification[0].label == "Right":
                shape[15:] = hand_shape  
                angle[1] = hand_angle  
            else:
                shape[:15] = hand_shape  
                angle[0] = hand_angle  

            mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pose_results.pose_landmarks:
        pose_landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height) for lm in pose_results.pose_landmarks.landmark]
        normalized_landmarks = normalize_landmarks(pose_landmarks)
        pos = get_hand_position(normalized_landmarks)
        mp_drawing.draw_landmarks(flipped_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Gesture Data Collection", flipped_frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('r'):
        gesture_name = input("Enter gesture name: ")
        if gesture_name not in data:
            data[gesture_name] = {}
        variation_count = 1
        print("Select features to record for all variations:")
        include_shape = input("Include shape? (y/n): ").strip().lower() == 'y'
        include_angle = input("Include angle? (y/n): ").strip().lower() == 'y'
        include_pos = input("Include position? (y/n): ").strip().lower() == 'y'
        print("Recording new gesture. Press SPACE to save each of 30 variations.")
        recording = True
    
    if key == ord(' ') and recording and variation_count <= 30:
        print(f"Recording variation {variation_count}/30")
        
        variation_data = {}
        if include_shape:
            variation_data["Shape"] = shape
        if include_angle:
            variation_data["Angle"] = angle
        if include_pos:
            variation_data["Pos"] = pos

        data[gesture_name][f"Variation {variation_count}"] = variation_data
        variation_count += 1
        save_data(data)
        print("Variation saved.")
        
        if variation_count > 30:
            print("30 variations recorded. Stopping recording.")
            recording = False
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
