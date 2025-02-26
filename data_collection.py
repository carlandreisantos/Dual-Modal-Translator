import cv2
import mediapipe as mp
import json
import os
from feature_extraction import normalize_landmarks, get_hand_shape, calculate_rotation_angle, get_hand_position

DATA_FILE = "gestures.json"

def load_data():
    """Load gesture data from JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    """Save gesture data to JSON file."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Load existing gesture data
data = load_data()

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Recording variables
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

    # Preprocess frame
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    frame_height, frame_width, _ = flipped_frame.shape

    # Initialize feature lists
    shape = [0] * 30
    pos = [0] * 8
    angle = [0] * 2

    # Process hand landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height) for lm in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks)
            hand_shape = get_hand_shape(normalized_landmarks)
            hand_angle = calculate_rotation_angle(landmarks)

            if hand_label.classification[0].label == "Right":
                shape[15:] = hand_shape  # Right hand
                angle[1] = hand_angle  
            else:
                shape[:15] = hand_shape  # Left hand
                angle[0] = hand_angle  

            mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Process pose landmarks
    if pose_results.pose_landmarks:
        pose_landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height) for lm in pose_results.pose_landmarks.landmark]
        normalized_landmarks = normalize_landmarks(pose_landmarks)
        pos = get_hand_position(normalized_landmarks)
        mp_drawing.draw_landmarks(flipped_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display frame
    cv2.imshow("Gesture Data Collection", flipped_frame)
    key = cv2.waitKey(5) & 0xFF

    # Start recording a new gesture
    if key == ord('r'):
        gesture_name = input("Enter gesture name: ").strip()
        
        if gesture_name not in data:
            data[gesture_name] = {}

        # Determine next available variation number
        existing_variations = len(data[gesture_name])
        variation_count = existing_variations + 1  

        print("Select features to record for all variations:")
        include_shape = input("Include shape? (y/n): ").strip().lower() == 'y'
        include_angle = input("Include angle? (y/n): ").strip().lower() == 'y'
        include_pos = input("Include position? (y/n): ").strip().lower() == 'y'

        print(f"Recording new gesture. Press SPACE to save each of up to 30 new variations.")
        recording = True

    # Record and save variations
    if key == ord(' ') and recording:
        if variation_count <= existing_variations + 30:  # Allow adding up to 30 new variations
            print(f"Recording variation {variation_count}")

            variation_data = {}
            if include_shape:
                variation_data["Shape"] = shape
            if include_angle:
                variation_data["Angle"] = angle
            if include_pos:
                variation_data["Pos"] = pos

            # Append new variation
            data[gesture_name][f"Variation {variation_count}"] = variation_data
            variation_count += 1
            save_data(data)
            print("Variation saved.")

        if variation_count > existing_variations + 30:
            print("30 new variations recorded. Stopping recording.")
            recording = False

    # Remove a gesture
    if key == ord('v'):
        gesture_name = input("Enter gesture name to delete: ").strip()

        if gesture_name in data:
            confirm = input(f"Are you sure you want to delete '{gesture_name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                del data[gesture_name]
                save_data(data)
                print(f"Gesture '{gesture_name}' has been deleted.")
            else:
                print("Deletion canceled.")
        else:
            print(f"Gesture '{gesture_name}' not found.")

    # Quit program
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
