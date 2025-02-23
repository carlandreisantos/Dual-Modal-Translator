import cv2
import mediapipe as mp
import math

def calculate_distance(point1, point2):
    """Calculate 3D Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)

def normalize_landmarks(landmarks):
    """Normalize hand landmarks in 3D (x, y, z)."""
    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]
    z_coords = [lm[2] for lm in landmarks]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    normalized_landmarks = [
        (
            (lm[0] - min_x) / (max_x - min_x),
            (lm[1] - min_y) / (max_y - min_y),
            (lm[2] - min_z) / (max_z - min_z)
        ) for lm in landmarks
    ]

    return normalized_landmarks

def get_hand_shape(landmarks):
    pairs = [(20, 0), (16, 0), (12, 0), (8, 0), (4, 0), (20, 16), (16, 12), 
             (12, 8), (8, 4), (4, 17), (8, 5), (12, 9), (16, 13), (10, 17)]
    return [calculate_distance(landmarks[p1], landmarks[p2]) for p1, p2 in pairs]

def calculate_rotation_angle(landmarks):
    wrist = landmarks[0]  
    index_finger = landmarks[5]  

    delta_x = index_finger[0] - wrist[0]
    delta_y = index_finger[1] - wrist[1]

    return math.degrees(math.atan2(delta_y, delta_x))

def get_hand_position(landmarks):
    left_pairs = [(18, 12), (18, 8), (18, 10), (18, 0)]
    right_pairs = [(17, 11), (17, 9), (17, 7), (17, 0)]

    left_pos = [calculate_distance(landmarks[p1], landmarks[p2]) for p1, p2 in left_pairs]
    right_pos = [calculate_distance(landmarks[p1], landmarks[p2]) for p1, p2 in right_pairs]
    
    return left_pos + right_pos

def calculate_similarity(list1, list2, threshold):
    return len(list1) == len(list2) and all(abs(a - b) <= threshold for a, b in zip(list1, list2))

def calculate_angle_similarity(ref_angles, curr_angles, threshold):

    angle_diffs = [abs(r - c) for r, c in zip(ref_angles, curr_angles)]
    avg_diff = sum(angle_diffs) / len(angle_diffs)
    return avg_diff <= threshold

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
print("Press 'q' to quit, 'r' to set the reference")

# Reference variables for both hands
reference_shape = reference_shape = [0] * 30
reference_pos = reference_pos = [0] * 8
reference_angle = reference_angle = [0] * 2

# Thresholds
shape_threshold = 0.3
position_threshold = 0.2
angle_threshold = 15  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    frame_height, frame_width, _ = flipped_frame.shape

    # Default values to prevent None issues
    shape = [0] * 30
    pos = [0] * 8
    angle = [0] * 2
    match_detected = False

    if hands_results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height, lm.z * frame_width) for lm in hand_landmarks.landmark]
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
        pose_landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height, lm.z * frame_width) for lm in pose_results.pose_landmarks.landmark]
        normalized_landmarks = normalize_landmarks(pose_landmarks)
        pos = get_hand_position(normalized_landmarks)
        mp_drawing.draw_landmarks(flipped_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    shape_match = calculate_similarity(shape, reference_shape, shape_threshold)
    position_match = calculate_similarity(pos, reference_pos, position_threshold)
    angle_match = calculate_angle_similarity(angle, reference_angle, angle_threshold)
    match_detected = shape_match and position_match and angle_match

    # Display Information
    cv2.putText(flipped_frame, f"Shape Match: {match_detected}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if match_detected else (0, 0, 255), 1)
    cv2.putText(flipped_frame, f"Pos Match: {position_match}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if position_match else (0, 0, 255), 1)



    cv2.imshow("Gesture Recognition", flipped_frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reference_shape = shape
        reference_pos = pos
        reference_angle = angle
        print("Reference updated for both hands!")

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
