import cv2
import mediapipe as mp
from feature_extraction import normalize_landmarks, get_hand_shape, calculate_rotation_angle, get_hand_position



mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
print("Press 'q' to quit, 'r' to set the reference")


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
