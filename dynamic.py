import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def normalize_angles(angles, min_val=0, max_val=180):
    return [(a - min_val) / (max_val - min_val) if a is not None else 0 for a in angles]

def extract_flexion_angles(landmarks):
    indices = [
        (7, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20),
        (12, 14, 16), (11, 13, 15)
    ]
    
    angles = []
    for idx1, idx2, idx3 in indices:
        if idx1 in landmarks and idx2 in landmarks and idx3 in landmarks:
            angles.append(calculate_angle(landmarks[idx1], landmarks[idx2], landmarks[idx3]))
    
    if not angles:
        return [0] * len(indices)  # Avoid empty sequences
    
    return normalize_angles(angles)

def recognize_gesture(input_seq, reference_seq):
    if len(input_seq) < 20 or len(reference_seq) < 20:
        return "No reference recorded", None
    
    distance, _ = fastdtw(np.array(input_seq), np.array(reference_seq), dist=euclidean)
    return ("Match" if distance < 10 else "No Match"), distance  # Adjusted threshold

def main():
    cap = cv2.VideoCapture(0)
    reference_seq = []
    input_seq = []
    recording = False
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            all_landmarks = {}
            for landmark_set in [results.right_hand_landmarks, results.left_hand_landmarks, results.pose_landmarks]:
                if landmark_set:
                    for i, lm in enumerate(landmark_set.landmark):
                        all_landmarks[i] = (lm.x, lm.y, lm.z)
            
            flexion_angles = extract_flexion_angles(all_landmarks)
            
            if recording:
                reference_seq.append(flexion_angles)
                if len(reference_seq) == 20:
                    recording = False
                    print("Recording complete")
            
            if len(reference_seq) == 20:
                input_seq.append(flexion_angles)
                if len(input_seq) > 20:
                    input_seq.pop(0)
                
                if len(input_seq) == 20:
                    gesture_name, distance = recognize_gesture(input_seq, reference_seq)
                    cv2.putText(image, f"Gesture: {gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if distance is not None:
                        cv2.putText(image, f"Distance: {distance:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Recognition', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                reference_seq.clear()
                input_seq.clear()
                recording = True
                print("Recording started...")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()