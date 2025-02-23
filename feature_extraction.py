import cv2
import mediapipe as mp
import math

def calculate_distance(point1, point2):

    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def normalize_landmarks(landmarks):

    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)


    return [
        (
            (lm[0] - min_x) / (max_x - min_x),
            (lm[1] - min_y) / (max_y - min_y)
        ) for lm in landmarks
    ]

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
