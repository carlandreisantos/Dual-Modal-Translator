import cv2
import mediapipe as mp
import json
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import threading
import speech_to_text
from feature_extraction import normalize_landmarks, get_hand_shape, calculate_rotation_angle, get_hand_position

last_detected_gesture = None

gesture_buffer = []  # Store last few detected gestures
buffer_size = 5  # Number of frames to consider
min_consistency = 3 

# Load gesture data
def load_data():
    with open("gestures.json", "r") as f:
        return json.load(f)

data = load_data()

def calculate_similarity(list1, list2, threshold):
    return len(list1) == len(list2) and all(abs(a - b) <= threshold for a, b in zip(list1, list2))

def calculate_angle_similarity(ref_angles, curr_angles, threshold):
    angle_diffs = [abs(r - c) for r, c in zip(ref_angles, curr_angles)]
    avg_diff = sum(angle_diffs) / len(angle_diffs)
    return avg_diff <= threshold

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Tkinter UI Setup
root = tk.Tk()
root.title("Gesture & Speech Recognition")
root.geometry("800x600")
root.configure(bg='white')
root.attributes('-fullscreen', True)
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
# Create a frame to center content
main_frame = tk.Frame(root, bg='white')
main_frame.pack(expand=True, fill='both')
main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Gesture History Label
gesture_label = tk.Label(main_frame, text="Gesture Recognition History:", font=("Arial", 12, "bold"), bg='white')
gesture_label.pack()

# Gesture History Box
gesture_history = scrolledtext.ScrolledText(main_frame, height=1, width=80)
gesture_history.pack()

# OpenCV Video Feed
video_label = tk.Label(main_frame)
video_label.pack()

# Speech-to-Text History Label
speech_label = tk.Label(main_frame, text="Speech-to-Text History:", font=("Arial", 12, "bold"), bg='white')
speech_label.pack()

# Speech-to-Text History Box
speech_history = scrolledtext.ScrolledText(main_frame, height=1, width=80)
speech_history.pack()

# Buttons
button_frame = tk.Frame(main_frame, bg='white')
button_frame.pack()

speech_button = tk.Button(button_frame, text="Start Speech Recognition", width=25)
speech_button.pack(side=tk.BOTTOM, padx=10, pady=5)

# Gesture Matching Thresholds
shape_threshold = 0.14
position_threshold = 0.15
angle_threshold = 15

# Update Video Feed
def update_frame():
    global last_detected_gesture
    success, frame = cap.read()
    if not success:
        return

    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    frame_height, frame_width, _ = flipped_frame.shape

    shape = [0] * 30
    pos = [0] * 8
    angle = [0] * 2
    best_match = None
    best_score = float('inf')

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

            mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pose_results.pose_landmarks:
        pose_landmarks = [(lm.x * frame_width, (1 - lm.y) * frame_height) for lm in pose_results.pose_landmarks.landmark]
        normalized_landmarks = normalize_landmarks(pose_landmarks)
        pos = get_hand_position(normalized_landmarks)
        mp_drawing.draw_landmarks(rgb_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    for gesture_name, variations in data.items():
        for variation_name, features in variations.items():
            score = 0
            valid = True
            
            if "Shape" in features:
                if not calculate_similarity(features["Shape"], shape, shape_threshold):
                    valid = False
                else:
                    score += sum(abs(a - b) for a, b in zip(features["Shape"], shape))
            
            if "Angle" in features:
                if not calculate_angle_similarity(features["Angle"], angle, angle_threshold):
                    valid = False
                else:
                    score += sum(abs(a - b) for a, b in zip(features["Angle"], angle))
            
            if "Pos" in features:
                if not calculate_similarity(features["Pos"], pos, position_threshold):
                    valid = False
                else:
                    score += sum(abs(a - b) for a, b in zip(features["Pos"], pos))
            
            if valid and score < best_score:
                best_match = gesture_name
                best_score = score


    if best_match:
        gesture_buffer.append(best_match)

        # Keep buffer within size limit
        if len(gesture_buffer) > buffer_size:
            gesture_buffer.pop(0)

        # Count occurrences of the most common gesture in the buffer
        most_common_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
        count = gesture_buffer.count(most_common_gesture)

        # Only update if a gesture appears consistently in the buffer
        if count >= min_consistency and most_common_gesture != last_detected_gesture:
            last_detected_gesture = most_common_gesture
            gesture_history.insert(tk.END, most_common_gesture + " ")
            gesture_history.yview(tk.END)  # Auto-scroll
    
    
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# Speech Recognition Thread
speech_running = False

def speech_callback(transcription):
    speech_history.delete("1.0", tk.END)  # Clear previous text
    speech_history.insert(tk.END, transcription)  # Insert only the current detection
    speech_history.yview(tk.END)  # Auto-scroll


def start_speech_recognition():
    global speech_running
    if speech_running:
        speech_running = False
        speech_button.config(text="Start Speech Recognition")
    else:
        speech_running = True
        speech_button.config(text="Stop Speech Recognition")
        threading.Thread(target=speech_to_text.start_speech_recognition, args=(speech_callback,), daemon=True).start()




speech_button.config(command=start_speech_recognition)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
