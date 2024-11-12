import cv2
import numpy as np
import os 
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = model.process(image)          
    image.flags.writeable = True               
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,115,83), thickness=1, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(245,115,83), thickness=1, circle_radius=3)
                             ) 

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,115,83), thickness=1, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(245,115,83), thickness=1, circle_radius=3)
                             ) 
  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,115,83), thickness=1, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(245,115,83), thickness=1, circle_radius=3)
                             ) 


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

DATA_PATH = os.path.join('Training Data') 


no_sequences = 30

sequence_length = 20


def new_action():
    action = input("Input Action: ")
    
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
    return action

action = new_action()

cap = cv2.VideoCapture(0)
# Set mediapipe model 

with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence = 0.5, 
    model_complexity = 1, 
    smooth_landmarks = True) as holistic:

    
    for sequence in range(no_sequences):

        for frame_num in range(sequence_length):

            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_landmarks(image, results)
            

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (30, 30)
            fontScale = .5
            color = (255, 60, 0)
            thickness = 1
    
            if frame_num == 0: 
                text = "Preparing for sequence " + str(sequence) 
                image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)
                cv2.waitKey(1000)
                
            else: 
                text = "Collecting frame " + str(frame_num) + " for sequence " +  str(sequence)
                image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)
            

            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                        
    cap.release()
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
        
    



    