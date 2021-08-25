
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from scipy.signal import argrelextrema
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import tensorflow as tf
import sys
import os 
from tensorflow import keras
import numpy as np
import argparse
import imutils
import joblib
import time
import dlib
import cv2
import os

######### load movenet model ##########
pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)
import utils
from movenet import Movenet
movenet = Movenet('movenet_thunder')

model = keras.models.load_model('/home/oussa/Desktop/yoga_pose_model.h5')

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)



############################## detection #######################################



vs = cv2.VideoCapture(-1)
while True :
    ret, frame = vs.read()
    
    if ret == False:
        break 
    frame = cv2.resize(frame, (256,256))
    pose_landmarks = movenet.detect(frame)
    draw_connections(frame,pose_landmarks,EDGES,0)
    pose_landmarks = pose_landmarks[:,0:2].reshape(1,17,2) 
    max = np.argmax(model.predict(pose_landmarks))
    if max == 0 : 
        cv2.putText(frame, "chair", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if max == 1 : 
        cv2.putText(frame, "cobra", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if max == 2 : 
        cv2.putText(frame, "dog", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if max == 3 : 
        cv2.putText(frame, "tree",(10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if max == 4 : 
        cv2.putText(frame, "warrior", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    frame = cv2.resize(frame, (1000,1000))    
    cv2.imshow('yoga-pose',frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break

cv2.destroyAllWindows()    