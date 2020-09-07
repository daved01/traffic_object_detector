"""
Author: David Kirchhoff
Last edit: Sep 03 2020

v. 0.01

"""

import numpy as np
import cv2 as cv
import math

import torch
import torchvision
from torchvision.transforms import ToTensor

import modules.detector as detector
import modules.formatter as formatter


#%% Model stuff
# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, 
                                                             pretrained_backbone=True, trainable_backbone_layers=3,
                                                            min_size=360)
model.eval()

#%% Image stuff
frame_rate = 10.0 # Rate at which detection is performed. E.g. 10 means detection on each 10th image.

cap = cv.VideoCapture(0)
out = cv.VideoWriter('output/output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1.0, (640, 360))

if not cap.isOpened():
    print("ERROR! Cannot open camera")
    exit()

#%% Loop through each frame
frame_id = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is true
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if (frame_id % math.floor(frame_rate) == 0):
        
        # Prepare image
        frame = formatter.rescale_frame(frame, percent=50)
        frame_normed = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX) 
        frame_normed = ToTensor()(frame_normed)

        im_list = []
        im_list.append(frame_normed)
    
        # Predict and make bounding boxes
        predictions = model(im_list)
        frame_out = detector.draw_boxes(frame, predictions[0], thresh=0.8) 

        # Display the resulting frame
        cv.imshow('frame', frame_out)
        out.write(frame_out)

    if cv.waitKey(1) == ord('q'):
        break
    
    frame_id += 1 # Iterate through images

# When everything is done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()