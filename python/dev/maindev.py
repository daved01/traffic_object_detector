import numpy as np
import cv2 as cv
import math

import torch
import torchvision
from torchvision.transforms import ToTensor

import modules.detector as detector
import modules.formatter as formatter


def under_dev():


    return



#%% Model stuff
# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, 
                                                             pretrained_backbone=True, trainable_backbone_layers=3,
                                                            min_size=360)
model.eval()

frame = cv.imread('./dev/Intersection.jpg')


#%%

# Prepare image
frame = formatter.rescale_frame(frame, percent=100)
frame_normed = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX) 
frame_normed = ToTensor()(frame_normed)

im_list = []
im_list.append(frame_normed)
    
# Predict and make bounding boxes
predictions = model(im_list)
frame_out = detector.draw_boxes(frame, predictions[0], thresh=0.9) 


#%%

cv.imshow('image',frame_out)
cv.waitKey(0)
cv.destroyAllWindows()