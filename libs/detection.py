import os
import sys
import numpy as np
from libs.utils import get_yolo_boxes
from keras.models import load_model


# example prediction taken from yolov3/predict.py
NET_H, NET_W = 416, 416
OBJ_THRESH, NMS_THRESH = 0.9, 0.45
ANCHORS = [7,9, 8,11, 10,12, 11,15, 14,18, 16,22, 20,26, 25,32, 33,42]

class Detector:
    def __init__(self, weights_path=None):
        if weights_path is None: 
            print('Missing weights: No path provided')
            sys.exit(-1)
        self.model = load_model(weights_path)

    def predictBoxes(self, img=None):
        if img is None:
            print('Missing image: No image provided')
            sys.exit(-1)
        else:
            boxes = get_yolo_boxes(self.model, [img], NET_H, NET_W, ANCHORS, OBJ_THRESH, NMS_THRESH)[0]
            return boxes
