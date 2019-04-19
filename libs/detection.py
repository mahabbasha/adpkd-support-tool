import os
import sys
import numpy as np
# from libs.utils import get_yolo_boxes
from keras.models import load_model
from libs.bbox import BoundBox
from libs.mrcnn.config import Config
from libs.mrcnn import model as modellib, utils
from libs.unet import model as unetModel
from skimage.transform import resize
# from skimage.io import imsave
from skimage import img_as_ubyte
# import cv2

# example prediction taken from yolov3/predict.py
NET_H, NET_W = 416, 416
OBJ_THRESH, NMS_THRESH = 0.8, 0.45
ANCHORS = [7,9, 8,11, 10,12, 11,15, 14,18, 16,22, 20,26, 25,32, 33,42]
LOGS_DIR = '../logs'


# modified example from matterport Mask RCNN repo
class CellConfig(Config):
    NAME = "cell"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + cell
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8

class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNDetector:
    def __init__(self, weights_path=None):
        if weights_path is None:
            print('Missing weights: No path provided')
            sys.exit(-1)
        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', config=config, model_dir=LOGS_DIR)
        self.model.load_weights(weights_path, by_name=True)

    def buildContourPoints(self, bin_img):
        points = list()
        ls = bin_img.astype(np.uint8)
        left_side, right_side = list(), list()
        top_side, bottom_side = list(), list()
        for y in range(ls.shape[0]):
            try:
                x_left, x_right = np.where(ls[y, :] == 1)[0][0], np.where(ls[y, :] == 1)[0][-1]
                left_side.append((y, x_left))
                right_side.append((y, x_right))
            except Exception as e:
                # print(e)
                continue
        for x in range(ls.shape[1]):
            try:
                y_top, y_bottom = np.where(ls[:, x] == ls[:, x].max())[0][0], np.where(ls[:, x] == ls[:, x].max())[0][-1]
                top_side.append((y_top, x))
                bottom_side.append((y_bottom, x))
            except Exception as e:
                # print(e)
                continue
        points = [x for i, x in enumerate(left_side+list(reversed(right_side))) if (x in top_side+list(reversed(bottom_side)))]
        if len(points) > 30: 
            points = [x for i,x in enumerate(points) if i % 4 == 0]
        # points = [x for i, x in enumerate(points) if i % 5 == 0]
        return points
    
    def predictBoxesAndContour(self, img=None):
        boxes = list()
        if img is None: 
            print('Missing image: No image provided')
            sys.exit(-1)
        else:
            height, width = img.shape[0], img.shape[1]
            r = self.model.detect([img], verbose=0)[0]
            rois, masks, confidences = r['rois'], r['masks'], r['scores']
            masks = np.rollaxis(masks, 2, 0)
            for roi, mask, confidence in zip(rois, masks, confidences):
                # print(roi, mask)
                ymin, ymax, xmin, xmax = np.clip(roi[0] - 5, 0, height), np.clip(roi[2] + 5, 0, height), np.clip(roi[1] - 5, 0, width), np.clip(roi[3] + 5, 0, width)  # 5 pixel border for bigger local canvas
                m = (mask[ymin:ymax, xmin:xmax])
                bin_img = np.zeros(m.shape).astype(np.uint8)
                bin_img[m] = 1
                contour = self.buildContourPoints(bin_img)
                boxes.append(BoundBox(xmin, ymin, xmax, ymax, contour=contour, confidence=confidence))
        return boxes

class UNetSegmentation:
    def __init__(self, weights_path=None):
        if weights_path is None: 
            print('Missing weights: No path provided')
            sys.exit(-1)
        self.model = unetModel.unet()
        self.model.load_weights(weights_path)

    def predictContour(self, img=None):
        if img is None: 
            return
        else:
            orig_size = img.shape
            # img = img / 255
            img = resize(img, (256, 256))
            img = np.reshape(img, img.shape + (1, ))
            img = np.reshape(img, (1, ) + img.shape)
            result = self.model.predict(img)
            result = np.squeeze(result, axis=0)
            result = np.squeeze(result, axis=2)
            label = resize(result, orig_size)

            left_side, right_side = list(), list()
            top_side, bottom_side = list(), list()
            img = img_as_ubyte(label)
            img2 = np.ones(img.shape).astype(np.uint8)
            img2[img < 210] = 0
            for y in range(img2.shape[0]):
                try:
                    x_left, x_right = np.where(img2[y, :] == 0)[0][0], np.where(img2[y,:] == 0)[0][-1]
                    left_side.append((y, x_left))
                    right_side.append((y, x_right))
                except Exception as e:
                    continue
            for x in range(img2.shape[1]):
                try:
                    y_top, y_bottom = np.where(img2[:, x] == 0)[0][0], np.where(img2[:, x] == 0)[0][-1]
                    top_side.append((y_top, x))
                    bottom_side.append((y_bottom, x))
                except Exception as e:
                    continue
            points = [x for i, x in enumerate(left_side + list(reversed(right_side))) if (x in top_side + list(reversed(bottom_side)))]
            # print(points)
            return points
