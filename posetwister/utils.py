import cv2
import time
import numpy as np


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return image

def load_video(path, get_images=False, max_images=0):
    stream = cv2.VideoCapture(path)

    if not stream.isOpened():
        Exception("Error opening video stream or file")
        return None

    if not get_images:
        return stream

    frames = []
    frame_i = 0
    while (stream.isOpened()):

        ret, frame = stream.read()
        if ret == True:
            frames.append(frame)
            frame_i += 1
            if max_images > 0 and frame_i >= max_images:
                break
        else:
            break
    stream.release()
    return frames


def iou(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.max([box1[0], box2[0]])
    yA = np.max([box1[1], box2[1]])
    xB = np.min([box1[2], box2[2]])
    yB = np.min([box1[3], box2[3]])

    # compute the area of intersection rectangle
    inter_area = np.abs(np.max((xB - xA, 0)) * np.max((yB - yA, 0)))
    if inter_area == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = np.abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = np.abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def get_iou_mat(bb0, bb1):
    iou_mat = np.zeros([len(bb0), len(bb1)])
    for y, b0 in enumerate(bb0):
        for x, b1 in enumerate(bb1):
            iou_mat[y, x] = iou(b0, b1)
    return iou_mat


class Timer:
    def __init__(self, message=""):
        self.message = message if message else "Code"

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.toc = time.time()
        exec_time = self.toc - self.tic
        print(f"{self.message} took: {exec_time:0.3f}s to execute.")
