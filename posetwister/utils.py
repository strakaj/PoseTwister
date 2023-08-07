import cv2
import time
import json
import yaml
import numpy as np
from posetwister.representation import Pose, PredictionResult


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def representation_form_json(path):
    data = load_json(path)
    if "pose" in data:
        pose = Pose(
            boxes=data["pose"]["boxes"],
            keypoints=data["pose"]["keypoints"],
            conf=data["pose"]["conf"]
        )

    return PredictionResult(None, pose)


def load_image(path):
    iamge = cv2.imread(path)
    image = cv2.cvtColor(iamge, cv2.COLOR_BGR2RGB)
    return image


def load_video(path, get_images=False, max_images=0):
    stream = cv2.VideoCapture(path)

    if not stream.isOpened():
        raise Exception("Error opening video stream or file")

    if not get_images:
        return stream

    frames = []
    frame_i = 0
    while stream.isOpened():

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
    x0 = np.max([box1[0], box2[0]])
    y0 = np.max([box1[1], box2[1]])
    x1 = np.min([box1[2], box2[2]])
    y1 = np.min([box1[3], box2[3]])

    # compute the area of intersection rectangle
    inter_area = np.abs(np.max((x1 - x0, 0)) * np.max((y1 - y0, 0)))
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


def exponential_filtration(x0, x1, alpha: float = 0.75):
    x = None
    if x1 is None:
        return x
    if x0 is None:
        return x1

    x = alpha * x0 + (1 - alpha) * x1
    return x


def reset_running_variable(variable, max_in_memory):
    if len(variable) > max_in_memory:
        variable = variable[-max_in_memory::]
    return variable


def reshape_image(input_image, new_size=1080) -> np.ndarray:
    h, w = input_image.shape[:2]
    if h > w:
        input_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)

    org_width = input_image.shape[1]
    new_height = int(np.round((input_image.shape[0] * new_size) / org_width))
    resized_image = cv2.resize(input_image, (new_size, new_height))

    if h > w:
        resized_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return resized_image


class Timer:
    def __init__(self, message=""):
        self.message = message if message else "Code"

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.toc = time.time()
        exec_time = self.toc - self.tic
        print(f"{self.message} took: {exec_time:0.3f}s to execute.")
