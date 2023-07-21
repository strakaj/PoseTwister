import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from posetwister.model import YoloModel
from posetwister.utils import load_image
from posetwister.visualization import get_parameters

edges = [[15, 13],  # l_ankle -> l_knee
         [13, 11],  # l_knee -> l_hip
         [11, 5],  # l_hip -> l_shoulder
         [12, 14],  # r_hip -> r_knee
         [14, 16],  # r_knee -> r_ankle
         [12, 6],  # r_hip  -> r_shoulder
         [12, 11],  # r_hip, l_hip
         [3, 1],  # l_ear -> l_eye
         [1, 2],  # l_eye -> r_eye
         [1, 0],  # l_eye -> nose
         [0, 2],  # nose -> r_eye
         [2, 4],  # r_eye -> r_ear
         [9, 7],  # l_wrist -> l_elbow
         [7, 5],  # l_elbow -> l_shoulder
         [5, 6],  # l_shoulder -> r_shoulder
         [6, 8],  # r_shoulder -> r_elbow
         [8, 10]]  # r_elbow -> r_wrist


def get_alpha(size):
    h, w = size, size

    # Create radial alpha/transparency layer. 255 in centre, 0 at edge
    Y = np.linspace(-1, 1, h)[None, :] * 255
    X = np.linspace(-1, 1, w)[:, None] * 255
    alpha = np.sqrt(X ** 2 + Y ** 2)
    alpha = 255 - np.clip(0, 255, alpha)
    alpha = alpha / 255
    return np.repeat(alpha[:, :, np.newaxis], 3, axis=2)


def add_gradient(image, center, color, radius):
    x, y = center
    alpha = get_alpha(radius)
    h1, w1 = np.ceil(np.array(alpha.shape[:2]) / 2).astype(int)
    h2, w2 = np.floor(np.array(alpha.shape[:2]) / 2).astype(int)
    ah0 = 0
    aw0 = 0
    ah, aw = alpha.shape[:2]

    ih, iw = image.shape[:2]
    if y + h2 > ih:
        dif = y + h2 - ih
        ah -= dif
    if y - h1 < 0:
        ah0 += h1 - y

    if x + w2 > iw:
        dif = x + w2 - iw
        aw -= dif
    if x - w1 < 0:
        aw0 += w1 - x

    image[np.clip(y - h1, 0, None):y + h2, np.clip(x - w1, 0, None):x + w2] = \
        (image[np.clip(y - h1, 0, None):y + h2, np.clip(x - w1, 0, None):x + w2] *
         (1 - alpha[ah0:ah,aw0:aw])) + (np.array(color) * (alpha[ah0:ah, aw0:aw]))

    return image


def add_keypoints_demo(image, prediction_result, parameters):
    keypoints = prediction_result.keypoints
    confs = prediction_result.conf
    conf_thr = 0.8
    leg_points = [15, 13, 14, 16]
    hand_points = [8, 10, 7, 9]

    param = get_parameters(image.shape)
    param["radius"] = int(np.ceil(param["radius"] * parameters["radius_multiplier"]))
    param["thickness"] = int(np.ceil(param["thickness"] * parameters["line_multiplier"]))
    skip_keypoints = parameters["skip_keypoints"]

    # ra = 0.00038
    # ih, iw = image.shape[:2]
    # param["radius"] = int(np.ceil( np.sqrt( (ra*ih*iw) / np.pi) ))

    body_color = parameters["body_color"]
    hand_color = parameters["hand_color"]
    leg_color = parameters["leg_color"]

    for i, kps in enumerate(keypoints):
        conf = confs[i]
        # add skeleton
        for edge in edges:
            if edge[0] in skip_keypoints or edge[1] in skip_keypoints:
                continue
            if conf[edge[0]] < conf_thr or conf[edge[1]] < conf_thr:
                continue

            color = body_color
            if parameters["limbs_same_color"]:
                if (edge[0] in leg_points) or (edge[1] in leg_points):
                    color = leg_color
                if (edge[0] in hand_points) or (edge[1] in hand_points):
                    color = hand_color
            else:
                if (edge[0] in leg_points) and (edge[1] in leg_points):
                    color = leg_color
                if (edge[0] in hand_points) and (edge[1] in hand_points):
                    color = hand_color

            start_position = np.round(kps[edge[0]]).astype(int)
            end_position = np.round(kps[edge[1]]).astype(int)

            cv2.line(image, start_position, end_position, color, thickness=param["thickness"], lineType=cv2.LINE_AA)

        # add keypoints
        for j, kp in enumerate(kps):

            color = body_color
            if j in leg_points:
                color = leg_color
            if j in hand_points:
                color = hand_color

            if j in skip_keypoints:
                continue
            if conf[j] < conf_thr:
                continue

            x, y = np.round(kp).astype(int)

            image = add_gradient(image, [x, y], color, param["radius"] * parameters["glow1_size"])
            image = add_gradient(image, [x, y], color, param["radius"] * parameters["glow2_size"])
            image = cv2.circle(image, (x, y), radius=param["radius"], color=color, thickness=-1)

    return image


def select_top_prediction(image, prediction):
    image_center = np.array(image.shape[:2]) / 2

    boxes = prediction.segmentation.boxes
    conf = prediction.segmentation.conf

    distance_to_center = []
    for b in boxes:
        x0, y0, x1, y1 = b
        box_center = np.array([y0 + (y1 - y0) / 2, x0 + (x1 - x0) / 2])
        dist = np.linalg.norm(image_center - box_center)
        distance_to_center.append(dist)

    score = [c / d for c, d in zip(conf, distance_to_center)]
    top_box_idx = np.argmax(score)
    prediction.keep_by_idx([top_box_idx])
    return prediction


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


if __name__ == "__main__":
    colors_v0 = {"body_color": [98, 254, 153], "hand_color": [98, 254, 153], "leg_color": [98, 254, 153]}
    colors_v1 = {"body_color": [244, 211, 94], "hand_color": [98, 254, 153], "leg_color": [15, 163, 177]}
    colors_v2 = {"body_color": [244, 211, 94], "hand_color": [98, 254, 153], "leg_color": [232, 116, 97]}
    colors_v3 = {"body_color": [153, 217, 140], "hand_color": [82, 182, 154], "leg_color": [217, 237, 146]}

    colors = colors_v0
    parameters = {
        "radius_multiplier": 0.5,
        "line_multiplier": 0.5,
        "glow1_size": 8,
        "glow2_size": 6,
        "skip_keypoints": [0, 1, 2, 3, 4],
        "limbs_same_color": False
    }
    parameters.update(colors)

    iamges_path = "data/iconic_poses"

    yolo_model = YoloModel("yolov8x")
    output_path = os.path.join(iamges_path, "pose_estimation_predictions")
    os.makedirs(output_path, exist_ok=True)

    images_paths = glob(os.path.join(iamges_path, "*.*"))
    image = [reshape_image(load_image(i)) for i in images_paths]

    predictions = yolo_model.predict(image)

    for i, (img, prd) in enumerate(zip(image, predictions)):
        prd = select_top_prediction(img, prd)
        if prd is not None:
            img = add_keypoints_demo(img, prd.pose, parameters)

        cv2.imwrite(os.path.join(output_path, os.path.basename(images_paths[i])).split(".")[0] + ".png",
                    img[:, :, ::-1])
        plt.imshow(img)
        plt.show()
