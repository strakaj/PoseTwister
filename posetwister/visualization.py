import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def get_parameters(image_size):
    ih, iw, _ = image_size

    thickness = 2
    font_scale = 1
    radius = 9
    font = cv2.FONT_HERSHEY_COMPLEX
    line = cv2.LINE_AA
    if iw * ih <= 640 * 360:
        scale = 0.8
    elif iw * ih <= 720 * 480:
        scale = 1
    elif iw * ih <= 1080 * 720:
        scale = 1.2
    elif iw * ih <= 1920 * 1080:
        scale = 1.4
    else:
        scale = 1.6
    return {"thickness": int(np.ceil(thickness * scale)), "font_scale": font_scale * scale,
            "radius": int(np.ceil(radius * scale)), "font": font, "line": line}


def add_rectangles(image, prediction_result, color=None, add_conf=False):
    boxes = prediction_result.boxes
    conf = prediction_result.conf

    colors = np.array([cm.tab20(i / len(boxes)) for i in range(len(boxes))]) * 255
    param = get_parameters(image.shape)

    for i, bb in enumerate(boxes):
        bb = np.round(bb).astype(int)
        start_point = bb[:2]
        end_point = bb[2:]
        color = color if color is not None else colors[i]

        image = cv2.rectangle(image, start_point, end_point, color, param["thickness"])
        if add_conf:
            text = f"{conf[i]:.3f}"
            size, _ = cv2.getTextSize(text, param["font"], param["font_scale"], param["thickness"])
            start_point2 = start_point - np.array([0, size[1]])
            end_point2 = start_point + np.array([size[0], 0])
            image = cv2.rectangle(image, start_point2, end_point2, color, -1)
            image = cv2.putText(image, text, start_point, param["font"],
                                param["font_scale"], [255, 255, 255], param["thickness"], param["line"])

    return image


def add_keypoints(image, prediction_result):
    keypoints = prediction_result.keypoints
    param = get_parameters(image.shape)

    colors = np.array([plt.cm.tab20(i / len(KEYPOINT_NAMES)) for i in range(len(KEYPOINT_NAMES))]) * 255

    for i, kps in enumerate(keypoints):
        for j, kp in enumerate(kps):
            x, y = np.round(kp).astype(int)
            image = cv2.circle(image, (x, y), radius=param["radius"], color=colors[j], thickness=-1)

    return image


def add_masks(image, prediction_result, color=[255, 0, 0], alpha=0.8):
    masks = prediction_result.masks
    size = image.shape
    for mask in masks:
        image[mask > 0] = (image[mask > 0] * alpha) + (np.array(color) * (1 - alpha))

    return image


def get_color_gradient(num_colors=9, key_colors=[[168, 50, 50], [214, 101, 26], [50, 168, 82]], plot=False):
    num_between = int((num_colors - len(key_colors)) / (len(key_colors) - 1))

    colors = []
    for i in range(1, len(key_colors)):
        colors.append(key_colors[i - 1])
        c1 = key_colors[i - 1]
        c2 = key_colors[i]

        r = np.round(np.linspace(c1[0], c2[0], num=num_between + 2)).astype(int)[1:-1]
        g = np.round(np.linspace(c1[1], c2[1], num=num_between + 2)).astype(int)[1:-1]
        b = np.round(np.linspace(c1[2], c2[2], num=num_between + 2)).astype(int)[1:-1]

        c = list(zip(r, g, b))
        colors.extend(c)
    colors.append(key_colors[-1])

    if plot:
        plt.imshow(np.array([colors]))
        plt.show()

    return colors


if __name__ == "__main__":
    colors = get_color_gradient(plot=True)
    n = 9
    thresholds = (np.linspace(0, 0.75-(1/(n-1)), n))
    similarity = 0.3
    idx = 0
    for i, t in enumerate(thresholds):
        if t > similarity:
            break
        idx = i

    print(thresholds, idx, np.max([i for i, t in enumerate(thresholds) if t < similarity]))


