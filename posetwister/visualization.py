import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

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
         (1 - alpha[ah0:ah, aw0:aw])) + (np.array(color) * (alpha[ah0:ah, aw0:aw]))

    return image


def add_keypoint(image, keypoint, in_row_norm, color, neighbors=None, sim=0.5):
    in_row_norm = np.abs(in_row_norm - 1)
    parameters = {
        "radius_multiplier": 0.5,
        "line_multiplier": 0.5,
        "glow1_size": 8,
        "glow2_size": 6,
    }

    param = get_parameters(image.shape)
    param["radius"] = int(np.ceil(param["radius"] * parameters["radius_multiplier"]))
    param["thickness"] = int(np.ceil(param["thickness"] * parameters["line_multiplier"]))

    x, y = np.round(keypoint).astype(int)

    if in_row_norm == 0:
        image = add_gradient(image, [x, y], color, param["radius"] * parameters["glow1_size"])
        image = add_gradient(image, [x, y], color, param["radius"] * parameters["glow2_size"])
    image = cv2.circle(image, (x, y), radius=param["radius"], color=color, thickness=-1)

    # if sim > 0.6:
    #     sim = 0.6
    # indicator_circle_radius = np.floor(24 * in_row_norm).astype(int)
    indicator_circle_radius = np.floor(24 * (1 - sim) * 1.2 + 18 * in_row_norm).astype(int)
    image = cv2.circle(image, (x, y), radius=indicator_circle_radius, color=color, thickness=param["thickness"])

    if neighbors is not None:
        for neighbor in neighbors:
            nb_vector = neighbor - (x, y)
            nb_normed = nb_vector/np.linalg.norm(nb_vector)
            nb_sized = nb_normed*30*1.5
            image = cv2.line(image, (x, y), np.round((x,y)+nb_sized).astype(int), color, 3)

    return image


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


def add_poses(image, prediction_result):
    keypoints = prediction_result.keypoints
    param = get_parameters(image.shape)

    colors = np.array([plt.cm.Set1(i / len(keypoints)) for i in range(len(keypoints))]) * 255

    pose_lines = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                  (6, 12), (12, 14), (14, 16)]

    for i, kps in enumerate(keypoints):
        for line in pose_lines:
            start_point = (round(kps[line[0], 0]), round(kps[line[0], 1]))
            end_point = (round(kps[line[1], 0]), round(kps[line[1], 1]))
            image = cv2.line(image, start_point, end_point, color=colors[i], thickness=param["thickness"])

    return image


def add_direction(frame, kp, vc, color):
    param = get_parameters(frame.shape)
    param["thickness"] = int(np.ceil(param["thickness"]))
    kp = np.round(kp).astype(int)
    vc = np.round(vc).astype(int)
    frame = cv2.line(frame, kp, vc, color, thickness=param["thickness"], lineType=cv2.LINE_AA)
    return frame

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

        r = np.round(np.linspace(c1[0], c2[0], num=num_between + 2))[1:-1]
        g = np.round(np.linspace(c1[1], c2[1], num=num_between + 2))[1:-1]
        b = np.round(np.linspace(c1[2], c2[2], num=num_between + 2))[1:-1]

        c = list(zip(r, g, b))
        colors.extend(c)
    colors.append(key_colors[-1])
    colors = [[int(v) for v in c] for c in colors]

    if plot:
        plt.imshow(np.array([colors]))
        plt.show()

    return colors


if __name__ == "__main__":
    colors = get_color_gradient(plot=True)
    n = 9
    thresholds = (np.linspace(0, 0.75 - (1 / (n - 1)), n))
    similarity = 0.3
    idx = 0
    for i, t in enumerate(thresholds):
        if t > similarity:
            break
        idx = i

    print(thresholds, idx, np.max([i for i, t in enumerate(thresholds) if t < similarity]))
