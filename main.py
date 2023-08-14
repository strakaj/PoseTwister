import argparse
import os.path
import time
from glob import glob

import cv2
import numpy as np

from posetwister.create_reference_pose import create_representation
from posetwister.model import YoloModel
from posetwister.objective import SessionObjective
from posetwister.predictors import DefaultVideoPredictor
from posetwister.representation import Pose
from posetwister.utils import representation_form_json, exponential_filtration, iou, load_image, load_yaml
from posetwister.visualization import add_rectangles, add_keypoints, get_parameters, add_masks, get_color_gradient, \
    add_keypoint


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--image_path', type=str, nargs='+')
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--use_camera', action='store_true')

    parser.add_argument('--config_path', type=str)

    parser.add_argument('--output_path', type=str)
    parser.add_argument('--export_path', type=str)

    return parser


def combine_images_side_by_side(image1, image2):
    # Resize the images to have the same height
    height = max(image1.shape[0], image2.shape[0])
    width1, width2 = image1.shape[1], image2.shape[1]

    ratio1 = height / image1.shape[0]
    ratio2 = height / image2.shape[0]

    image1 = cv2.resize(image1, (int(width1 * ratio1), height))
    image2 = cv2.resize(image2, (int(width2 * ratio2), height))

    # Combine the images horizontally
    combined_image = np.hstack((image1, image2))

    return combined_image


def reshape_width(image, new_width):
    h, w = image.shape[:2]

    new_height = np.round(new_width / w * h).astype(int)

    image = cv2.resize(image, (new_width, new_height))
    return image

def reshape_height(image, new_height):
    h, w = image.shape[:2]

    new_width = np.round(new_height / h * w).astype(int)

    image = cv2.resize(image, (new_width, new_height))
    return image


def combine_images(image1, image2, target_shape):
    th, tw = target_shape
    combined = combine_images_side_by_side(image1, image2)

    """
    h1, w1 = combined.shape[:2]
    if w1/tw > h1/th:
        combined_reshaped = reshape_width(combined, tw)
    else:
        combined_reshaped = reshape_height(combined, th)
    """

    combined_reshaped = reshape_width(combined, tw)
    padet = np.zeros([th, tw, 3]).astype(np.uint8)
    h, w = combined_reshaped.shape[:2]
    pad = th - h
    padet[np.floor(pad / 2).astype(int):np.floor(pad / 2).astype(int) + h, :] = combined_reshaped
    return padet


def crop(image1, image2, bbox):
    margin = 1.1
    _, w1 = image1.shape[:2]
    _, w2 = image2.shape[:2]
    x0, _, x1, _ = bbox
    wb = x1 - x0
    cb = x0 + wb/2

    target_width = (w1 - w2) * margin
    start = 0
    end = w1
    if target_width > wb:
        start = np.ceil(cb - target_width/2).astype(int)
        end = np.floor(cb + target_width/2).astype(int)
    if 0 < target_width <= wb:
        new_width = wb * margin
        start = np.ceil(cb - new_width/2).astype(int)
        end = np.floor(cb + new_width/2).astype(int)
    return image1[:, start:end, :]


class VideoPredictor(DefaultVideoPredictor):
    def __init__(self,
                 model,
                 objective=None,
                 num_colors=21,
                 bot_similarity_threshold=0.4,
                 top_similarity_threshold=0.75,
                 show_result_time=5,
                 key_colors=[[232, 116, 97], [244, 211, 94], [98, 254, 153]],
                 debug_values=True):
        super().__init__(model)
        self.objective = objective
        self.top_similarity_threshold = top_similarity_threshold
        self.bot_similarity_threshold = bot_similarity_threshold
        self.num_colors = num_colors
        self.colors = get_color_gradient(self.num_colors, plot=False,
                                         key_colors=key_colors)
        self.color_thresholds = np.linspace(self.bot_similarity_threshold,
                                            self.top_similarity_threshold - (1 / (self.num_colors - 1)),
                                            self.num_colors)

        self.colors = [self.colors[0], *self.colors]
        self.color_thresholds = [0.0, *self.color_thresholds]

        self.completion_frame = None
        self.completion_timer = 0
        self.last_frame_time = 0
        self.show_completion_for_time = show_result_time

        self.debug_values = debug_values

    def filter_box_predictions(self, prediction):
        boxes_new = []
        if len(self.predictions) > 1:
            boxes0 = self.predictions[-2].pose.boxes
            boxes1 = prediction.pose.boxes
            for box0, box1 in zip(boxes0, boxes1):
                box_new = [exponential_filtration(x0, x1) for x0, x1 in zip(box0, box1)]
                boxes_new.append(box_new)

            pose_new = Pose(boxes=np.array(boxes_new))
            prediction.pose = pose_new
            self.predictions[-1].pose = pose_new

    def filter_keypoint_predictions(self, prediction):
        keypoints_new = []
        if len(self.predictions) > 1:
            keypoints0_all = self.predictions[-2].pose.keypoints
            keypoints1_all = prediction.pose.keypoints
            for keypoints0, keypoints1 in zip(keypoints0_all, keypoints1_all):
                for kp0, kp1 in zip(keypoints0, keypoints1):
                    kp_new = [exponential_filtration(x0, x1) for x0, x1 in zip(kp0, kp1)]
                    keypoints_new.append(kp_new)

            pose_new = Pose(keypoints=np.array([keypoints_new]))
            prediction.pose = pose_new
            self.predictions[-1].pose = pose_new

    def select_top_prediction(self, frame, prediction, debug_centers=False):
        image_center = np.array(frame.shape[:2]) / 2
        if debug_centers:
            cv2.circle(frame, (int(image_center[1]), int(image_center[0])), radius=10, color=[255, 0, 0], thickness=-1)

        boxes = prediction.segmentation.boxes
        conf = prediction.segmentation.conf

        distance_to_center = []
        for b in boxes:
            x0, y0, x1, y1 = b
            box_center = np.array([y0 + (y1 - y0) / 2, x0 + (x1 - x0) / 2])
            dist = np.linalg.norm(image_center - box_center)
            distance_to_center.append(dist)
            if debug_centers:
                cv2.circle(frame, (int(box_center[1]), int(box_center[0])), radius=10, color=[0, 255, 0], thickness=-1)

        score = [c / d for c, d in zip(conf, distance_to_center)]
        top_box_idx = np.argmax(score)
        prediction.keep_by_idx([top_box_idx])

    def select_center_prediction(self, frame, prediction, debug_centers=False):
        fh, fw = frame.shape[:2]
        bh, bw = fh * 0.9, (fh * 0.9) * 0.666
        cy, cx = np.array(frame.shape[:2]) / 2
        center_box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]).astype(int)

        # if debug_centers:
        #   cv2.rectangle(frame, center_box[:2], center_box[2:], color=[192,192,192], thickness=2)

        boxes = prediction.pose.boxes
        conf = prediction.pose.conf

        bious = []
        for b in boxes:
            biou = iou(b, center_box)
            bious.append(biou)

            if debug_centers:
                b = np.array(b).astype(int)
                cv2.rectangle(frame, b[:2], b[2:], color=[0, 255, 0], thickness=2)
                cv2.putText(frame, f"{biou:0.2f}", b[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [255, 255, 255], 2, cv2.LINE_AA)

        score = [b for c, b in zip(conf, bious)]
        top_box_idx = np.argmax(score)
        prediction.keep_by_idx([top_box_idx])

        cv2.rectangle(frame, center_box[:2], center_box[2:], color=[192, 192, 192], thickness=2)

    def add_frame_rate(self, frame, mov_avg=12):
        param = get_parameters(frame.shape)

        time = self.prediction_times[-1]
        if len(self.prediction_times) >= mov_avg:
            time = self.prediction_times[-mov_avg::]
            time = np.mean(time)
            self.prediction_times.pop(0)

        fps = np.round(1 / time, 1)
        size, _ = cv2.getTextSize(str(fps), param["font"], param["font_scale"] + 1, param["thickness"])
        frame = cv2.putText(frame, str(fps), (20, size[1] + 20), param["font"],
                            param["font_scale"] + 1, [255, 255, 255], param["thickness"], param["line"])
        return frame

    def add_similarity(self, frame, text):
        param = get_parameters(frame.shape)
        size, _ = cv2.getTextSize(str(text), param["font"], param["font_scale"] + 1, param["thickness"])
        frame = cv2.putText(frame, str(text), (20, 2 * size[1] + 20 + 10), param["font"],
                            param["font_scale"], [255, 255, 255], param["thickness"], param["line"])
        return frame

    def after_prediction(self, frame, prediction):

        if self.completion_frame is not None:
            # print(self.completion_timer)
            if self.completion_timer < self.show_completion_for_time:
                self.completion_timer += (time.time() - self.last_frame_time)
                self.last_frame_time = time.time()
                return self.completion_frame
            else:
                objective.wait = False
                self.completion_frame = None
                self.completion_timer = 0

        if len(prediction) > 0:
            # self.select_top_prediction(frame, prediction)
            self.select_center_prediction(frame, prediction)
            self.filter_box_predictions(prediction)
            self.filter_keypoint_predictions(prediction)

            if self.objective is not None:
                similarity, correct_in_row = objective(prediction)

                # do something if pose was completed
                if objective.pose_completed:
                    if objective.pose_image:
                        objective.wait = True
                        pose_image = objective.pose_image[objective.progress]
                        pose_target = objective.target[objective.progress]
                        for kp_id in similarity:
                            kp = prediction.pose.keypoints[0][kp_id]
                            idx = np.max([i for i, t in enumerate(self.color_thresholds) if t < similarity[kp_id]])
                            color = self.colors[idx]
                            frame = add_keypoint(frame, kp, correct_in_row[kp_id], color)
                            pose_image = add_keypoint(pose_image, pose_target.pose.keypoints[0][kp_id], 1,
                                                      self.colors[-1])
                        target_shape = frame.shape[:2]
                        #frame = crop(frame, pose_image, prediction.pose.boxes[0])
                        combined_image = combine_images(frame, pose_image, target_shape=target_shape) #combine_images_side_by_side(frame, pose_image)
                        self.completion_frame = combined_image
                        frame = combined_image
                        self.last_frame_time = time.time()
                        return frame

                # visualize predicted pose
                if similarity is not None:
                    if len(similarity) == 1 and -1 in similarity:
                        similarity = similarity[-1]
                        idx = np.max([i for i, t in enumerate(self.color_thresholds) if t < similarity])
                        color = self.colors[idx]
                        frame = add_masks(frame, prediction.segmentation, color, alpha=0.20)
                        if self.debug_values:
                            frame = self.add_similarity(frame, f"{similarity:.2f}")
                    else:
                        sim_text = " ".join([f"{s:0.2f}" for k, s in similarity.items()])
                        for kp_id in similarity:
                            kp = prediction.pose.keypoints[0][kp_id]
                            idx = np.max([i for i, t in enumerate(self.color_thresholds) if t < similarity[kp_id]])
                            color = self.colors[idx]
                            frame = add_keypoint(frame, kp, correct_in_row[kp_id], color)
                        if self.debug_values:
                            frame = self.add_similarity(frame, sim_text)

            if similarity is not None and len(similarity) == 1 and -1 in similarity:
                frame = add_rectangles(frame, prediction.segmentation, add_conf=True)
                frame = add_keypoints(frame, prediction.pose)

        else:
            # if no pose detected, reset previous predictions
            self.predictions = []

        if self.prediction_times and self.debug_values:
            frame = self.add_frame_rate(frame, mov_avg=12)

        # frame = cv2.resize(frame, np.array(frame.shape[:2][::-1]) * 2)
        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    config = load_yaml(args.config_path)

    yolo_model = YoloModel(config["prediction_model"])

    # create reference poses
    """
    ref_pose_image_paths = glob(os.path.join(config["ref_pose_input_path"], "*"))
    create_representation(
        ref_pose_image_paths,
        config["ref_pose_output_path"],
        model_name=config["ref_pose_model"]
    )
    """

    # load reference poses and corresponding images
    ref_poses_representation_paths = glob(os.path.join(config["ref_pose_output_path"], "*.json"))
    ref_poses_representation = [representation_form_json(p) for p in ref_poses_representation_paths]

    image_names = [os.path.basename(p).split(".")[0] for p in ref_poses_representation_paths]
    image_paths = [glob(os.path.join(config["ref_pose_input_path"], f"{name}.*"))[0] for name in image_names]
    pose_images = [load_image(p) for p in image_paths]

    # create objective and predictor
    objective = SessionObjective(
        ref_poses_representation,
        config["comparison_method"],
        config["after_complete"],
        in_row=config["poses_in_row"],
        threshold=config["similarity_threshold"],
        # pose_image=pose_images
    )

    video_predictor = VideoPredictor(
        yolo_model,
        objective,
        num_colors=config["number_of_colors"],
        bot_similarity_threshold=config["bot_similarity_threshold"],
        top_similarity_threshold=config["similarity_threshold"],
        show_result_time=config["show_result_time"],
        key_colors=config["key_colors"],
        debug_values=config["debug_numbers_in_image"]
    )

    # camera source
    video_predictor.predict(config["video_source"], camera_resolution=config["camera_resolution"])
