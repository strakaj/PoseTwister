import argparse
import os.path
from datetime import datetime
from glob import glob

import cv2
import numpy as np

from posetwister.model import YoloModel
from posetwister.objective import SessionObjective
from posetwister.predictors import DefaultVideoPredictor
from posetwister.representation import Pose
from posetwister.utils import representation_form_json, exponential_filtration, iou, load_yaml, save_json
from posetwister.visualization import add_rectangles, add_keypoints, get_parameters, add_masks, get_color_gradient, \
    add_keypoint


KEYPOINT_NEIGHBORS = {5: (6, 7),  # left_shoulder
                      6: (5, 8),  # right_shoulder
                      7: (5, 9),  # left_elbow
                      8: (6, 10),  # right_elbow
                      11: (5, 13),  # left_hip
                      12: (6, 14),  # right_hip
                      }

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--config_path', default="config.yaml", type=str)

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
    cb = x0 + wb / 2

    target_width = (w1 - w2) * margin
    start = 0
    end = w1
    if target_width > wb:
        start = np.ceil(cb - target_width / 2).astype(int)
        end = np.floor(cb + target_width / 2).astype(int)
    if 0 < target_width <= wb:
        new_width = wb * margin
        start = np.ceil(cb - new_width / 2).astype(int)
        end = np.floor(cb + new_width / 2).astype(int)
    return image1[:, start:end, :]


class VideoPredictor(DefaultVideoPredictor):
    def __init__(self, config, model, objective=None):
        super().__init__(model)
        self.config = config
        self.objective = objective
        self.completion_frame = None
        self.pose_video_out = None
        self.show_center_box = config.get("show_center_box", True)

        self.top_similarity_threshold = config.get("similarity_threshold", 0.4)
        self.bot_similarity_threshold = config.get("bot_similarity_threshold", 0.75)

        # get visualization colors
        self.num_colors = config.get("num_colors", 21)
        self.colors = get_color_gradient(self.num_colors,
                                         plot=False,
                                         key_colors=config.get("key_colors", [[232, 116, 97], [244, 211, 94], [98, 254, 153]]))
        self.color_thresholds = np.linspace(self.bot_similarity_threshold,
                                            self.top_similarity_threshold - (1 / (self.num_colors - 1)),
                                            self.num_colors)
        self.colors = [self.colors[0], *self.colors]
        self.color_thresholds = [0.0, *self.color_thresholds]


        # get variables for debugging
        self.debug_values = config.get("debug_numbers_in_image", False)
        self.debug_images = {"save": config["save_debug_poses"]}
        if config["save_debug_poses"]:
            now = datetime.now()
            path = os.path.join(config["debug_images_path"], now.strftime("%d_%m_%Y-%H_%M_%S"))
            os.makedirs(path)
            self.debug_images["path"] = path

        if not self.debug_images:
            self.debug_images["save"] = False

        if self.debug_images["save"]:
            self.debug_images.update({
                "first": 0, "mid1": False, "mid2": False, "last": False
            })

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

        if self.show_center_box:
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

    def save_debug_data(self, frame, prediction, path, name):
        t = datetime.now().strftime('%H_%M_%S')
        cv2.imwrite(os.path.join(path, f"{name}-{t}.png"), frame[..., ::-1])
        ref_pose = prediction.pose.export()
        pose_representation = {"pose": ref_pose}
        save_json(pose_representation, os.path.join(path, f"{name}-{t}.json"))

    def prepare_perpendicular_vectors(self, prediction, perpendicular_vectors, scaler=30):
        kp = prediction.pose.keypoints[0]
        bbox = prediction.pose.boxes[0]

        for kp_idx in perpendicular_vectors:
            perpendicular_vectors[kp_idx] = perpendicular_vectors[kp_idx] * scaler  # resize vector
            perpendicular_vectors[kp_idx] = perpendicular_vectors[kp_idx] + kp[kp_idx]  # move vector

        return perpendicular_vectors

    def init_video(self, frame, objective):
        h, w = frame.shape[:2]
        name_idx = objective.progress + 1
        if name_idx == len(objective.pose_name):
            name_idx = 0
        pth = os.path.join(self.debug_images["path"], f"{objective.pose_name[name_idx]}.avi")
        self.pose_video_out = cv2.VideoWriter(pth, cv2.VideoWriter_fourcc(*'XVID'), 24.0, (w, h))

    def after_prediction(self, frame, prediction):
        empty_frame = frame.copy()
        if self.completion_frame is not None:
            # print(self.completion_timer)
            next_pose = False

            if self.key_pressed is not None and self.key_pressed == "n":  # cv2.waitKey(1) & 0xFF == ord('n')
                self.key_pressed = None
                next_pose = True

            if not next_pose:
                return self.completion_frame
            else:
                objective.wait = False
                self.completion_frame = None

        if self.objective is not None:
            if self.key_pressed is not None and self.key_pressed == "r":  # cv2.waitKey(1) & 0xFF == ord('r')
                self.key_pressed = None
                self.objective.progress = -1
                self.objective.reset_pose_variables()
                print("Start:", objective.pose_name[objective.progress + 1])

            if self.key_pressed is not None and self.key_pressed == "s":  # cv2.waitKey(1) & 0xFF == ord('s'):
                self.key_pressed = None
                if self.config["save_video"]:
                    self.pose_video_out.release()
                    self.pose_video_out = None

                self.objective.complete_pose(skip=True)

        if len(prediction) > 0:
            # self.select_top_prediction(frame, prediction)
            self.select_center_prediction(frame, prediction)
            self.filter_box_predictions(prediction)
            self.filter_keypoint_predictions(prediction)

            if self.objective is not None:
                if self.config["save_video"]:
                    if self.pose_video_out is None:
                        self.init_video(frame, objective)
                    self.pose_video_out.write(cv2.cvtColor(empty_frame, cv2.COLOR_RGB2BGR))

                similarity, correct_in_row, perpendicular_vectors = objective(prediction)
                # perpendicular_vectors = self.prepare_perpendicular_vectors(prediction, perpendicular_vectors)

                # do something if pose was completed
                if objective.pose_completed:
                    # stop recording video
                    if self.config["save_video"]:
                        self.pose_video_out.release()
                        self.pose_video_out = None

                    # save debug images
                    if self.debug_images["save"] and not self.debug_images["last"]:
                        self.save_debug_data(empty_frame, prediction, self.debug_images["path"],
                                             f"{objective.pose_name[objective.progress]}_3")
                        self.debug_images["last"] = True

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
                        # frame = crop(frame, pose_image, prediction.pose.boxes[0])
                        combined_image = combine_images(frame, pose_image,
                                                        target_shape=target_shape)  # combine_images_side_by_side(frame, pose_image)
                        self.completion_frame = combined_image
                        frame = combined_image
                        if self.debug_images["save"]:
                            self.save_debug_data(frame, prediction, self.debug_images["path"],
                                                 f"{objective.pose_name[objective.progress]}_completed")
                        return frame
                else:
                    if self.debug_images["save"] and self.debug_images["last"]:
                        self.debug_images.update({"first": 0, "mid1": False, "mid2": False, "last": False})

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
                        for kp_id, vc_id in zip(similarity, perpendicular_vectors):
                            kp = prediction.pose.keypoints[0][kp_id]
                            idx = np.max([i for i, t in enumerate(self.color_thresholds) if t < similarity[kp_id]])
                            color = self.colors[idx]
                            # frame = add_direction(frame, kp, perpendicular_vectors[vc_id], color)
                            neighbors = [prediction.pose.keypoints[0][neighbor] for neighbor in KEYPOINT_NEIGHBORS[kp_id]]
                            frame = add_keypoint(frame, kp, correct_in_row[kp_id], color, neighbors=neighbors, sim=similarity[kp_id])
                        if self.debug_values:
                            frame = self.add_similarity(frame, sim_text)

                    # save debug images
                    if self.debug_images["save"]:
                        if self.debug_images["first"] == 100:
                            self.save_debug_data(empty_frame, prediction, self.debug_images["path"],
                                                 f"{objective.pose_name[objective.progress + 1]}_0")
                            self.debug_images["first"] += 1
                        elif self.debug_images["first"] < 100:
                            self.debug_images["first"] += 1

                        s = [s for s in objective.state_in_row.values() if s >= objective.objective_in_row]
                        if len(s) == 2 and not self.debug_images["mid1"]:
                            self.save_debug_data(empty_frame, prediction, self.debug_images["path"],
                                                 f"{objective.pose_name[objective.progress + 1]}_1")
                            self.debug_images["mid1"] = True
                        if len(s) == 3 and not self.debug_images["mid2"]:
                            self.save_debug_data(empty_frame, prediction, self.debug_images["path"],
                                                 f"{objective.pose_name[objective.progress + 1]}_2")
                            self.debug_images["mid2"] = True

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

    # load reference poses and corresponding images
    prediction_pose_names = config.get("prediction_pose_names", [])
    if prediction_pose_names:
        ref_poses_representation_paths = [os.path.join(config["ref_pose_output_path"], name) for name in prediction_pose_names]
    else:
        ref_poses_representation_paths = glob(os.path.join(config["ref_pose_output_path"], "*.json"))
    ref_poses_representation = [representation_form_json(p) for p in ref_poses_representation_paths]

    # create objective and predictor
    objective = SessionObjective(
        config,
        ref_poses_representation,
    )

    video_predictor = VideoPredictor(
        config,
        yolo_model,
        objective,
    )

    video_predictor.predict(config["video_source"], camera_resolution=config["camera_resolution"],
                            multi=config["multi"])
