import argparse
from posetwister.predictors import DefaultImagePredictor, DefaultVideoPredictor
from posetwister.model import YoloModel
from posetwister.visualization import add_rectangles, add_keypoints, get_parameters, add_masks, get_color_gradient
from posetwister.representation import Pose, Segmentation
from posetwister.objective import SessionObjective
from posetwister.utils import representation_form_json
import cv2
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--image_path', type=str, nargs='+')
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--use_camera', action='store_true')

    parser.add_argument('--output_path', type=str)
    parser.add_argument('--export_path', type=str)

    return parser


class VideoPredictor(DefaultVideoPredictor):
    def __init__(self, model, objective=None):
        super().__init__(model)
        self.objective = objective
        self.objective_score = []
        self.objective_completed_threshold = 0.75
        self.num_colors = 21
        self.colors = get_color_gradient(self.num_colors, plot=True)
        self.color_thresholds = np.linspace(0.4, self.objective_completed_threshold - (1 / (self.num_colors - 1)), self.num_colors)

        self.colors = [self.colors[0], *self.colors]
        self.color_thresholds = [0.0, *self.color_thresholds]

    def reset_running_variable(self, max_in_memory):
        super().reset_running_variable(max_in_memory)
        if len(self.objective_score) > max_in_memory:
            self.objective_score = self.objective_score[-max_in_memory::]

    def exponential_filtration(self, x0, x1, alpha: float = 0.75):
        x = None
        if x1 is None:
            return x
        if x0 is None:
            return x1

        x = alpha * x0 + (1 - alpha) * x1
        return x

    def filter_box_predictions(self, prediction):
        boxes_new = []
        if len(self.predictions) > 1:
            boxes0 = self.predictions[-2].segmentation.boxes
            boxes1 = prediction.segmentation.boxes
            for box0, box1 in zip(boxes0, boxes1):
                box_new = [self.exponential_filtration(x0, x1) for x0, x1 in zip(box0, box1)]
                boxes_new.append(box_new)

            segmentation_new = Segmentation(boxes=np.array(boxes_new))
            prediction.segmentation = segmentation_new
            self.predictions[-1].segmentation = segmentation_new

    def filter_keypoint_predictions(self, prediction):
        keypoints_new = []
        if len(self.predictions) > 1:
            keypoints0_all = self.predictions[-2].pose.keypoints
            keypoints1_all = prediction.pose.keypoints
            for keypoints0, keypoints1 in zip(keypoints0_all, keypoints1_all):
                for kp0, kp1 in zip(keypoints0, keypoints1):
                    kp_new = [self.exponential_filtration(x0, x1) for x0, x1 in zip(kp0, kp1)]
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
                            param["font_scale"] + 1, [255, 255, 255], param["thickness"], param["line"])
        return frame

    def after_prediction(self, frame, prediction):

        if len(prediction) > 0:
            self.select_top_prediction(frame, prediction)
            self.filter_box_predictions(prediction)
            self.filter_keypoint_predictions(prediction)

            if self.objective is not None:
                similarity = objective(prediction)
                if similarity is not None:
                    self.objective_score.append(similarity)
                    idx = np.max([i for i, t in enumerate(self.color_thresholds) if t < similarity])
                    color = self.colors[idx]
                    frame = add_masks(frame, prediction.segmentation, color, alpha=0.20)
                    frame = self.add_similarity(frame, f"{similarity:.2f}")

            frame = add_rectangles(frame, prediction.segmentation, add_conf=True)
            frame = add_keypoints(frame, prediction.pose)

        else:
            # if no pose detected, reset previous predictions
            self.predictions = []

        if self.prediction_times:
            frame = self.add_frame_rate(frame, mov_avg=12)

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    yolo_model = YoloModel("yolov8n")
    image_predictor = DefaultImagePredictor(yolo_model)
    if args.image_path is not None:
        image_predictions = image_predictor.predict(args.image_path)

    pose0 = representation_form_json("data/ref_poses/y-0.json")
    pose1 = representation_form_json("data/ref_poses/c-0.json")
    pose2 = representation_form_json("data/ref_poses/a-0.json")
    posea = representation_form_json("data/ref_poses/t_pose-2.json")

    objective = SessionObjective([pose0, pose2], "angle", "end")
    video_predictor = VideoPredictor(yolo_model, objective)
    video_predictor.predict(None)
    if args.video_path is not None:
        video_predictor.predict(args.video_path)
