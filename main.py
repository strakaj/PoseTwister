import argparse
from posetwister.predictors import DefaultImagePredictor, DefaultVideoPredictor
from posetwister.model import YoloModel
from posetwister.visualization import add_rectangles, add_keypoints, get_parameters
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

    def filter_predictions(self):
        pass

    def select_top_prediction(self, frame, prediction, debug_centers = False):
        image_center = np.array(frame.shape[:2]) / 2
        if debug_centers:
            cv2.circle(frame, (int(image_center[1]), int(image_center[0])), radius=10, color=[255,0,0], thickness=-1)

        boxes = prediction.segmentation.boxes
        conf = prediction.segmentation.conf

        distance_to_center = []
        for b in boxes:
            x0, y0, x1, y1 = b
            box_center = np.array([y0 + (y1 - y0)/2, x0 + (x1 - x0)/2])
            dist = np.linalg.norm(image_center - box_center)
            distance_to_center.append(dist)
            if debug_centers:
                cv2.circle(frame, (int(box_center[1]), int(box_center[0])), radius=10, color=[0, 255, 0], thickness=-1)

        score = [c/d for c, d in zip(conf, distance_to_center)]
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

    def after_prediction(self, frame, prediction):

        if prediction is not None:
            self.select_top_prediction(frame, prediction)
            frame = add_rectangles(frame, prediction.segmentation, add_conf=True)
            frame = add_keypoints(frame, prediction.pose)

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

    video_predictor = VideoPredictor(yolo_model)
    if args.video_path is not None:
        video_predictor.predict(args.video_path)
