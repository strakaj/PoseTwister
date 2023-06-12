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

    def select_top_prediction(self):
        pass

    def after_prediction(self, frame, prediction):

        if prediction is not None:
            frame = add_rectangles(frame, prediction.segmentation, add_conf=True)
            frame = add_keypoints(frame, prediction.pose)

        if self.prediction_times:
            param = get_parameters(frame.shape)
            mov_avg_n = 12

            time = self.prediction_times[-1]
            if len(self.prediction_times) >= mov_avg_n:
                time = self.prediction_times[-mov_avg_n::]
                time = np.mean(time)
                self.prediction_times.pop(0)

            fps = np.round(1 / time, 1)

            size, _ = cv2.getTextSize(str(fps), param["font"], param["font_scale"] + 1, param["thickness"])
            frame = cv2.putText(frame, str(fps), (20, size[1] + 20), param["font"],
                                param["font_scale"] + 1, [255, 255, 255], param["thickness"], param["line"])

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    yolo_model = YoloModel("yolov8n")
    image_predictor = DefaultImagePredictor(yolo_model)
    if args.image_path is not None:
        image_predictor.predict(args.image_path)

    video_predictor = VideoPredictor(yolo_model)
    if args.video_path is not None:
        video_predictor.predict(args.video_path)
