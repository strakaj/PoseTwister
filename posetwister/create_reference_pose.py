import argparse
import os.path

from posetwister.model import YoloModel
from posetwister.predictors import DefaultImagePredictor
from posetwister.utils import save_json


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--image_path', type=str, nargs='+')
    parser.add_argument('--output_path', type=str)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    yolo_model = YoloModel("yolov8n")
    image_predictor = DefaultImagePredictor(yolo_model)
    if args.image_path is not None:
        image_predictions = image_predictor.predict(args.image_path)

        for i in range(len(image_predictions)):
            ref_pose = image_predictions[i].pose.export()
            #TODO: add segmentation
            pose_representation = {"pose": ref_pose}
            if args.output_path:
                path = args.output_path
                if os.path.isdir(args.output_path):
                    path = os.path.join(path, os.path.basename(args.image_path[i]).split(".")[0] + ".json")
                save_json(pose_representation, path)
