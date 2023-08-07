import argparse
import os.path
from tqdm import tqdm

import cv2

from posetwister.model import YoloModel
from posetwister.predictors import DefaultImagePredictor
from posetwister.utils import save_json, load_image, reshape_image
from posetwister.visualization import add_keypoints

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--image_path', type=str, nargs='+')
    parser.add_argument('--output_path', type=str)

    return parser


def create_representation(image_path, output_path,  model_name="yolov8n"):
    yolo_model = YoloModel(model_name)
    image_predictor = DefaultImagePredictor(yolo_model)
    if image_path is not None:
        if isinstance(image_path, str):
            image_path = [image_path]

        for img_pth in tqdm(image_path, desc="Creating reference poses"):
            image = load_image(img_pth)
            #image = reshape_image(image, 1080)
            image_prediction = image_predictor.predict_image(image)

            ref_pose = image_prediction[0].pose.export()
            pose_representation = {"pose": ref_pose}
            if output_path:
                path = output_path
                if os.path.isdir(output_path):
                    path = os.path.join(path, os.path.basename(img_pth).split(".")[0] + ".json")
                save_json(pose_representation, path)

                path = output_path
                if not os.path.isdir(output_path):
                    path = os.path.dirname(path)

                pred_path = os.path.join(path, "prediction_images")
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                image = add_keypoints(image, image_prediction[0].pose)
                cv2.imwrite(os.path.join(pred_path, os.path.basename(img_pth)), image[...,::-1])



if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    create_representation(args.image_path, args.output_path, model_name="yolov8x")

