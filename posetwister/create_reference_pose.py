import argparse
import os.path
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from posetwister.model import YoloModel
from posetwister.predictors import DefaultImagePredictor
from posetwister.utils import save_json, load_image
from posetwister.visualization import add_keypoints


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--image_path', type=str, nargs='+')
    parser.add_argument('--output_path', type=str)

    return parser


def get_larges(prediction):
    pose = prediction.pose
    areas = []
    for b in pose.boxes:
        x0, y0, x1, y1 = b
        w, h = x1 - x0, y1 - y0
        areas.append(w * h)
    idx = np.argmax(areas)

    pose.boxes = np.array([pose.boxes[idx]])
    pose.keypoints = np.array([pose.keypoints[idx]])
    pose.conf = np.array([pose.conf[idx]])

    return prediction


def create_representation(image_path, output_path, model_name="yolov8n", thr=0.85, kp_names=["left_shoulder", "left_elbow", "right_shoulder", "right_elbow", "left_hip", "right_hip"]):
    yolo_model = YoloModel(model_name)
    image_predictor = DefaultImagePredictor(yolo_model)
    if image_path is not None:
        if isinstance(image_path, str):
            image_path = [image_path]

        for img_pth in tqdm(image_path, desc="Creating reference poses"):
            image = load_image(img_pth)
            # image = reshape_image(image, 1080)
            image_prediction = image_predictor.predict_image(image)
            if len(image_prediction[0]) == 0:
                continue
            image_prediction[0] = get_larges(image_prediction[0])
            image_prediction[0].pose.image_path = os.path.basename(img_pth)

            image_prediction[0].pose.keypoint_similarity_threshold = {k: thr for k in kp_names}

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
                cv2.imwrite(os.path.join(pred_path, os.path.basename(img_pth)), image[..., ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    paths = glob(os.path.join(args.image_path[0], "*"))
    create_representation(paths, args.output_path, model_name="yolov8x")
