import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import cv2

from posetwister.representation import PredictionResult, Pose
from posetwister.utils import load_image
from posetwister.visualization import add_rectangles, add_keypoints, add_poses


class YoloModel:
    def __init__(self, model_name: str, min_iou_threshold: float = 0.5):
        self.model_pose = YOLO(f"{model_name}-pose.pt")

        self.min_iou_threshold = min_iou_threshold
        self.filter_classes = [0]
        self.model_params = {"verbose": False}
        if self.filter_classes:
            self.model_params["classes"] = self.filter_classes

    def _predict_pose(self, image: np.ndarray):
        predictions = self.model_pose(image, **self.model_params)
        return predictions

    def predict(self, image: np.ndarray):
        pose_prediction = self._predict_pose(image)

        results = []
        for pp in pose_prediction:
            if len(pp) > 0:
                pose_boxes = pp.boxes.xyxy.cpu().numpy()
                pose = Pose(
                    boxes=pose_boxes,
                    keypoints=pp.keypoints.xy.cpu().numpy(),
                    conf=pp.keypoints.conf.cpu().numpy()
                )
                results.append(PredictionResult(pose))
            else:
                results.append(PredictionResult(Pose()))

        return results


if __name__ == "__main__":
    yolo_model = YoloModel("yolov8x")
    image = [load_image("../data/input/image/car.jpg"),
             load_image("../data/input/image/t_pose-0.jpg"),
             load_image("../data/input/image/ymca-0.jpg")]
    #image = [load_image("../data/input/image/ymca-0.jpg")]
    predictions = yolo_model.predict(image)

    for img, prd in zip(image, predictions):
        if prd.pose.keypoints is not None:
            img = add_rectangles(img, prd.pose, add_conf=False)
            img = add_keypoints(img, prd.pose)
            # img = add_poses(img, prd.pose)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
