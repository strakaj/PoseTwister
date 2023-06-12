import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from posetwister.representation import PredictionResult, Pose, Segmentation
from posetwister.utils import load_image, get_iou_mat
from posetwister.visualization import add_rectangles, add_keypoints


class YoloModel:
    def __init__(self, model_name: str, min_iou_threshold: float = 0.5):
        self.model_pose = YOLO(f"{model_name}-pose.pt")
        self.model_seg = YOLO(f"{model_name}-seg.pt")

        self.min_iou_threshold = min_iou_threshold
        self.filter_classes = [0]
        self.model_params = {"verbose": False}
        if self.filter_classes:
            self.model_params["classes"] = self.filter_classes

    def _predict_pose(self, image: np.ndarray):
        predictions = self.model_pose(image, **self.model_params)
        return predictions

    def _predict_seg(self, image: np.ndarray):
        predictions = self.model_seg(image, **self.model_params)
        return predictions

    def predict(self, image: np.ndarray):
        pose_prediction = self._predict_pose(image)
        seg_prediction = self._predict_seg(image)

        results = []
        for pp, sp in zip(pose_prediction, seg_prediction):
            if len(pp) == 0 or len(sp) == 0:
                results.append(None)
            else:
                spr, ppr = self.crete_representation(sp, pp)
                results.append(PredictionResult(spr, ppr))

        return results

    def crete_representation(self, seg_prediction, pose_prediction):
        pose_boxes = pose_prediction.boxes.xyxy.cpu().numpy()
        seg_boxes = seg_prediction.boxes.xyxy.cpu().numpy()

        iou_mat = get_iou_mat(seg_boxes, pose_boxes)
        iou_mat = np.where(iou_mat > self.min_iou_threshold, iou_mat, 0)

        seg_ind, pose_ind = linear_sum_assignment(iou_mat, maximize=True)

        idx_to_keep = []
        for i, (s, p) in enumerate(zip(seg_ind, pose_ind)):
            if iou_mat[s, p] > 0:
                idx_to_keep.append(i)
        seg_ind = seg_ind[idx_to_keep]
        pose_ind = pose_ind[idx_to_keep]

        pose = Pose(
            boxes=pose_boxes[pose_ind],
            keypoints=pose_prediction.keypoints.xy.cpu().numpy()[pose_ind],
            conf=pose_prediction.keypoints.conf.cpu().numpy()[pose_ind]
        )

        segmentation = Segmentation(
            boxes=seg_boxes[seg_ind],
            masks=np.array([cv2.resize(msk, seg_prediction.masks.orig_shape[::-1]) for msk in
                            seg_prediction.masks.data.cpu().numpy()[seg_ind]]),
            conf=seg_prediction.boxes.conf.cpu().numpy()[seg_ind]
        )

        return segmentation, pose


if __name__ == "__main__":
    yolo_model = YoloModel("yolov8x")
    image = [load_image("../data/input/image/car.jpg"),
             load_image("../data/input/image/t_pose-0.jpg"),
             load_image("../data/input/image/ymca-0.jpg")]
    predictions = yolo_model.predict(image)

    for img, prd in zip(image, predictions):
        if prd is not None:
            img = add_rectangles(img, prd.segmentation, add_conf=True)
            img = add_keypoints(img, prd.pose)
        plt.imshow(img)
        plt.show()
