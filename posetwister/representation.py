from dataclasses import dataclass
import numpy as np
from copy import deepcopy


@dataclass
class Pose:
    boxes: np.ndarray = None
    keypoints: np.ndarray = None
    conf: np.ndarray = None

    image_path = None
    keypoint_similarity_threshold = None

    def is_empty(self):
        if self.boxes is None or self.keypoints is None or self.conf is None:
            return True
        return False

    def __len__(self):
        if self.is_empty():
            return -1
        if len(self.boxes) == len(self.keypoints) == len(self.conf):
            return len(self.boxes)
        else:
            return -1

    def export(self):
        data = {
            "boxes": self.boxes.tolist(),
            "keypoints": self.keypoints.tolist(),
            "conf": self.conf.tolist(),
            "image_path": self.image_path,
            "keypoint_similarity_threshold": self.keypoint_similarity_threshold
        }
        return data


class PredictionResult:
    def __init__(self, pose_prediction: Pose):
        self._pose = pose_prediction

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, new_pose: Pose):
        if new_pose.boxes is not None:
            self._pose.boxes = new_pose.boxes
        if new_pose.conf is not None:
            self._pose.conf = new_pose.conf
        if new_pose.keypoints is not None:
            self._pose.keypoints = new_pose.keypoints

    @property
    def result(self):
        return {"pose": self._pose}

    def __len__(self):
        if self._pose.is_empty():
            return 0
        return len(self._pose)

    def keep_by_idx(self, indexes: list):
        self._pose.boxes = self._pose.boxes[indexes]
        self._pose.keypoints = self._pose.keypoints[indexes]
        self._pose.conf = self._pose.conf[indexes]

    def select_by_index(self, indexes: list):
        selected_pose = Pose(
            boxes=deepcopy(self._pose.boxes[indexes]),
            keypoints=deepcopy(self._pose.keypoints[indexes]),
            conf=deepcopy(self._pose.conf[indexes]),
        )
        return PredictionResult(selected_pose)
