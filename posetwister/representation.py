from dataclasses import dataclass
import numpy as np
from copy import deepcopy

@dataclass
class Pose:
    boxes: np.ndarray = None
    keypoints: np.ndarray = None
    conf: np.ndarray = None


@dataclass
class Segmentation:
    boxes: np.ndarray = None
    masks: np.ndarray = None
    conf: np.ndarray = None

class PredictionResult:
    def __init__(self, seg_prediction: Segmentation, pose_prediction: Pose):
        self._pose = pose_prediction
        self._segmentation = seg_prediction

    @property
    def pose(self):
        return self._pose

    @property
    def segmentation(self):
        return self._segmentation

    @pose.setter
    def pose(self, new_pose: Pose):
        if new_pose.boxes is not None:
            self._pose.boxes = new_pose.boxes
        if new_pose.conf is not None:
            self._pose.conf = new_pose.conf
        if new_pose.keypoints is not None:
            self._pose.keypoints = new_pose.keypoints

    @segmentation.setter
    def segmentation(self, new_segmentation: Segmentation):
        if new_segmentation.boxes is not None:
            self._segmentation.boxes = new_segmentation.boxes
        if new_segmentation.conf is not None:
            self._segmentation.conf = new_segmentation.conf
        if new_segmentation.masks is not None:
            self._segmentation.masks = new_segmentation.keypoints

    @property
    def result(self):
        return {"pose": self._pose, "segmentation": self._segmentation}

    def keep_by_idx(self, indexes: list):
        self._pose.boxes = self._pose.boxes[indexes]
        self._pose.keypoints = self._pose.keypoints[indexes]
        self._pose.conf = self._pose.conf[indexes]

        self._segmentation.boxes = self._segmentation.boxes[indexes]
        self._segmentation.masks = self._segmentation.masks[indexes]
        self._segmentation.conf = self._segmentation.conf[indexes]

    def select_by_index(self, indexes: list):
        selected_pose = Pose(
            boxes = deepcopy(self._pose.boxes[indexes]),
            keypoints = deepcopy(self._pose.keypoints[indexes]),
            conf = deepcopy(self._pose.conf[indexes]),
        )
        selected_segmentation = Pose(
            boxes=deepcopy(self._segmentation.boxes[indexes]),
            masks=deepcopy(self._segmentation.masks[indexes]),
            conf=deepcopy(self._segmentation.conf[indexes]),
        )
        return PredictionResult(selected_segmentation, selected_pose)

    def export_to_json(self):
        # TODO: implement
        pass
