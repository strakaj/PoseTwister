from dataclasses import dataclass
import numpy as np
from copy import deepcopy

@dataclass
class Pose:
    boxes: np.ndarray
    keypoints: np.ndarray
    conf: np.ndarray


@dataclass
class Segmentation:
    boxes: np.ndarray
    masks: np.ndarray
    conf: np.ndarray

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
