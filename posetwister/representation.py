from dataclasses import dataclass
import numpy as np


class PredictionResult:
    def __init__(self, seg_prediction, pose_prediction):
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

    def export_to_json(self):
        # TODO: implement
        pass


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
