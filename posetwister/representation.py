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