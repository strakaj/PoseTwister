from posetwister.representation import PredictionResult
from posetwister.comparison import compare_pose_angels
from typing import Union, List
from utils import representation_form_json

class SessionObjective:
    def __init__(self, target: Union[PredictionResult, List[PredictionResult]], comparison_method: str,
                 after_complete: str):
        """
        :param target:
        :param comparison_method: ('angle')
        :param after_complete: ('end', 'reset')
        """
        if isinstance(target, list):
            self.target_type = "sequence"
        else:
            self.target_type = "separate"
            target = [target]
        self.target = target
        self.progress = -1
        self.comparison_method = comparison_method
        self.after_complete = after_complete

    def is_complete(self):
        if len(self.target) == self.progress + 1:
            return True
        return False

    def _compare_angels(self, ref: PredictionResult, prediction: PredictionResult):
        if ref.pose.keypoints is None or prediction.pose.keypoints is None:
            return None
        similarity = compare_pose_angels(ref.pose, prediction.pose)
        return similarity

    def __call__(self, prediction: PredictionResult):
        similarity = 0
        if self.is_complete():
            if self.after_complete == "end":
                return None
            elif self.after_complete == "reset":
                self.progress = -1

        ref_representation = self.target[self.progress + 1]
        if self.comparison_method == "angle":
            similarity = self._compare_angels(ref_representation, prediction)

        # TODO: evaluate if similar enought -> self.progress =+ 1

        return similarity


if __name__=="__main__":
    pose0 = representation_form_json("../data/ref_poses/t_pose-0.json")
    pose1 = representation_form_json("../data/ref_poses/t_pose-1.json")
    pose2 = representation_form_json("../data/ref_poses/t_pose-2.json")

    objective = SessionObjective(
        pose0, "angle", "end"
    )
    print(objective(pose1))
    print(objective(pose2))

