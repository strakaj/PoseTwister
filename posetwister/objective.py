from posetwister.representation import PredictionResult
from posetwister.comparison import compare_pose_angels
from typing import Union, List
from posetwister.utils import representation_form_json,reset_running_variable, exponential_filtration

class SessionObjective:
    def __init__(self, target: Union[PredictionResult, List[PredictionResult]], comparison_method: str,
                 after_complete: str, threshold: float = 0.75, in_row: int = 0):
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
        self.threshold = threshold

        self.objective_in_row = in_row
        self.state_in_row = 0

        self.alpha = 0.9
        self.max_in_memory = 12
        self.similarities = []

    def is_complete(self):
        if len(self.target) == self.progress + 1:
            return True
        return False

    def _compare_angels(self, ref: PredictionResult, prediction: PredictionResult):
        if ref.pose.keypoints is None or prediction.pose.keypoints is None:
            return None
        similarity = compare_pose_angels(ref.pose, prediction.pose)
        return similarity

    def _check_similarity(self, similarity):
        if similarity >= self.threshold:
            return True
        return False

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

        if similarity is not None:
            if len(self.similarities) > 0:
                similarity = exponential_filtration(self.similarities[-1], similarity, self.alpha)
            self.similarities.append(similarity)
            self.similarities = reset_running_variable(self.similarities, self.max_in_memory)

        if similarity is not None and self._check_similarity(similarity):
            self.state_in_row += 1
            print(f"Pose {self.progress+1}: {self.state_in_row}/{self.objective_in_row}")
            if self.state_in_row >= self.objective_in_row:
                print(f" Pose {self.progress+1} completed: {similarity:0.2f}")
                self.progress += 1
                self.similarities = []
                self.state_in_row = 0

        else:
            self.similarities = []
            self.state_in_row = 0

        if self.is_complete():
            print("     Objective completed!")

        return similarity


if __name__=="__main__":
    pose0 = representation_form_json("../data/ref_poses/t_pose-0.json")
    pose1 = representation_form_json("../data/ref_poses/t_pose-1.json")
    pose2 = representation_form_json("../data/ref_poses/t_pose-2.json")

    objective = SessionObjective(
        [pose0, pose2, pose0], "angle", "reset"
    )
    print(objective(pose1))
    print(objective(pose2))
    print(objective(pose1))


