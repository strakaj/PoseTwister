from posetwister.representation import PredictionResult
from posetwister.comparison import compare_pose_angels
from typing import Union, List
from posetwister.utils import representation_form_json, reset_running_variable, exponential_filtration
from collections import defaultdict
import numpy as np


class SessionObjective:
    def __init__(self, target: Union[PredictionResult, List[PredictionResult]], comparison_method: str,
                 after_complete: str, threshold: float = 0.75, alpha: float = 0.9, in_row: int = 0, pose_image: List[np.ndarray] = []):
        """
        :param target:
        :param comparison_method: ('angle', 'multi_angle')
        :param after_complete: ('end', 'reset')
        """
        if isinstance(target, list):
            self.target_type = "sequence"
        else:
            self.target_type = "separate"
            target = [target]
        self.target = target
        self.pose_image = pose_image
        self.progress = -1
        self.comparison_method = comparison_method
        self.after_complete = after_complete
        self.threshold = threshold

        self.objective_in_row = in_row
        self.state_in_row = defaultdict(lambda: 0)

        self.alpha = alpha
        self.max_in_memory = 12
        self.similarities = defaultdict(list)

        self._wait = False
        self._pose_completed = False

    def is_complete(self):
        if len(self.target) == self.progress + 1:
            return True
        return False

    def _compare_angels(self, ref: PredictionResult, prediction: PredictionResult):
        if ref.pose.keypoints is None or prediction.pose.keypoints is None:
            return None
        if self.comparison_method == "multi_angle":
            similarity = compare_pose_angels(ref.pose, prediction.pose, get_angle_scores=True)
        else:
            similarity = compare_pose_angels(ref.pose, prediction.pose)
        return similarity

    def _check_similarity(self, similarity):
        if similarity >= self.threshold:
            return True
        return False

    @property
    def wait(self):
        return self._wait

    @wait.setter
    def wait(self, w):
        if isinstance(w, bool):
            self._wait = w

    @property
    def pose_completed(self):
        return self._pose_completed

    def __call__(self, prediction: PredictionResult):
        if self._wait:
            return None, None

        if self._pose_completed:
            self._pose_completed = False
            self.similarities = defaultdict(list)
            self.state_in_row = defaultdict(lambda: 0)
            similarity = {}

        if self.is_complete():
            if self.after_complete == "end":
                return None, None
            elif self.after_complete == "reset":
                self.progress = -1

        # get similarity between reference and predicted pose
        ref_representation = self.target[self.progress + 1]
        if self.comparison_method == "angle" or "multi_angle":
            similarity = self._compare_angels(ref_representation, prediction)

        if similarity is not None:
            for kp_id in similarity:
                # apply filtration
                if kp_id in self.similarities and len(self.similarities[kp_id]) > 0:
                    similarity[kp_id] = exponential_filtration(self.similarities[kp_id][-1], similarity[kp_id],
                                                               self.alpha)
                self.similarities[kp_id].append(similarity[kp_id])
                self.similarities[kp_id] = reset_running_variable(self.similarities[kp_id], self.max_in_memory)

                # for each keypoint check if prediction is similar enough
                if self._check_similarity(similarity[kp_id]):
                    self.state_in_row[kp_id] = np.clip(self.state_in_row[kp_id] + 1, 0, self.objective_in_row)
                    print(f"Pose {self.progress + 1} kp {kp_id}: {self.state_in_row[kp_id]}/{self.objective_in_row}")
                else:
                    # self.similarities[kp_id] = []
                    self.state_in_row[kp_id] = 0

            keypoints_completed = {kp_id: True for kp_id, sir in self.state_in_row.items() if
                                   sir >= self.objective_in_row}
            if len(self.state_in_row) == len(keypoints_completed):
                sim_text = " ".join([f"kp {k}: {s:0.2f}" for k, s in similarity.items()])
                print(f" Pose {self.progress + 1} completed: {sim_text}")  #: {similarity:0.2f}
                self.progress += 1
                self._pose_completed = True

        if self.is_complete():
            print("     Objective completed!")

        return similarity, {k: v / self.objective_in_row for k, v in self.state_in_row.items()}


if __name__ == "__main__":
    pose0 = representation_form_json("../data/ref_poses/t_pose-0.json")
    pose1 = representation_form_json("../data/ref_poses/t_pose-1.json")
    pose2 = representation_form_json("../data/ref_poses/t_pose-2.json")

    objective = SessionObjective(
        [pose0, pose2, pose0], "angle", "reset"
    )
    print(objective(pose1))
    print(objective(pose2))
    print(objective(pose1))
