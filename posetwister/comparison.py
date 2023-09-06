from posetwister.representation import Pose
import numpy as np
from posetwister.visualization import KEYPOINT_NAMES
import math
kp_name_to_id = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def calculate_angle(ref_kp, prd_kp):
    x1, y1 = ref_kp
    x2, y2 = prd_kp

    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle = math.degrees(math.atan2(det, dot))
    # angle = np.degrees(np.arccos(np.dot(ref_kp, prd_kp) / (
    #          np.linalg.norm(ref_kp) * np.linalg.norm(prd_kp))))
    return angle


def compare_pose_angels(pose1: Pose, pose2: Pose, get_angle_scores: bool = False) -> dict[int: float]:
    angle_pairs = {
        "left_shoulder": "left_elbow",
        "left_elbow": "left_wrist",
        "right_shoulder": "right_elbow",
        "right_elbow": "right_wrist",
        "left_hip": "left_knee",
        "right_hip": "right_knee"
    }

    calculated_angles = {}
    similarity = {}
    perpendicular_vectors = {}

    ref_pose = pose1.keypoints[0]
    prd_pose = pose2.keypoints[0]

    for start_kp_name, end_kp_name in angle_pairs.items():
        start_kp_id = kp_name_to_id[start_kp_name]
        end_kp_id = kp_name_to_id[end_kp_name]

        # move vector to origin
        ref_kp = np.array(ref_pose[end_kp_id]) - np.array(ref_pose[start_kp_id])
        prd_kp = np.array(prd_pose[end_kp_id]) - np.array(prd_pose[start_kp_id])

        angle = calculate_angle(ref_kp, prd_kp)

        # normalize
        similarity[start_kp_name] = (180 - np.abs(angle)) / 180
        calculated_angles[start_kp_name] = angle

        # create perpendicular vectors
        sign = angle / np.abs(angle)
        ortho_prd_kp = perpendicular(prd_kp)
        #ortho_prd_kp = ortho_prd_kp / np.linalg.norm(ortho_prd_kp)  # normalize
        #print(ortho_prd_kp)
        ortho_prd_kp *= sign                                        # adjust direction
        ortho_prd_kp *= np.abs(angle) / 180                         # resize
        ortho_prd_kp = ortho_prd_kp + np.array(prd_pose[end_kp_id])  # move vector
        perpendicular_vectors[end_kp_id] = ortho_prd_kp

    if get_angle_scores:
        sim = {kp_name_to_id[name]: s for name, s in similarity.items()}
        return sim, perpendicular_vectors
    return {-1: np.mean([s for s in similarity.values()])}, perpendicular_vectors
