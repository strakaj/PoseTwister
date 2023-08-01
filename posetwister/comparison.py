from posetwister.representation import Pose
import numpy as np
from posetwister.visualization import KEYPOINT_NAMES


def angle(vect1, vect2):
    return np.degrees(np.arccos(np.dot(vect1, vect2) / (
            np.linalg.norm(vect1) * np.linalg.norm(vect2))))


def get_angles(reference_keypoints):
    keypoints_ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

    point_lsh = reference_keypoints[keypoints_ids['left_shoulder']][:2]
    point_rsh = reference_keypoints[keypoints_ids['right_shoulder']][:2]
    point_lhip = reference_keypoints[keypoints_ids['left_hip']][:2]
    point_rhip = reference_keypoints[keypoints_ids['right_hip']][:2]
    point_lelb = reference_keypoints[keypoints_ids['left_elbow']][:2]
    point_relb = reference_keypoints[keypoints_ids['right_elbow']][:2]
    point_lwr = reference_keypoints[keypoints_ids['left_wrist']][:2]
    point_rwr = reference_keypoints[keypoints_ids['right_wrist']][:2]
    point_lkn = reference_keypoints[keypoints_ids['left_knee']][:2]
    point_rkn = reference_keypoints[keypoints_ids['right_knee']][:2]
    point_lank = reference_keypoints[keypoints_ids['left_ankle']][:2]
    point_rank = reference_keypoints[keypoints_ids['right_ankle']][:2]

    torso = np.mean([point_lsh, point_rsh], axis=0)
    hips = np.mean([point_lhip, point_rhip], axis=0)
    vect_body = [n - m for (n, m) in zip(torso, hips)]

    vect_left_arm = [n - m for (n, m) in zip(point_lelb, point_lsh)]
    vect_left_wrist = [n - m for (n, m) in zip(point_lwr, point_lelb)]
    vect_left_knee = [n - m for (n, m) in zip(point_lkn, point_lhip)]
    vect_left_ankle = [n - m for (n, m) in zip(point_lank, point_lkn)]

    vect_right_arm = [n - m for (n, m) in zip(point_relb, point_rsh)]
    vect_right_wrist = [n - m for (n, m) in zip(point_rwr, point_relb)]
    vect_right_knee = [n - m for (n, m) in zip(point_rkn, point_rhip)]
    vect_right_ankle = [n - m for (n, m) in zip(point_rank, point_rkn)]

    angle_lsh = angle(vect_left_arm, vect_body)
    angle_lelb = angle(vect_left_wrist, vect_left_arm)
    angle_lhip = angle(vect_left_knee, vect_body)
    angle_lkne = angle(vect_left_ankle, vect_left_knee)
    angle_rsh = angle(vect_right_arm, vect_body)
    angle_relb = angle(vect_right_wrist, vect_right_arm)
    angle_rhip = angle(vect_right_knee, vect_body)
    angle_rkne = angle(vect_right_ankle, vect_right_knee)
    angles = [angle_lsh, angle_rsh, angle_lelb, angle_relb, angle_lhip, angle_rhip, angle_lkne, angle_rkne]
    angles = [angle_lsh, angle_rsh, angle_lelb, angle_relb, angle_lhip, angle_rhip]
    return angles


def compare_pose_angels(pose1: Pose, pose2: Pose, get_angle_scores: bool = False) -> dict[int: float]:
    keypoints_ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

    ref_angles = get_angles(pose1.keypoints[0])
    cand_angles = get_angles(pose2.keypoints[0])

    sim = []
    for (n, m) in zip(ref_angles, cand_angles):
        if n < m:
            sim.append(n / m)
        else:
            sim.append(m / n)

    if get_angle_scores:
        sim = dict(zip(
            [
                keypoints_ids['left_shoulder'],
                keypoints_ids['right_shoulder'],
                keypoints_ids['left_elbow'],
                keypoints_ids['right_elbow'],
                keypoints_ids['left_hip'],
                keypoints_ids['right_hip']
            ], sim))
        return sim
    return {-1: np.mean(sim)}  # np.sum(np.abs([n - m for (n, m) in zip(ref_angles, cand_angles)]))
