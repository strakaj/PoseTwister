# PoseTwister

## Installation

### Create environment:
```shell
conda create --name posetwister python=3.9
```

### Install dependencies:
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## New pose
1. Download the image in `data/pose_templates/images`
2. Run:
```shell
python posetwister/create_reference_pose.py --image_path data/pose_templates/images/new_pose.jpg --output_path data/pose_templates/predictions
```
3. You can check predicted keypoints in `data/pose_templates/prediction/prediction_images`
4. Adjust the difficulty of individual keypoints in generated file `data/pose_templates/prediction/new_pose.json` under `keypoint_similarity_threshold`. One means a perfect match is expected.
5. Add `new_pose.json` in `config.yaml` in `prediction_pose_names` list.
Optional: Add description in `data/pose_templates/texts.json`