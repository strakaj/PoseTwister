from typing import Union, List
from posetwister.utils import load_image, load_video
from posetwister.visualization import add_rectangles, add_keypoints
from posetwister.representation import PredictionResult
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


class DefaultImagePredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, images: Union[str, List[str]]):
        images_paths = images if isinstance(images, list) else list(images)
        iamges = []
        for p in images_paths:
            if not os.path.isfile(p):
                Exception(f"{p} is not a file.")
                continue
            image = load_image(p)
            if image is None:
                Exception(f"Could not load image from {p}.")
                continue
            iamges.append(image)
        predictions = self.predict_image(images)
        return predictions

    def predict_image(self, images: Union[np.ndarray, List[np.ndarray]]):
        images = images if isinstance(images, list) else [images]

        predictions = self.model.predict(images)
        return predictions


class DefaultVideoPredictor:
    def __init__(self, model):
        self.image_predictor = DefaultImagePredictor(model)
        self.prediction_times = []
        self.predictions = []
        self.running_variables = [self.prediction_times, self.predictions]
        self.max_var_in_memory = 12

    def reset_running_variable(self, max):
        for v in self.running_variables:
            if len(v) > max:
                v = []

    def predict(self, video: str):
        self.reset_running_variable(0)

        if not os.path.isfile(video):
            Exception(f"{video} is not a file.")
        out_path = video.replace("input", "output")
        out_path = out_path.split(".")[0] + '.avi'
        if not os.path.isdir(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        video_stream = load_video(video)
        width = int(video_stream.get(3))  # or int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(video_stream.get(4))  # or int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        video_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 24.0, (width, height))

        if video_stream is None:
            Exception(f"Could not load image from {video}.")

        while (video_stream.isOpened()):
            self.reset_running_variable(self.max_var_in_memory)
            tic = time.time()

            ret, frame = video_stream.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = self.image_predictor.predict_image(frame)[0]
            toc = time.time()
            self.prediction_times.append(toc - tic)
            self.predictions.append(predictions)

            frame = self.after_prediction(frame, predictions)
            video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_stream.release()
        video_out.release()

    def after_prediction(self, frame: np.ndarray, prediction: PredictionResult) -> np.ndarray:
        return frame
