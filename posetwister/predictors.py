import os
import time
from datetime import datetime
from typing import Union, List, Optional

import cv2
import numpy as np

from posetwister.representation import PredictionResult
from posetwister.utils import load_image, load_video


class DefaultImagePredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, images: Union[str, List[str]]):
        images_paths = images if isinstance(images, list) else list(images)
        iamges = []
        for p in images_paths:
            if not os.path.isfile(p):
                raise Exception(f"{p} is not a file.")
            image = load_image(p)
            if image is None:
                raise Exception(f"Could not load image from {p}.")
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
        self.max_var_in_memory = 12

    def reset_running_variable(self, max_in_memory):
        if len(self.prediction_times) > max_in_memory:
            self.prediction_times = self.prediction_times[-max_in_memory::]
        if len(self.predictions) > max_in_memory:
            self.predictions = self.predictions[-max_in_memory::]


    def predict(self, source: Union[str, int], output_path: Optional[str] = None):
        self.reset_running_variable(0)

        if isinstance(source, str) and not os.path.isfile(source):
            raise Exception(f"{source} is not a file.")

        if output_path is not None:
            if "." in os.path.basename(output_path):
                output_path = output_path.split(".")[0] + '.avi'
            else:
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                output_path = os.path.join(os.path.join(output_path, f"out_{timestamp}.avi"))
            if not os.path.isdir(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

        video_stream = load_video(source)
        width = int(video_stream.get(3))  # or int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(video_stream.get(4))  # or int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        if output_path is not None:
            video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 24.0, (width, height))

        if video_stream is None:
            raise Exception(f"Could not load image from {source}.")

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
            if output_path is not None:
                video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow('frame', frame[:,:,::-1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video_stream.release()
        if output_path is not None:
            video_out.release()

    def after_prediction(self, frame: np.ndarray, prediction: PredictionResult) -> np.ndarray:
        return frame
