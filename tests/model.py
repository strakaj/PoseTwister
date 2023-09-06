from posetwister.model import YoloModel
from posetwister.utils import load_image

if __name__=="__main__":
    yolo_model = YoloModel("yolov8n")

    image_0 = load_image("../data/input/image/car.jpg")
    image_1 = load_image("../data/input/image/t_pose-0.jpg")
    image_2 = load_image("../data/input/image/ymca-0.jpg")

    predictions_0 = yolo_model.predict(image_0)
    predictions_1 = yolo_model.predict(image_1)
    predictions_2 = yolo_model.predict(image_2)

    print(predictions_0[0].is_consistent, len(predictions_0[0]))
    print(predictions_1[0].is_consistent, len(predictions_1[0]))
    print(predictions_2[0].is_consistent, len(predictions_2[0]))