import os
import cv2
from utils import reshape_image, load_image

if __name__ == "__main__":
    path = "../data/pose_templates_new/images"
    out_path = "../data/pose_templates_new/images_resized"
    file_names = os.listdir(path)

    for name in file_names:
        img_path = os.path.join(path, name)
        image = load_image(img_path)
        image = reshape_image(image, 1080)

        name = name.split(".")[0] + ".png"
        cv2.imwrite(os.path.join(out_path, name), image[...,::-1])


