import cv2
import numpy as np


def annotate_logo(filename, output_path):
    img = cv2.imread(filename)
    anno_img = img.copy()
    mask = np.all(img == [255, 255, 255], axis=-1)
    anno_img[mask] = [0, 0, 0]
    mask = np.all(anno_img != [0, 0, 0], axis=-1)
    anno_img[mask] = [0, 0, 192]
    cv2.imwrite(output_path, anno_img)


if __name__ == "__main__":
    # This program annotates a transparent png logo file
    filename = "../resources/original/python_logo.png"
    output_path = "../resources/annotation/python_logo.png"
    annotate_logo(filename, output_path)
