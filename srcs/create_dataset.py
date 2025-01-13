import glob
import os

import cv2
import numpy as np


def augmentation(ori_img_dir, output_size, aug_data_num):
    ori_img_list = glob.glob(os.path.join(ori_img_dir, "*.png"))
    min_scale = 0.05
    max_scale = 0.3
    combined_img = np.zeros((output_size, output_size))
    for ori_img_path in ori_img_list:
        img = cv2.imread(ori_img_path)
        h, w, c = img.shape
        anno_img_path = ori_img_path.replace("original", "annotation")
        if os.path.exists(anno_img_path):
            anno_img = cv2.imread(anno_img_path)
        else:
            anno_img = np.zeros_like(img)
        if max(h, w) > output_size:
            ratio = output_size / max(h, w)
            img = cv2.resize(
                img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
            )
            anno_img = cv2.resize(
                anno_img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
            )
        # Create combined image


if __name__ == "__main__":
    ori_img_dir = "../resources/original"
    output_size = 572
    aug_data_num = 10
    augmentation(ori_img_dir, output_size, aug_data_num)
