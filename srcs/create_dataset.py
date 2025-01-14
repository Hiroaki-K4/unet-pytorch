import glob
import os

import cv2
import numpy as np


def augmentation(ori_img_dir, output_dir, output_size, aug_data_num):
    ori_img_list = glob.glob(os.path.join(ori_img_dir, "*.png"))
    min_scale = 0.1
    max_scale = 0.5
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for i in range(aug_data_num):
        combined_img = np.ones((output_size, output_size, 3)) * 255
        combined_anno_img = np.zeros((output_size, output_size, 3))
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
            resize_ratio = np.random.uniform(min_scale, max_scale)
            img = cv2.resize(
                img,
                None,
                fx=resize_ratio,
                fy=resize_ratio,
                interpolation=cv2.INTER_NEAREST,
            )
            anno_img = cv2.resize(
                anno_img,
                None,
                fx=resize_ratio,
                fy=resize_ratio,
                interpolation=cv2.INTER_NEAREST,
            )
            max_start_y = output_size - img.shape[0]
            max_start_x = output_size - img.shape[1]
            start_y = np.random.randint(0, max_start_y)
            start_x = np.random.randint(0, max_start_x)
            combined_img[
                start_y : start_y + img.shape[0], start_x : start_x + img.shape[1]
            ] = img
            combined_anno_img[
                start_y : start_y + img.shape[0], start_x : start_x + img.shape[1]
            ] = anno_img

        img_filename = f"{i:05}.png"
        anno_filename = f"{i:05}_anno.png"
        output_img_path = os.path.join(output_dir, img_filename)
        output_anno_path = os.path.join(output_dir, anno_filename)
        cv2.imwrite(output_img_path, combined_img)
        cv2.imwrite(output_anno_path, combined_anno_img)


if __name__ == "__main__":
    ori_img_dir = "../resources/original"
    output_dir = "../resources/augmentation"
    output_size = 572
    aug_data_num = 100
    augmentation(ori_img_dir, output_dir, output_size, aug_data_num)
