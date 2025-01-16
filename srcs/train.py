import glob
import os

import cv2
import torch

from unet import UNet


def train(aug_data_path, device):
    img_path_list = glob.glob(os.path.join(aug_data_path, "*.png"))
    img_paths = [
        path
        for path in img_path_list
        if path.endswith(".png") and not path.endswith("_anno.png")
    ]
    anno_paths = [path for path in img_path_list if path.endswith("_anno.png")]
    img_paths.sort()
    anno_paths.sort()
    model = UNet().to(device)
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # TODO: Fix error
        model.forward(gray_img)
        input()

    # print(model)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))
    aug_data_path = "../resources/augmentation"
    train(aug_data_path, device)
