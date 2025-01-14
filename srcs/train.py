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
    print(img_paths)
    print(anno_paths)
    # TODO: Create label data

    # model = UNet().to(device)
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
