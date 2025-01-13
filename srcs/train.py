import torch
import cv2

from unet import UNet


def create_label_data(anno_data_path):
    print()


def train(input_data_path, anno_data_path, device):
    img = cv2.imread(input_data_path)
    print(img.shape)
    input()
    create_label_data(anno_data_path)
    model = UNet().to(device)
    print(model)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))
    input_data_path = "../resources/original/python.png"
    anno_data_path = "../resources/annotation/python.png"
    train(input_data_path, anno_data_path, device)
