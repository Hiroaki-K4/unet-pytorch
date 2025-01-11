import torch

from unet import UNet


def train(device):
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
    train(device)
