from unet import UNet
import cv2
import torch
import numpy as np


def create_segmentation_image(pred):
    print(pred.shape)
    result = (pred[0] < pred[1]).int()  # Converts the boolean result to int (1 or 0)
    result *= 255
    array = result.cpu().numpy()
    array_3_channel = np.repeat(array[:, :, np.newaxis], 3, axis=2)
    print(array_3_channel)
    print(array_3_channel.shape)
    return array_3_channel


def predict(image_path, model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray_img).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 572, 572)
    tensor = tensor / 255.0
    pred = model.forward(tensor)
    for i in range(pred.shape[0]):
        seg_img = create_segmentation_image(pred[i])
        cv2.imwrite("test.png", seg_img)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))
    image_path = "../resources/augmentation/00001.png"
    model_path = "unet.pth"
    predict(image_path, model_path, device)
