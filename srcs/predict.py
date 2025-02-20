import cv2
import os
import numpy as np
import torch

from unet import UNet


def create_segmentation_image(pred):
    result = (torch.sigmoid(pred) > 0.5).int()
    result *= 255
    if result.shape[0] == 1:
        result = result.squeeze(0)
    array = result.cpu().numpy()
    array_3_channel = np.repeat(array[:, :, np.newaxis], 3, axis=2)

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

    return pred


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))
    model_path = "unet.pth"
    image_dir = "../resources/augmentation"
    output_dir = "../resources/output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    img_num = 5
    for i in range(img_num):
        img_filename = f"{i:05}.png"
        image_path = os.path.join(image_dir, img_filename)
        pred = predict(image_path, model_path, device)
        seg_img = create_segmentation_image(pred[0])
        output_path = os.path.join(output_dir, img_filename)
        input_img = cv2.imread(image_path)
        seg_img = cv2.resize(
            seg_img,
            (input_img.shape[1], input_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        seg_img = seg_img.astype(np.uint8)
        blended = cv2.addWeighted(input_img, 0.7, seg_img, 0.3, 0)
        cv2.imwrite(output_path, blended)
