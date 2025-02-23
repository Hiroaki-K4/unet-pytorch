import os

import cv2
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
    array_3_channel[array == 255] = [0, 0, 255]
    array_3_channel[array == 0] = [255, 255, 255]

    return array_3_channel


def add_label(img, text):
    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(
        0.5, min(img.shape[1], img.shape[0]) / 500
    )  # Scale text size dynamically
    thickness = 2
    text_color = (0, 0, 0)  # Black text
    # Position text at the top-left with some padding
    text_position = (10, 30)
    # Add text to image
    cv2.putText(
        img, text, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA
    )

    return img


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
        input_img_filename = f"{i:05}.png"
        seg_img_filename = f"{i:05}_seg.png"
        image_path = os.path.join(image_dir, input_img_filename)
        pred = predict(image_path, model_path, device)
        seg_img = create_segmentation_image(pred[0])
        input_path = os.path.join(output_dir, input_img_filename)
        seg_path = os.path.join(output_dir, seg_img_filename)
        input_img = cv2.imread(image_path)
        h_input, w_input = input_img.shape[:2]
        h_output, w_output = seg_img.shape[:2]
        h_offset = (h_input - h_output) // 2
        w_offset = (w_input - w_output) // 2
        input_cropped = input_img[
            h_offset : h_offset + h_output, w_offset : w_offset + w_output
        ]
        seg_img = seg_img.astype(np.uint8)
        input_cropped = add_label(input_cropped, "Input")
        seg_img = add_label(seg_img, "Result")
        cv2.imwrite(input_path, input_cropped)
        cv2.imwrite(seg_path, seg_img)
