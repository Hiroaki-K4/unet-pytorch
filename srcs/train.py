import glob
import os

import cv2
import torch
import torch.nn as nn

from unet import UNet


def create_label_data(anno_img, output_size):
    anno_img = cv2.resize(
        anno_img, (output_size, output_size), interpolation=cv2.INTER_NEAREST
    )
    label = torch.zeros((2, output_size, output_size))
    # Check if pixel is (255, 255, 255) and update the label
    white_mask = (anno_img == [255, 255, 255]).all(axis=-1)
    label[1][white_mask] = 1  # Set channel 1 to 1 where mask is True
    label[0][~white_mask] = 1  # Set channel 0 to 1 where mask is False
    label = label.unsqueeze(0)

    return label


def train(aug_data_path, output_size, learning_rate, epochs, device):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tensor = torch.from_numpy(gray_img).float()
            tensor = (
                tensor.unsqueeze(0).unsqueeze(0).to(device)
            )  # Shape: (1, 1, 572, 572)
            pred = model.forward(tensor)

            anno_img_path = anno_paths[i]
            anno_img = cv2.imread(anno_img_path)
            label = create_label_data(anno_img, output_size)
            loss = loss_fn(pred, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        print("Epoch {0}, Loss={1}".format(epoch, round(epoch_loss / count, 5)))


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
    output_size = 388
    learning_rate = 1e-3
    epochs = 10
    train(aug_data_path, output_size, learning_rate, epochs, device)
