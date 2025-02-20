import glob
import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from predict import create_segmentation_image
from unet import UNet


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, anno_paths, output_size, device):
        self.img_paths = img_paths
        self.anno_paths = anno_paths
        self.output_size = output_size
        self.device = device

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray_img).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        # Load and preprocess annotation
        anno_img = cv2.imread(self.anno_paths[idx])
        label = self.create_label_data(anno_img)
        return tensor, label

    def create_label_data(self, anno_img):
        anno_img = cv2.resize(
            anno_img,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_NEAREST,
        )
        white_mask = (anno_img == [255, 255, 255]).all(axis=-1)
        label = torch.zeros((self.output_size, self.output_size), dtype=torch.float32)
        label[white_mask] = 1  # Only use foreground class (binary mask)
        return label.unsqueeze(0).to(self.device)  # Add channel dimension (C, H, W)


def train(aug_data_path, output_size, learning_rate, epochs, output_model_path, device):
    img_path_list = glob.glob(os.path.join(aug_data_path, "*.png"))
    img_paths = [
        path
        for path in img_path_list
        if path.endswith(".png") and not path.endswith("_anno.png")
    ]
    anno_paths = [path for path in img_path_list if path.endswith("_anno.png")]
    img_paths.sort()
    anno_paths.sort()

    dataset = SegmentationDataset(img_paths, anno_paths, output_size, device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))

    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        # i = 0
        for batch in dataloader:
            img_batch, label = batch  # Get batch
            pred = model(img_batch)
            # if epoch % 2 == 0:  # Save predictions every 2 epochs
            #     with torch.no_grad():
            #         pred_mask = create_segmentation_image(pred[0])
            #         cv2.imwrite(f"debug_pred_epoch_{epoch}_{i}.png", pred_mask)
            #         i += 1

            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        print("Epoch {0}, Loss={1}".format(epoch, round(epoch_loss / count, 5)))

    print("Finished training!!")
    torch.save(model.state_dict(), output_model_path)
    print("Saved model: ", output_model_path)


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
    learning_rate = 1e-4
    epochs = 5
    output_model_path = "unet.pth"
    train(aug_data_path, output_size, learning_rate, epochs, output_model_path, device)
