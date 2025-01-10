# unet-pytorch

<br></br>

# U-Net architecture
U-Net is built on the architecture shown below. The height and width of the image are reduced, and when the number of channels reaches $1024$, the height and width of the image are increased little by little in the opposite direction. This structure is called U-Net architecture because it resembles the shape of **U**.

<img src="resources/reference/unet_architecture.png" width='600'>

U-Net consists mainly of the following parts.

- Convolution block
- Max pooling
- Up convolution
- Copy and crop

## Convolution block

## Max pooling
## Up convolution
## Copy and crop


```
The contracting path follows the typical architecture of a convolutional network.
It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
each followed by a rectiﬁed linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling.
```

You can create local environment by running following commands.

```bash
conda create -n unet python=3.11
conda activate unet
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
