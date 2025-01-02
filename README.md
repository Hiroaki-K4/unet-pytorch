# unet-pytorch

You can create local environment by running following commands.

```bash
conda create -n unet python=3.11
conda activate unet
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

```
The contracting path follows the typical architecture of a convolutional network.
It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
each followed by a rectiﬁed linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling.
```
