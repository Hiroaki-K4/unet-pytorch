import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        # Encoder
        self.enc_0 = ConvBlock(in_channels, 64)
        self.enc_1 = ConvBlock(64, 128)
        self.enc_2 = ConvBlock(128, 256)
        self.enc_3 = ConvBlock(256, 512)
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        # Decoder
        self.dec_3 = ConvBlock(1024, 512)
        self.dec_2 = ConvBlock(512, 256)
        self.dec_1 = ConvBlock(256, 128)
        self.dec_0 = ConvBlock(128, 64)

    # def forward(self, x):
    #     return self.net(input_data)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))

    print()
    print("Check ConvBlock")
    conv_block = ConvBlock().to(device)
    sample_input_conv_block = torch.randn(1, 1, 572, 572).to(
        device
    )  # Batch size 1, 1 channel, 128x128 image
    output_conv_block = conv_block(sample_input_conv_block)
    print("Double conv: ", conv_block)
    print(
        "Double conv: Input {0} -> Output {1}".format(
            sample_input_conv_block.size(), output_conv_block.size()
        )
    )  # Should be [1, 64, 568, 568]
    print()

    # model = UNet().to(device)
    # print(model)
