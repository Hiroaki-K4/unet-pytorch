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


def align_dim_between_encoder_decoder(enc_dim, dec_dim):
    start = (enc_dim - dec_dim) // 2
    end = start + dec_dim
    return start, end


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
        self.upconv_3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_3 = ConvBlock(1024, 512)
        self.upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_2 = ConvBlock(512, 256)
        self.upconv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_1 = ConvBlock(256, 128)
        self.upconv_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_0 = ConvBlock(128, 64)
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc_0 = self.enc_0(x)
        enc_1 = self.enc_1(nn.MaxPool2d(2)(enc_0))
        enc_2 = self.enc_2(nn.MaxPool2d(2)(enc_1))
        enc_3 = self.enc_3(nn.MaxPool2d(2)(enc_2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc_3))

        # Decoder path
        dec_3 = self.upconv_3(bottleneck)
        start, end = align_dim_between_encoder_decoder(enc_3.shape[2], dec_3.shape[2])
        enc_3 = enc_3[:, :, start:end, start:end]
        dec_3 = self.dec_3(torch.cat([dec_3, enc_3], dim=1))
        dec_2 = self.upconv_2(dec_3)
        start, end = align_dim_between_encoder_decoder(enc_2.shape[2], dec_2.shape[2])
        enc_2 = enc_2[:, :, start:end, start:end]
        dec_2 = self.dec_2(torch.cat([dec_2, enc_2], dim=1))
        dec_1 = self.upconv_1(dec_2)
        start, end = align_dim_between_encoder_decoder(enc_1.shape[2], dec_1.shape[2])
        enc_1 = enc_1[:, :, start:end, start:end]
        dec_1 = self.dec_1(torch.cat([dec_1, enc_1], dim=1))
        dec_0 = self.upconv_0(dec_1)
        start, end = align_dim_between_encoder_decoder(enc_0.shape[2], dec_0.shape[2])
        enc_0 = enc_0[:, :, start:end, start:end]
        dec_0 = self.dec_0(torch.cat([dec_0, enc_0], dim=1))

        return self.out_conv(dec_0)


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
    print("Conv block: ", conv_block)
    print(
        "Conv block: Input {0} -> Output {1}".format(
            sample_input_conv_block.size(), output_conv_block.size()
        )
    )  # Should be [1, 64, 568, 568]
    print()

    unet = UNet().to(device)
    output_unet = unet(sample_input_conv_block)
    print(unet)
    print(
        "UNet: Input {0} -> Output {1}".format(
            sample_input_conv_block.size(), output_unet.size()
        )
    )  # Output should be [1, 1, 388, 388]
    print(output_unet)
