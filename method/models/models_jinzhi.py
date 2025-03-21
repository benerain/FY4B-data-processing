# import numpy as np
# import torch
# import torch.nn as nn


# # Utility functions for convolution
# def conv3x3(in_channels, out_channels):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


# def conv_transpose3x3(in_channels, out_channels, stride=2):
#     """3x3 transposed convolution"""
#     return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)


# # Residual Block
# class ResidualBlock(nn.Module):
#     """
#     Residual Block with Convolution, BatchNorm, and ReLU
#     """

#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(channels, channels)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
#         self.conv2 = conv3x3(channels, channels)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity
#         out = self.relu(out)

#         return out


# # Encoder network
# class Encoder(nn.Module):
#     """
#     Encoder Network with Residual Blocks
#     """

#     def __init__(self, n_channels, layer_channels, n_residual_blocks):
#         """
#         :param n_channels: Input channels
#         :param layer_channels: List of channels for each layer
#         :param n_residual_blocks: Number of residual blocks per layer
#         """
#         super(Encoder, self).__init__()
#         layers = []
#         in_channels = n_channels
#         for out_channels in layer_channels:
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
#             layers.extend([ResidualBlock(out_channels) for _ in range(n_residual_blocks)])
#             in_channels = out_channels
#         self.layers = nn.Sequential(*layers)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.layers(x)
#         return self.flatten(x)


# # Decoder network
# class Decoder(nn.Module):
#     """
#     Decoder Network with Transposed Convolutions and Residual Blocks
#     """

#     def __init__(self, n_channels, layer_channels, unflat_size, n_residual_blocks):
#         """
#         :param n_channels: Output channels
#         :param layer_channels: List of channels for each layer in reverse
#         :param unflat_size: Size to unflatten the latent vector
#         :param n_residual_blocks: Number of residual blocks per layer
#         """
#         super(Decoder, self).__init__()
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflat_size)
#         layers = []
#         in_channels = layer_channels[0]
#         layer_channels = layer_channels + [layer_channels[-1]]
#         for out_channels in layer_channels[1:]:
#             layers.extend([ResidualBlock(in_channels) for _ in range(n_residual_blocks)])
#             layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
#             in_channels = out_channels
#         self.layers = nn.Sequential(*layers)
#         self.final_layer = nn.Conv2d(layer_channels[-1], n_channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = self.unflatten(x)
#         x = self.layers(x)
#         x = self.final_layer(x)
#         return x


# # Complete AutoEncoder
# class ConvAutoEncoder(nn.Module):
#     """
#     Convolutional Auto-Encoder
#     """

#     def __init__(self, n_channels=3, input_grid_size=128, latent_dim=256, layer_channels=None, n_residual_blocks=2):
#         """
#         :param n_channels: Input and output channels
#         :param input_grid_size: Grid size of the input image
#         :param latent_dim: Latent space dimension
#         :param layer_channels: List of channels for each layer
#         :param n_residual_blocks: Number of residual blocks per layer
#         """
#         super(ConvAutoEncoder, self).__init__()

#         if layer_channels is None:
#             layer_channels = [64, 128, 256]

#         self.input_grid_size = input_grid_size
#         self.latent_dim = latent_dim

#         # Encoder
#         self.encoder = Encoder(n_channels, layer_channels, n_residual_blocks)

#         # Decoder
#         unflat_size = (layer_channels[-1], input_grid_size // (2 ** len(layer_channels)),
#                        input_grid_size // (2 ** len(layer_channels)))
#         self.decoder = Decoder(n_channels, list(reversed(layer_channels)), unflat_size, n_residual_blocks)

#     def forward(self, x):
#         encoded = self.encode(x)
#         decoded = self.decode(encoded)
#         return encoded, decoded

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, x):
#         return self.decoder(x)


# # Testing
# if __name__ == "__main__":
#     n_channels = 3
#     input_grid_size = 128
#     latent_dim = 4
#     layer_channels = [64, 128, 256]
#     n_residual_blocks = 2

#     model = ConvAutoEncoder(n_channels=n_channels, input_grid_size=input_grid_size,
#                             latent_dim=latent_dim, layer_channels=layer_channels,
#                             n_residual_blocks=n_residual_blocks)

#     dummy_input = torch.randn(1, n_channels, input_grid_size, input_grid_size)
#     encoded, decoded = model(dummy_input)
#     print(f"Input shape: {dummy_input.shape}, Encoded shape: {encoded.shape}, Decoded shape: {decoded.shape}")



# # 示例输入输出

# # 假设：
# # 	•	输入为 3 通道的 RGB 图像，大小为 128x128。
# # 	•	layer_channels = [64, 128, 256]，n_residual_blocks = 2。

# # 网络结构：

# # 	1.	第一层：输入 3x128x128 → 卷积 64x64x64 → BatchNorm → LeakyReLU → 2 个残差块。
# # 	2.	第二层：输入 64x64x64 → 卷积 128x32x32 → BatchNorm → LeakyReLU → 2 个残差块。
# # 	3.	第三层：输入 128x32x32 → 卷积 256x16x16 → BatchNorm → LeakyReLU → 2 个残差块。
# # 	4.	展平：256x16x16 → 展平为一维向量。

# # 输出：

# # 展平后的向量大小为 256 * 16 * 16 = 65536。


import numpy as np
import torch
import torch.nn as nn

# -------------------------- #
# 1. 一些工具函数
# -------------------------- #
def conv3x3(in_channels, out_channels):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

def conv_transpose3x3(in_channels, out_channels, stride=2):
    """3x3 transposed convolution"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                              padding=1, output_padding=1, bias=False)


# -------------------------- #
# 2. 残差块 (Residual Block)
# -------------------------- #
class ResidualBlock(nn.Module):
    """
    Residual Block with Convolution, BatchNorm, and LeakyReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += identity
        out = self.relu(out)
        return out


# -------------------------- #
# 3. 编码器 (Encoder)
# -------------------------- #
class Encoder(nn.Module):
    """
    Encoder Network with multiple Conv layers and Residual Blocks
    """
    def __init__(self, n_channels, layer_channels, n_residual_blocks):
        """
        :param n_channels: 输入图像的通道数
        :param layer_channels: 每层输出通道数的列表
        :param n_residual_blocks: 每层叠加的残差块数量
        """
        super(Encoder, self).__init__()
        layers = []
        in_channels = n_channels
        for out_channels in layer_channels:
            # 主卷积层，stride=2 以减小特征图空间尺寸
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                    stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

            # 添加 n_residual_blocks 个残差块
            for _ in range(n_residual_blocks):
                layers.append(ResidualBlock(out_channels))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 输出 shape: [batch_size, layer_channels[-1], H', W']
        x = self.layers(x)
        # 展平后 shape: [batch_size, layer_channels[-1] * H' * W']
        return self.flatten(x)


# -------------------------- #
# 4. 解码器 (Decoder)
# -------------------------- #
class Decoder(nn.Module):
    """
    Decoder Network with Transposed Convolutions and Residual Blocks
    """
    def __init__(self, n_channels, layer_channels, unflat_size, n_residual_blocks):
        """
        :param n_channels: 输出图像的通道数
        :param layer_channels: 逆向的通道列表 (encoder 的反转)
        :param unflat_size: 解码器输入时 unflatten 的尺寸 (C, H, W)
        :param n_residual_blocks: 每层叠加的残差块数量
        """
        super(Decoder, self).__init__()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflat_size)

        layers = []
        in_channels = layer_channels[0]

        # 逐层构建反卷积 + 残差块
        # 因为是 reversed 的列表, 所以把它往后多补一层,
        # 以匹配 encoder 的分层数
        extended_channels = layer_channels + [layer_channels[-1]]

        for out_channels in extended_channels[1:]:
            # 先加 n_residual_blocks 个残差块
            for _ in range(n_residual_blocks):
                layers.append(ResidualBlock(in_channels))

            # 反卷积 (转置卷积)，stride=2 用于空间上的上采样
            layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=3, stride=2, 
                                             padding=1, output_padding=1, 
                                             bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

            in_channels = out_channels

        # 最后的 3x3 卷积，用于恢复到 n_channels 的输出
        self.final_layer = nn.Conv2d(layer_channels[-1], n_channels,
                                     kernel_size=3, padding=1)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 先将一维向量恢复为 (C, H, W) 的特征图
        x = self.unflatten(x)
        x = self.layers(x)
        x = self.final_layer(x)
        return x


# -------------------------- #
# 5. 完整的卷积自编码器 (ConvAutoEncoder)
# -------------------------- #
class ConvAutoEncoder(nn.Module):
    """
    Convolutional Auto-Encoder with a latent space mapping
    """
    def __init__(self, n_channels=3, input_grid_size=256, latent_dim=512,
                 layer_channels=None, n_residual_blocks=2):
        """
        :param n_channels: 输入与输出的通道数
        :param input_grid_size: 输入图像的高宽
        :param latent_dim: 潜在空间维度
        :param layer_channels: 每层输出通道数
        :param n_residual_blocks: 每层的残差块数量
        """
        super(ConvAutoEncoder, self).__init__()

        if layer_channels is None:
            layer_channels = [64, 128, 256]

        self.input_grid_size = input_grid_size
        self.latent_dim = latent_dim

        # -------------------- 编码器 -------------------- #
        self.encoder = Encoder(n_channels, layer_channels, n_residual_blocks)

        # 计算编码器输出的特征图大小 (最后一层的通道数, H', W')
        # 每层 stride=2, 共 len(layer_channels) 层
        # 因此 H' = W' = input_grid_size // 2^len(layer_channels)
        with torch.no_grad():
            dummy_input = torch.randn(1, n_channels, input_grid_size, input_grid_size)
            dummy_output = self.encoder(dummy_input)
            self.flat_size = dummy_output.shape[1]
            print(f"动态计算的展平维度: {self.flat_size}")  # 调试输出


        self.latent_space = nn.Linear(self.flat_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)




        # self.flat_size = final_channels * final_spatial * final_spatial

        # # 将编码器输出映射到 latent_dim
        # self.latent_space = nn.Linear(self.flat_size, latent_dim)
        # # 将 latent_dim 映射回 flat_size
        # self.decoder_input = nn.Linear(latent_dim, self.flat_size)

        # -------------------- 解码器 -------------------- #
       

        final_spatial = input_grid_size // (2 ** len(layer_channels))
        final_channels = layer_channels[-1]

        unflat_size = (final_channels, final_spatial, final_spatial)
        self.decoder = Decoder(n_channels, list(reversed(layer_channels)),
                               unflat_size, n_residual_blocks)

    def forward(self, x):
        # 编码 -> 潜在空间 -> 解码
        z = self.encode(x)
        reconstructed = self.decode(z)
        return z, reconstructed

    def encode(self, x):
        # 1. 卷积编码器输出
        features = self.encoder(x)                 # [batch_size, flat_size]
        # 2. 全连接层，得到 latent_dim 大小的向量
        z = self.latent_space(features)            # [batch_size, latent_dim]
        return z

    def decode(self, z):
        # 1. 将 latent_dim 映射回 flat_size
        x = self.decoder_input(z)                  # [batch_size, flat_size]
        # 2. 通过解码器反卷积和残差块
        out = self.decoder(x)                      # [batch_size, n_channels, H, W]
        return out


# -------------------------- #
# 6. 测试 (Testing)
# -------------------------- #
if __name__ == "__main__":
    n_channels = 15
    input_grid_size = 256
    latent_dim = 512  # 将特征压缩到 4 维
    layer_channels = [64, 128, 256]
    n_residual_blocks = 2

    model = ConvAutoEncoder(n_channels=n_channels,
                            input_grid_size=input_grid_size,
                            latent_dim=latent_dim,
                            layer_channels=layer_channels,
                            n_residual_blocks=n_residual_blocks)

    dummy_input = torch.randn(1, n_channels, input_grid_size, input_grid_size)
    encoded, decoded = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")