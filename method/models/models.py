"""

Implementation of the Auto-encoder model architecture described in the paper:
Marine cloud base height retrieval from MODIS cloud properties using supervised machine learning.

Edited by Julien LENHARDT

Classes and functions are either taken or adapted from the torchvision implementation.

References:
https://github.com/pytorch/vision
https://github.com/pytorch/vision/tree/main/torchvision/models
TorchVision maintainers and contributors:
    TorchVision: PyTorch's Computer Vision library, GitHub repository, https://github.com/pytorch/vision, 2016.

"""

import numpy as np
import torch
import torch.nn as nn
import io
from contextlib import redirect_stdout
# from torchinfo import summary

# region Convolution layers


def conv3_3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """
    3x3 convolution

    :param in_channels:
    :param out_channels:
    :param stride:
    :param groups:
    :param dilation:
    :return:
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1_1(in_channels, out_channels, stride=1):
    """
    1x1 convolution

    :param in_channels:
    :param out_channels:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3_3_transpose(in_channels, out_channels, stride=2, groups=1):
    """
    3x3 transposed convolution

    :param in_channels:
    :param out_channels:
    :param stride:
    :param groups:
    :return:
    """
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, groups=groups)


def conv1_1_transpose(in_channels, out_channels, stride=1):
    """
    1x1 transposed convolution

    :param in_channels:
    :param out_channels:
    :param stride:
    :return:
    """
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


# endregion

# region Convolution Blocks

# Convolution block

class ConvBlock(nn.Module):
    """
    Convolutional Block => Downsampling

    Structure:
        3 * Conv2d followed by LeakyReLu
        Maximum Pooling (kernel_size = 2 stride = 2)
        Batch Normalization
    """

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('ConvBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ConvBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3_3(inplanes, planes, stride)
        self.conv2 = conv3_3(planes, planes, stride)
        self.conv3 = conv3_3(planes, planes, stride)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = norm_layer(planes)

    def forward(self, x):

        # Convolution 1
        out = self.conv1(x)
        out = self.leakyrelu(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.leakyrelu(out)

        # Convolution 3 + Batch Normalization
        out = self.conv3(out)
        out = self.bn(out)
        out = self.leakyrelu(out)

        # Maximum pooling (downsampling)
        out = self.maxpool(out)

        return out


# Convolution Transposed block

class ConvTransposeBlock(nn.Module):
    """
    Convolutional Transposed Block => Upsampling

    Structure:
        ConvTransposed2d (out_channels = in_channels // 2)
        3 * Conv2d followed by LeakyReLu
        Batch Normalization
    """

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 dilation=1, norm_layer=None):
        super(ConvTransposeBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('ConvBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ConvBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3_3(inplanes, inplanes, stride)
        self.conv2 = conv3_3(inplanes, inplanes, stride)
        self.conv3 = conv3_3(inplanes, inplanes, stride)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.upsample = conv3_3_transpose(inplanes, planes, stride=2)
        self.bn = norm_layer(inplanes)

    def forward(self, x):

        # Convolution 1
        out = self.conv1(x)
        out = self.leakyrelu(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.leakyrelu(out)

        # Convolution 3 + Batch Normalization
        out = self.conv3(out)
        out = self.bn(out)
        out = self.leakyrelu(out)

        # Upsampling
        out = self.upsample(out)

        return out


# endregion

# region Autoencoder network

# Encoder network

class Encoder(nn.Module):
    """

    """
    def get_last_dim(self, input_dim):
        """

        :param input_dim:
        :return:
        """
        f = io.StringIO()
        with redirect_stdout(f):
            summary(self, input_dim)
        out = f.getvalue()
        # out = out.split('ConvBlock')[-1].split()[2:5]
        out = out.split('0\n')[-4].split()[2:]
        dims = tuple([int(''.join(e for e in s if e.isalnum())) for s in out])
        return dims

    def __init__(self, n_channels=3, encoding_space=1024):
        """

        :param n_channels:
        """
        super(Encoder, self).__init__()

        self.n_channels = n_channels
        self.encoding_space = encoding_space

        # Encoder layers
        self.conv1 = conv3_3(self.n_channels, self.n_channels, stride=2)
        self.down1 = ConvBlock(n_channels, 6)
        self.down2 = ConvBlock(6, 32)
        self.down3 = ConvBlock(32, 64)
        self.down4 = ConvBlock(64, 128)
        self.down5 = ConvBlock(128, 256)
        if self.encoding_space == 256:
            self.down_max = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten(start_dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        if self.encoding_space == 256:
            x = self.down_max(x)
        x = self.flat(x)

        return x


# Decoder network

class Decoder(nn.Module):
    """

    """
    def __init__(self, n_channels=3, unflat_size=None, encoding_space=1024):
        """

        :param n_channels:
        """
        super(Decoder, self).__init__()

        self.n_channels = n_channels
        if unflat_size is None:
            self.unflat_size = (256, 1, 1)
        else:
            self.unflat_size = unflat_size
        self.encoding_space = encoding_space

        # Decoder layers
        self.unflat = nn.Unflatten(dim=1,
                                   unflattened_size=self.unflat_size)
        if self.encoding_space == 256:
            self.up_sample = conv3_3_transpose(256, 256, stride=2)
        self.conv1 = conv3_3_transpose(256, 256, stride=2)
        self.up1 = ConvTransposeBlock(256, 128)
        self.up2 = ConvTransposeBlock(128, 64)
        self.up3 = ConvTransposeBlock(64, 32)
        self.up4 = ConvTransposeBlock(32, 6)
        self.up5 = ConvTransposeBlock(6, self.n_channels)
        # self.conv2 = conv3_3_transpose(6, self.n_channels, stride=1)

    def forward(self, x):

        x = self.unflat(x)
        if self.encoding_space == 256:
            x = self.up_sample(x)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        # x = self.conv2(x)

        return x


# Convolutional Auto-Encoder class definition

class ConvAutoEncoder_old(nn.Module):
    """
    Convolutional Auto-Encoder pytorch model class.
    """
    def __init__(self,
                 n_channels=3,
                 input_grid_size=128,
                 only_eval=False,
                 latent_dim=256):
        super(ConvAutoEncoder_old, self).__init__()

        self.n_channels = n_channels
        self.input_grid_size = input_grid_size
        # self.only_eval = only_eval
        self.latent_dim = latent_dim

        self.encoder = Encoder(n_channels=self.n_channels)

        # self.unflat_size = self.encoder.get_last_dim((self.n_channels,
        #                                               self.input_grid_size,
        #                                               self.input_grid_size))
        self.unflat_size = (256, 2, 2)
        self.model_dim = np.prod(self.unflat_size)

        # latent space distributions
        self.latent_space = nn.Linear(self.model_dim, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, self.model_dim)
        self.decoder = Decoder(n_channels=self.n_channels,
                               unflat_size=self.unflat_size)

        self.final_actfunc = None

        self.__name__ = 'autoencoder_' + '_'.join([str(dim) for dim in self.unflat_size])

    def forward(self, x):

        encoded = self.encode(x)

        decoded = self.decode(encoded)

        return encoded, decoded

    def encode(self, input):
        """
        Encodes the input through encoder network.
        :param input:
        :return:
        """
        # Feed input through encoder network
        output = self.encoder(input) #output shape (N,256,16,16)
        output = torch.flatten(output, start_dim=1)
        output = self.latent_space(output)
        return output

    def decode(self, z):
        """
        Decodes the given latent encoding onto input space.
        :param z:
        :return:
        """
        # Feed latent encoding through decoder network
        output = self.decoder_input(z)
        output = self.decoder(output)
        if self.final_actfunc is not None:
            output = self.final_actfunc(output)
        return output

# endregion



class ConvAutoEncoder(nn.Module):
    """
    Convolutional Auto-Encoder pytorch model class.
    """
    def __init__(self,
                 n_channels=3,
                 input_grid_size=128,
                 only_eval=False,
                 latent_dim=256):
        super(ConvAutoEncoder, self).__init__()

        self.n_channels = n_channels
        self.input_grid_size = input_grid_size
        # self.only_eval = only_eval
        self.latent_dim = latent_dim

        self.encoder = Encoder(n_channels=self.n_channels)

        # self.unflat_size = self.encoder.get_last_dim((self.n_channels,
        #                                               self.input_grid_size,
        #                                               self.input_grid_size))
        self.unflat_size = (256, 2, 2)
        self.model_dim = np.prod(self.unflat_size)

        # latent space distributions
        self.latent_space = nn.Linear(self.model_dim, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, self.model_dim)
        self.decoder = Decoder(n_channels=self.n_channels,
                               unflat_size=self.unflat_size)

        self.final_actfunc = None

        self.__name__ = 'autoencoder_' + '_'.join([str(dim) for dim in self.unflat_size])

    def forward(self, x):

        encoded = self.encode(x)

        decoded = self.decode(encoded)

        return encoded, decoded

    def encode(self, input):
        """
        Encodes the input through encoder network.
        :param input:
        :return:
        """
        # Feed input through encoder network
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)
        output = self.latent_space(output)
        return output

    def decode(self, z):
        """
        Decodes the given latent encoding onto input space.
        :param z:
        :return:
        """
        # Feed latent encoding through decoder network
        output = self.decoder_input(z)
        output = self.decoder(output)
        if self.final_actfunc is not None:
            output = self.final_actfunc(output)
        return output

# endregion


# # 新的 ConvAutoEncoder 类，包含跳跃连接
# class ConvAutoEncoder(nn.Module):
#     def __init__(self, n_channels=3, input_grid_size=128, latent_dim=256):
#         super(ConvAutoEncoder, self).__init__()

#         self.n_channels = n_channels
#         self.input_grid_size = input_grid_size
#         self.latent_dim = latent_dim

#         # Encoder layers with skip connections
#         self.encoder_conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
#         self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         # Bottleneck
#         self.bottleneck = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)

#         # Decoder layers with skip connections
#         self.upconv4 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=2, stride=2)
#         self.decoder_conv4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.decoder_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.decoder_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.decoder_conv1 = nn.Conv2d(128, n_channels, kernel_size=3, padding=1)

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Encoder with skip connections
#         e1 = self.relu(self.encoder_conv1(x))
#         e2 = self.relu(self.encoder_conv2(self.pool(e1)))
#         e3 = self.relu(self.encoder_conv3(self.pool(e2)))
#         e4 = self.relu(self.encoder_conv4(self.pool(e3)))

#         # Bottleneck
#         b = self.relu(self.bottleneck(self.pool(e4)))

#         # Decoder with skip connections
#         d4 = self.upconv4(b)
#         d4 = torch.cat((d4, e4), dim=1)
#         d4 = self.relu(self.decoder_conv4(d4))

#         d3 = self.upconv3(d4)
#         d3 = torch.cat((d3, e3), dim=1)
#         d3 = self.relu(self.decoder_conv3(d3))

#         d2 = self.upconv2(d3)
#         d2 = torch.cat((d2, e2), dim=1)
#         d2 = self.relu(self.decoder_conv2(d2))

#         d1 = self.upconv1(d2)
#         d1 = torch.cat((d1, e1), dim=1)
#         d1 = self.decoder_conv1(d1)  # 最后一层不使用激活函数

#         return d1