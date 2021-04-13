import numpy as np
import cv2 as cv
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()

        self.rgb_layers = self.get_layers_from_vgg16()
        self.depth_layers = self.get_layers_from_vgg16()

        # Create layers for RGB encoder
        self.CBR1_RGB_ENCODER = self.create_encoder_block(64, [0, 1], 'rgb')
        self.pool1_rgb = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.CBR2_RGB_ENCODER = self.create_encoder_block(128, [2, 3], 'rgb')
        self.pool2_rgb = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.CBR3_RGB_ENCODER = self.create_encoder_block(256, [4, 5, 6], 'rgb')
        self.pool3_rgb = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3_rgb = nn.Dropout(p=0.4)
        self.CBR4_RGB_ENCODER = self.create_encoder_block(512, [7, 8, 9], 'rgb')
        self.pool4_rgb = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4_rgb = nn.Dropout(p=0.4)
        self.CBR5_RGB_ENCODER = self.create_encoder_block(512, [10, 11, 12], 'rgb')
        self.pool5_rgb = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4_rgb = nn.Dropout(p=0.4)

        # Create layers for depth encoder
        # Average the weights of first layer in depth encoder,
        # as it will be accepting one-dimensional
        # inputs instead of three.
        avg = torch.mean(self.depth_layers[0].weight.data, dim=1)
        avg = avg.unsqueeze(1)
        self.conv1d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1d.weight.data = avg
        self.CBR1_DEPTH_ENCODER = nn.Sequential(
            self.conv1d,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.depth_layers[1],
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1_depth = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.CBR2_DEPTH_ENCODER = self.create_encoder_block(128, [2, 3], 'depth')
        self.pool2_depth = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.CBR3_DEPTH_ENCODER = self.create_encoder_block(256, [4, 5, 6], 'depth')
        self.pool3_depth = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3_depth = nn.Dropout(p=0.4)
        self.CBR4_DEPTH_ENCODER = self.create_encoder_block(512, [7, 8, 9], 'depth')
        self.pool4_depth = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4_depth = nn.Dropout(p=0.4)
        self.CBR5_DEPTH_ENCODER = self.create_encoder_block(512, [10, 11, 12], 'depth')

        # Create layers for decoder
        self.CBR5_DECODER = self.create_decoder_block(512, 512)
        self.CBR4_DECODER = self.create_decoder_block(512, 256)
        self.CBR3_DECODER = self.create_decoder_block(256, 128)
        self.CBR2_DECODER = self.create_decoder_block(128, 64, use_dropout=False)
        self.CBR1_DECODER = self.create_decoder_block(64, 1, use_dropout=False)

        print('[INFO] FuseNet model has been created')

    def get_layers_from_vgg16(self) -> List[nn.Conv2d]:
        # Create a list of all VGG16 layers
        layers = list(models.vgg16(pretrained=True).features.children())
        # Select convolutional layers of VGG16
        return [layer for layer in layers if isinstance(layer, nn.Conv2d)]

    def create_encoder_block(self, batch_norm_features: int, conv_layers_indices: List[int],
                             encoder_type) -> nn.Sequential:
        """


        :param encoder_type: "depth" to use self.depth_layers, "rgb" to use self.rgb_layers
        :param batch_norm_features: number of features for batch normalization layers
        :param conv_layers_list: indices of conv layers from self.layers
        :return nn.Sequential: CBR RGB encoder block
        """
        block_layers = []
        for index in conv_layers_indices:
            if encoder_type == 'rgb':
                block_layers += [self.rgb_layers[index],
                                 nn.BatchNorm2d(batch_norm_features),
                                 nn.ReLU(inplace=True)]
            elif encoder_type == 'depth':
                block_layers += [self.depth_layers[index],
                                 nn.BatchNorm2d(batch_norm_features),
                                 nn.ReLU(inplace=True)]
        return nn.Sequential(*block_layers)

    def create_decoder_block(self, in_features: int, out_features: int, use_dropout: bool = True) -> nn.Sequential:
        """
        :param use_dropout: indicates if last dropout layer of a block is used
        :param in_features: number of input features
        :param out_features: number of output features
        :return nn.Sequential: decoder block
        """
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Dropout(p=0.4)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_features),
                nn.ReLU(),
            )

    def forward(self, rgb_inputs: torch.Tensor, depth_inputs: torch.Tensor) -> torch.Tensor:
        ### Depth encoder forward pass ###
        # Stage 1
        x_1 = self.CBR1_DEPTH_ENCODER(depth_inputs)
        x, id1_d = self.pool1_depth(x_1)

        # Stage 2
        x_2 = self.CBR2_DEPTH_ENCODER(x)
        x, id2_d = self.pool2_depth(x_2)

        # Stage 3
        x_3 = self.CBR3_DEPTH_ENCODER(x)
        x, id3_d = self.pool3_depth(x_3)
        x = self.dropout3_depth(x)

        # Stage 4
        x_4 = self.CBR4_DEPTH_ENCODER(x)
        x, id4_d = self.pool4_depth(x_4)
        x = self.dropout4_depth(x)

        # Stage 5
        x_5 = self.CBR5_DEPTH_ENCODER(x)

        ### RGB encoder forward pass ###
        y = self.CBR1_RGB_ENCODER(rgb_inputs)
        y = torch.add(y, x_1)
        y, id1 = self.pool1_rgb(y)

        # Stage 2
        y = self.CBR2_RGB_ENCODER(y)
        y = torch.add(y, x_2)
        y, id2 = self.pool2(y)

        # Stage 3
        y = self.CBR3_RGB_ENCODER(y)
        y = torch.add(y, x_3)
        y, id3 = self.pool3(y)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB_ENCODER(y)
        y = torch.add(y, x_4)
        y, id4 = self.pool4(y)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB_ENCODER(y)
        y = torch.add(y, x_5)
        y, id5 = self.pool5(y)
        y = self.dropout5(y)

        ### Decoder forward pass ###
        # Stage 5
        y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y.size())
        y = self.CBR5_DECODER(y)

        # Stage 4
        y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
        y = self.CBR4_DECODER(y)

        # Stage 3
        y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
        y = self.CBR3_DECODER(y)

        # Stage 2
        y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
        y = self.CBR2_DECODER(y)

        # Stage 1
        y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
        y = self.CBR1_DECODER(y)

        return y


fn = FuseNet()
print(fn)
