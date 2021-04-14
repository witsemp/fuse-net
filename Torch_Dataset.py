import torch
from torch.utils.data import Dataset
import cv2 as cv
from natsort import natsorted
import numpy as np
import os


def tensor_from_1ch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).unsqueeze(dim=0)


def tensor_from_nch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)


def image_from_tensor(tensor: torch.Tensor, div: float) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy() * div
    return img.astype(np.float32)


class RotationDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.inputs_path = os.path.join(self.dataset_path, 'inputs')
        self.targets_path = os.path.join(self.dataset_path, 'targets')
        self.inputs = natsorted(os.listdir(self.inputs_path))
        self.targets = natsorted(os.listdir(self.targets_path))
        self.depth_div = 1.09
        self.rgb_div = 255.0

    def __getitem__(self, index):
        input_path, target_path = os.path.join(self.inputs_path, self.inputs[index]), \
                                  os.path.join(self.targets_path, self.targets[index])

        input_img, target_img = np.load(input_path).astype(np.float32), \
                                np.load(target_path).astype(np.float32) / self.depth_div
        input_rgb = input_img[:, :, 0:3] / self.rgb_div
        input_depth = input_img[:, :, 3] / self.depth_div
        input_rgb_tensor = tensor_from_nch_image(input_rgb)
        input_depth_tensor = tensor_from_1ch_image(input_depth)
        target_depth_tensor = tensor_from_1ch_image(target_img)

        return input_rgb_tensor, input_depth_tensor, target_depth_tensor

    def __len__(self):
        return len(self.inputs)
