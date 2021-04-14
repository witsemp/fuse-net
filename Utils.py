import numpy as np
from typing import Tuple, List
from shutil import copy
import os


def manual_resize(image: np.ndarray, dst_shape: Tuple) -> np.ndarray:
    h_in, w_in = image.shape
    h_dst, w_dst = dst_shape
    h_range = list(range(h_in))
    w_range = list(range(w_in))
    h_stride = h_in // h_dst
    w_stride = w_in // w_dst
    dst_image = np.zeros(dst_shape)
    for i, row in enumerate(h_range[0::h_stride]):
        for j, col in enumerate(w_range[0::w_stride]):
            dst_image[i, j] = image[row, col]
    return dst_image.astype(np.float32)


def make_dataset(dataset_path: str):
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    try:
        for path in [train_path, valid_path, test_path]:
            inputs_path = os.path.join(path, 'inputs')
            targets_path = os.path.join(path, 'targets')
            os.mkdir(path)
            os.mkdir(inputs_path)
            os.mkdir(targets_path)
    except OSError:
        print("Train/valid directories creation failed")


def move_inputs_targets(inputs_list: List,
                        targets_list: List,
                        target_dir_path: str):
    for i, (input_name, target_name) in enumerate(zip(inputs_list, targets_list)):
        copy(input_name, os.path.join(target_dir_path, 'inputs'))
        copy(target_name, os.path.join(target_dir_path, 'targets'))
