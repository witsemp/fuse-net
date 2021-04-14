from sklearn.model_selection import train_test_split
from natsort import natsorted
from shutil import copy
import numpy as np
import cv2 as cv
import os
import argparse
from Utils import manual_resize, make_dataset, move_inputs_targets
import random

# Add arguments to parser
parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='A path to folder containing classes')
parser.add_argument('save_path', type=str, help='A path to folder containing .npy files')
parser.add_argument('dataset_path', type=str, help='A path to folder containgin dataset')
parser.add_argument('--write_files', help='If true, writes .npy files to save_path directory', action='store_true')
parser.add_argument('--split_files', help='If true, splits files into dataset', action='store_true')
parser.add_argument('--visualise_random', help='If true, shows random image from created dataset', action='store_true')
# Parse arguments
args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path
dataset_path = args.dataset_path
# Create lists of classes and models in data_path folder
classes_list = [os.path.join(data_path, class_name) for class_name in os.listdir(data_path) if
                os.path.isdir(os.path.join(data_path, class_name))]
models_list = [os.path.join(class_path, model_name) for class_path in classes_list for model_name in
               os.listdir(class_path)]
inputs_rbg_list = []
inputs_depth_list = []
targets_depth_list = []
# For each model add paths to depth and rgb images
for model_path in models_list:
    depth_path = os.path.join(model_path, 'depth')
    rgb_path = os.path.join(model_path, 'rgb')
    for i, img_path in enumerate(natsorted(os.listdir(rgb_path))):
        if i % 2 == 0:
            inputs_rbg_list.append(os.path.join(rgb_path, img_path))
    for i, img_path in enumerate(natsorted(os.listdir(depth_path))):
        if i % 2 == 0:
            inputs_depth_list.append(os.path.join(depth_path, img_path))
        else:
            targets_depth_list.append(os.path.join(depth_path, img_path))
# Create a list of tuples containing input RGB path, input depth path and target depth path
paths = list(zip(inputs_rbg_list, inputs_depth_list, targets_depth_list))
if not os.path.exists(save_path):
    os.mkdir(save_path)
dst_shape = (120, 160)
if args.write_files:
    for i, (input_rgb_path, input_depth_path, target_depth_path) in enumerate(paths):
        print(f'Saved {i} out of {len(targets_depth_list) - 1} images')
        rgb_base = os.path.basename(input_rgb_path[:input_rgb_path.find('.')])
        depth_base = os.path.basename(input_depth_path[:input_depth_path.find('.')])
        target_depth_base = os.path.basename(target_depth_path[:target_depth_path.find('.')])
        print(rgb_base)
        print(depth_base)
        print(target_depth_base)
        if rgb_base != depth_base:
            print("Error - file name mismatch!")
            print("RGB path: ", input_rgb_path)
            print("Depth path: ", input_depth_path)
            break
        input_rgb_image = cv.imread(input_rgb_path, cv.IMREAD_COLOR)
        input_rgb_image = cv.cvtColor(input_rgb_image, cv.COLOR_BGR2RGB)
        input_rgb_image = cv.resize(input_rgb_image, (160, 120))
        input_depth_image = cv.imread(input_depth_path, cv.IMREAD_ANYDEPTH)
        input_depth_image = manual_resize(input_depth_image, dst_shape)
        input_depth_image = np.expand_dims(input_depth_image, axis=2)
        input_rgbd_image = np.concatenate((input_rgb_image, input_depth_image), axis=2)
        target_depth_image = cv.imread(target_depth_path, cv.IMREAD_ANYDEPTH)
        target_depth_image = manual_resize(target_depth_image, dst_shape)
        np.save(os.path.join(save_path, f'{depth_base}.npy'), input_rgbd_image)
        np.save(os.path.join(save_path, f'{target_depth_base}.npy'), target_depth_image)
if args.split_files:
    inputs = [os.path.join(save_path, filename) for i, filename in enumerate(natsorted(os.listdir(save_path))) if
              i % 2 == 0]
    targets = [os.path.join(save_path, filename) for i, filename in enumerate(natsorted(os.listdir(save_path))) if
               i % 2 != 0]

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.15,
                                                                              random_state=42,
                                                                              shuffle=True)
    inputs_valid, inputs_test, targets_valid, targets_test = train_test_split(inputs_test,
                                                                              targets_test,
                                                                              test_size=0.5,
                                                                              random_state=42,
                                                                              shuffle=True)
    make_dataset(dataset_path)
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')
    move_inputs_targets(inputs_train, targets_train, train_path)
    move_inputs_targets(inputs_valid, targets_valid, valid_path)
    move_inputs_targets(inputs_test, targets_test, test_path)

if args.visualise_random:
    f = random.choice(os.listdir(save_path))
    while int(f[f.find("_") + 1:f.find(".")])%2 != 0:
        f = random.choice(os.listdir(save_path))
        print(f[f.find("_") + 1:f.find(".")])
    i = np.load(os.path.join(save_path, f))
    cv.imshow('RGB', i[:, :, 0:3].astype(np.uint8))
    cv.imshow('depth', i[:, :, 3])
    cv.waitKey(0)

