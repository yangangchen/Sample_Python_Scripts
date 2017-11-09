# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image, ImageOps
from scipy.misc import imread, imshow


def read_data(root_dir="./data", image_size=(60, 40), split_ratio=(0.65, 0.15, 0.2)):
    all_images = []
    all_labels = []
    folder_list = os.listdir(root_dir)
    folder_num = -1
    for folder in folder_list:
        folder_num += 1
        print(str(folder_num) + ': ' + folder)
        folder_dir = os.path.join(root_dir, folder)
        file_list = os.listdir(folder_dir)
        for file in file_list:
            file_dir = os.path.join(folder_dir, file)
            # img = imread(file_dir)
            with Image.open(file_dir) as img:
                img = ImageOps.fit(img, image_size, method=0, bleed=0.0, centering=(0.5, 0.5))
                img = np.array(img)
                # imshow(img)
                if len(img.shape) == 3:
                    if img.shape[2] == 3:
                        all_images.append(img)
                        all_labels.append(folder_num)
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # Convert the labels to one-hot representation
    num_examples = len(all_labels)
    num_classes = folder_num + 1
    index_offset = np.arange(num_examples) * num_classes
    all_labels_one_hot = np.zeros((num_examples, num_classes))
    all_labels_one_hot.flat[index_offset + all_labels.ravel()] = 1
    all_labels = all_labels_one_hot

    # Shuffle the dataset
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    all_images = all_images[indices, :, :, :]
    all_labels = all_labels[indices, :]
    # print(all_images.shape)  # (1830, 40, 60, 3)
    # print(all_labels.shape)  # (1830, 3)

    # Train, validation, test
    split0 = int(num_examples * split_ratio[0]) // 50 * 50
    split1 = int(num_examples * (split_ratio[0] + split_ratio[1])) // 50 * 50
    train_images = all_images[:split0, :, :, :]
    train_labels = all_labels[:split0, :]
    validation_images = all_images[split0:split1, :, :, :]
    validation_labels = all_labels[split0:split1, :]
    test_images = all_images[split1:, :, :, :]
    test_labels = all_labels[split1:, :]

    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels
