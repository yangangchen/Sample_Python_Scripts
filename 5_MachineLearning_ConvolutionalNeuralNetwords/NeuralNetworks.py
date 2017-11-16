# NeuralNetworks.py
#
# Author: Yangang Chen, based on the TensorFlow library
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A deep image classifier using convolutional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ImportDatasets
import collections
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


####################################################

class ConstructDataSets(object):
    def __init__(self, images, labels):
        """Construct a DataSet."""
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        assert images.shape[3] == 3
        self._num_examples = labels.shape[0]

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._batch_index = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle_dataset(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        # Shuffle at the beginning of each epoch
        if self._batch_index == 0 and shuffle:
            self.shuffle_dataset()
        # Return the next batch
        if self._batch_index + batch_size >= self._num_examples:
            batch_images = self._images[self._batch_index:]
            batch_labels = self._labels[self._batch_index:]
            # Finished epoch
            self._epochs_completed += 1
            self._batch_index = 0
        else:
            batch_images = self._images[self._batch_index:self._batch_index + batch_size]
            batch_labels = self._labels[self._batch_index:self._batch_index + batch_size]
            self._batch_index += batch_size
        return batch_images, batch_labels


####################################################

def weight_bias_variables(weight_shape):
    """Generate a weight variable and a bias variable of a given shape."""
    initial_weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1))
    initial_bias = tf.Variable(tf.constant(0.1, shape=[weight_shape[-1]]))
    return initial_weight, initial_bias


def deepnn(x, info, layer_num_channels=(16, 32, 256)):
    """deepnn builds the graph for a deep net for classifying images.
    Args:
      x: an input tensor with the dimensions (N_examples, info['total_pixels']),
      where info['total_pixels'] = info['height'] * info['width'] * info['channel']
      is the total number of pixels in each image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, info['num_classes']),
      with values equal to the logits of classifying the images into one of 3 classes
      (airplanes, motorbikes, watch). keep_prob is a scalar placeholder for the probability
      of dropout.
    """

    # Reshape to use within a convolutional neural net.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, info['height'], info['width'], info['channel']])

    # First convolutional layer - maps one grayscale image to layer_channel[0] feature maps.
    with tf.name_scope('conv1'):
        W_conv1, b_conv1 = weight_bias_variables([5, 5, info['channel'], layer_num_channels[0]])
        z_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_conv1 = tf.nn.relu(z_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1,
                                 ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional layer -- maps layer_channel[0] feature maps to layer_channel[1].
    with tf.name_scope('conv2'):
        W_conv2, b_conv2 = weight_bias_variables([5, 5, layer_num_channels[0], layer_num_channels[1]])
        z_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        h_conv2 = tf.nn.relu(z_conv2)

    # Second pooling layer - downsamples by 2X.
    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2,
                                 ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer 1 -- after 2 round of downsampling, our 40x60 image
    # is down to 10x15xlayer_channel[1] feature maps -- maps this to layer_channel[2] features.
    with tf.name_scope('fc1'):
        W_fc1, b_fc1 = weight_bias_variables([info['height'] * info['width'] // 16 * layer_num_channels[1],
                                              layer_num_channels[2]])
        h_pool2_flat = tf.reshape(h_pool2, [-1, info['height'] * info['width'] // 16 * layer_num_channels[1]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the layer_channel[2] features to 3 classes
    with tf.name_scope('fc2'):
        W_fc2, b_fc2 = weight_bias_variables([layer_num_channels[2], info['num_classes']])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


####################################################

def main():
    # Import the datasets.
    train_images, train_labels, validation_images, validation_labels, \
    test_images, test_labels, info = ImportDatasets.import_datasets()

    # Construct the datasets.
    train = ConstructDataSets(train_images, train_labels)
    validation = ConstructDataSets(validation_images, validation_labels)
    test = ConstructDataSets(test_images, test_labels)

    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    datasets = Datasets(train=train, validation=validation, test=test)

    # print(datasets.train.images.shape)  # <class 'numpy.ndarray'>, (1189, 40, 60, 3)
    # print(datasets.train.labels.shape)  # <class 'numpy.ndarray'>, (1189, 3)
    # print(datasets.validation.images.shape)  # <class 'numpy.ndarray'>, (275, 40, 60, 3)
    # print(datasets.validation.labels.shape)  # <class 'numpy.ndarray'>, (275, 3)
    # print(datasets.test.images.shape)  # <class 'numpy.ndarray'>, (366, 40, 60, 3)
    # print(datasets.test.labels.shape)  # <class 'numpy.ndarray'>, (366, 3)

    # Create the model
    x = tf.placeholder(tf.float32, [None, info['height'], info['width'], info['channel']])
    y_ = tf.placeholder(tf.float32, [None, info['num_classes']])

    # Build the graph for the deep neural network
    y_conv, keep_prob = deepnn(x, info)

    # Define loss and optimizer
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Define the evaluation metric
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    # Run the tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000 + 1):
            batch_images, batch_labels = datasets.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})
                print('Step %d, training accuracy %g' % (i, train_accuracy))
            sess.run(optimizer,
                     feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})

        test_accuracy = sess.run(accuracy,
                                 feed_dict={x: datasets.test.images, y_: datasets.test.labels, keep_prob: 1.0})
        print('Test accuracy %g' % test_accuracy)


####################################################

if __name__ == '__main__':
    main()
