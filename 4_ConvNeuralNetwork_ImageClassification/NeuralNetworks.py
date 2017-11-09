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

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import ReadData
import collections
import numpy as np
import tensorflow as tf


class DataSet(object):
    def __init__(self, images, labels):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._height = images.shape[1]
        self._width = images.shape[2]
        self._channel = images.shape[3]
        self._total_pixels = self._height * self._width * self._channel
        self._num_classes = labels.shape[1]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 3)
        assert self._channel == 3
        images = images.reshape(self._num_examples, self._total_pixels)

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

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
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def channel(self):
        return self._channel

    @property
    def total_pixels(self):
        return self._total_pixels

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def deepnn(x, info):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    layer_channel = [16, 32, 256]

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, info['height'], info['width'], info['channel']])

    # First convolutional layer - maps one grayscale image to layer_channel[0] feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, info['channel'], layer_channel[0]])
        b_conv1 = bias_variable([layer_channel[0]])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps layer_channel[0] feature maps to layer_channel[1].
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, layer_channel[0], layer_channel[1]])
        b_conv2 = bias_variable([layer_channel[1]])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 40x60 image
    # is down to 10x15xlayer_channel[1] feature maps -- maps this to layer_channel[2] features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([info['height'] * info['width'] // 16 * layer_channel[1], layer_channel[2]])
        b_fc1 = bias_variable([layer_channel[2]])

        h_pool2_flat = tf.reshape(h_pool2, [-1, info['height'] * info['width'] // 16 * layer_channel[1]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the layer_channel[2] features to 3 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([layer_channel[2], info['num_classes']])
        b_fc2 = bias_variable([info['num_classes']])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main():
    # Import data
    train_images, train_labels, validation_images, validation_labels, \
    test_images, test_labels = ReadData.read_data()
    # print(train_images.shape)  # <class 'numpy.ndarray'>, (1150, 40, 60, 3)
    # print(train_labels.shape)  # <class 'numpy.ndarray'>, (1150, 3)
    # print(validation_images.shape)  # <class 'numpy.ndarray'>, (300, 40, 60, 3)
    # print(validation_labels.shape)  # <class 'numpy.ndarray'>, (300, 3)
    # print(test_images.shape)  # <class 'numpy.ndarray'>, (380, 40, 60, 3)
    # print(test_labels.shape)  # <class 'numpy.ndarray'>, (380, 3)

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)
    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    datasets = Datasets(train=train, validation=validation, test=test)
    # print(datasets.train.images.shape)  # <class 'numpy.ndarray'>, (1150, 7200)
    # print(datasets.train.labels.shape)  # <class 'numpy.ndarray'>, (1150, 3)
    # print(datasets.validation.images.shape)  # <class 'numpy.ndarray'>, (300, 7200)
    # print(datasets.validation.labels.shape)  # <class 'numpy.ndarray'>, (300, 3)
    # print(datasets.test.images.shape)  # <class 'numpy.ndarray'>, (380, 7200)
    # print(datasets.test.labels.shape)  # <class 'numpy.ndarray'>, (380, 3)

    # Create the model
    info = {'num_examples': train.num_examples, 'height': train.height,
            'width': train.width, 'channel': train.channel,
            'total_pixels': train.total_pixels,
            'num_classes': train.num_classes}
    x = tf.placeholder(tf.float32, [None, info['total_pixels']])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, info['num_classes']])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x, info)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch = datasets.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: datasets.test.images, y_: datasets.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
