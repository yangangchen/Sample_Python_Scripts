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

"""An image classifier using deep convolutional neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ImportDatasets
import numpy as np
import tensorflow as tf
import _pickle

np.random.seed(1)
tf.set_random_seed(1)


####################################################

class NeuralNetworkDatasets(object):
    def __init__(self, images, labels):
        """Construct a DataSet."""
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        assert images.shape[3] == 3
        self._num_examples = labels.shape[0]
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


class ConvolutionalNeuralNetwork:
    def __init__(self, info, filename=None):
        if filename is None:
            self.shapes = {}
            self.parameters = {}

            self.info = info
            self.conv_num_channels = [3, 16, 32]
            assert self.info['channel'] == self.conv_num_channels[0]
            self.fc_num_channels = 256

            ## Convolutional layers:
            for layer in range(1, len(self.conv_num_channels)):
                name = 'conv' + str(layer)
                self.shapes[name] = (5, 5, self.conv_num_channels[layer - 1], self.conv_num_channels[layer])
                self.parameters[name + '_W'] = 0.1 * np.random.randn(*self.shapes[name])
                self.parameters[name + '_b'] = 0.1 * np.ones(self.shapes[name][-1])

            ## Fully connected layer (hidden):
            name = 'fc_hidden'
            self.shapes[name] = (self.info['height'] * self.info['width']
                                 // (4 ** (len(self.conv_num_channels) - 1)) * self.conv_num_channels[-1],
                                 self.fc_num_channels)
            self.parameters[name + '_W'] = 0.1 * np.random.randn(*self.shapes[name])
            self.parameters[name + '_b'] = 0.1 * np.ones(self.shapes[name][-1])

            ## Fully connected layer (output):
            name = 'fc_output'
            self.shapes[name] = (self.fc_num_channels, self.info['num_classes'])
            self.parameters[name + '_W'] = 0.1 * np.random.randn(*self.shapes[name])
            self.parameters[name + '_b'] = 0.1 * np.ones(self.shapes[name][-1])

        else:
            tmp_dict = _pickle.load(open(filename, 'rb'))
            self.__dict__.update(tmp_dict)

        self.initialize()

    def initialize_onelayer(self, name):
        with tf.variable_scope(name, reuse=None):
            tf.get_variable(name + '_W', shape=self.shapes[name], dtype=tf.float32,
                            initializer=tf.constant_initializer(self.parameters[name + '_W']))
            tf.get_variable(name + '_b', shape=[self.shapes[name][-1]], dtype=tf.float32,
                            initializer=tf.constant_initializer(self.parameters[name + '_b']))

    def initialize(self):
        with tf.variable_scope('conv_neural_net', reuse=None):
            ## Convolutional layers:
            for layer in range(1, len(self.conv_num_channels)):
                self.initialize_onelayer('conv' + str(layer))

            ## Fully connected layer (hidden):
            self.initialize_onelayer('fc_hidden')

            ## Fully connected layer (output):
            self.initialize_onelayer('fc_output')

    def evaluate(self, x, keep_prob):
        """ builds the graph for a convolutional neural network for classifying images.
        Args:
          x: an input tensor with the dimensions (N_examples, info['total_pixels']),
          where info['total_pixels'] = info['height'] * info['width'] * info['channel']
          is the total number of pixels in each image.
        Returns:
          A tuple (y, keep_prob). y is a tensor of shape (N_examples, info['num_classes']),
          with values equal to the logits of classifying the images into one of 3 classes
          (airplanes, motorbikes, watch). keep_prob is a scalar placeholder for the probability
          of dropout.

        Convolutional layers:
          (?, 40, 60, 3)
          (?, 20, 30, 16)
          (?, 10, 15, 32)
        Fully connected layer (hidden):
          (?, 256)
        Fully connected layer (output):
          (?, 3)
        """

        assert x.get_shape()[1] == self.info['height']
        assert x.get_shape()[2] == self.info['width']
        assert x.get_shape()[3] == self.info['channel']

        with tf.variable_scope('conv_neural_net', reuse=True):
            for layer in range(1, len(self.conv_num_channels)):
                # Convolutional layer: Maps self.layer_num_channels[layer - 1] feature maps to
                # self.layer_num_channels[layer] feature maps.
                name = 'conv' + str(layer)
                with tf.variable_scope(name, reuse=True):
                    W = tf.get_variable(name + '_W', dtype=tf.float32)
                    b = tf.get_variable(name + '_b', dtype=tf.float32)
                    z = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
                    h = tf.nn.relu(z)

                # Pooling layer: Downsamples by 2X.
                name = 'pool' + str(layer)
                with tf.name_scope(name):
                    x = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Fully connected layer (hidden): After 2 round of downsampling, our 40x60x3 image is down
            # to 10x15x32 feature maps. Maps this to 256 features.
            name = 'fc_hidden'
            with tf.variable_scope(name, reuse=True):
                W = tf.get_variable(name + '_W', dtype=tf.float32)
                b = tf.get_variable(name + '_b', dtype=tf.float32)
                x = tf.reshape(x, [-1, self.shapes[name][0]])
                h = tf.nn.relu(tf.matmul(x, W) + b)

            # Dropout: Controls the complexity of the model, prevents co-adaptation of features.
            name = 'dropout'
            with tf.name_scope(name):
                h = tf.nn.dropout(h, keep_prob)

            # Fully connected layer (output): Map the 256 features to 3 classes
            name = 'fc_output'
            with tf.variable_scope(name, reuse=True):
                W = tf.get_variable(name + '_W', dtype=tf.float32)
                b = tf.get_variable(name + '_b', dtype=tf.float32)
                # y_conv = tf.nn.softmax(tf.matmul(h, W) + b)
                # Softmax is absorted into the loss function "tf.nn.softmax_cross_entropy_with_logits"
                y_conv = tf.matmul(h, W) + b

            return y_conv

    def sync_onelayer(self, sess, name):
        with tf.variable_scope(name, reuse=True):
            W = tf.get_variable(name + '_W', dtype=tf.float32)
            b = tf.get_variable(name + '_b', dtype=tf.float32)
            self.parameters[name + '_W'] = sess.run(W)
            self.parameters[name + '_b'] = sess.run(b)

    def sync(self, sess):
        with tf.variable_scope('conv_neural_net', reuse=True):
            ## Convolutional layers:
            for layer in range(1, len(self.conv_num_channels)):
                self.sync_onelayer(sess, 'conv' + str(layer))
            ## Fully connected layer (hidden):
            self.sync_onelayer(sess, 'fc_hidden')
            ## Fully connected layer (output):
            self.sync_onelayer(sess, 'fc_output')

    def save(self, sess, filename):
        self.sync(sess)
        _pickle.dump(self.__dict__, open(filename, 'wb'), protocol=4)


####################################################

def main(max_train_steps=1000, batch_size=50):
    ## Import the datasets.
    train_images, train_labels, validation_images, validation_labels, \
    test_images, test_labels, info = ImportDatasets.import_datasets()

    ## Construct the datasets.
    train = NeuralNetworkDatasets(train_images, train_labels)
    validation = NeuralNetworkDatasets(validation_images, validation_labels)
    test = NeuralNetworkDatasets(test_images, test_labels)

    # print(train.images.shape)  # <class 'numpy.ndarray'>, (1189, 40, 60, 3)
    # print(train.labels.shape)  # <class 'numpy.ndarray'>, (1189, 3)
    # print(validation.images.shape)  # <class 'numpy.ndarray'>, (275, 40, 60, 3)
    # print(validation.labels.shape)  # <class 'numpy.ndarray'>, (275, 3)
    # print(test.images.shape)  # <class 'numpy.ndarray'>, (366, 40, 60, 3)
    # print(test.labels.shape)  # <class 'numpy.ndarray'>, (366, 3)

    ## Create the model
    x = tf.placeholder(tf.float32, [None, info['height'], info['width'], info['channel']])
    y = tf.placeholder(tf.float32, [None, info['num_classes']])
    keep_prob = tf.placeholder(tf.float32)

    ## Define the deep convolutional neural network
    conv_net = ConvolutionalNeuralNetwork(info=info)
    # conv_net = ConvolutionalNeuralNetwork(info=info, filename='conv_neural_net.pkl')
    y_conv = conv_net.evaluate(x, keep_prob)

    ## Define the loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    ## Define the optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    ## Define the evaluation metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y, axis=1)), tf.float32))

    ## Run the tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(max_train_steps + 1):
            batch_images, batch_labels = train.next_batch(batch_size)

            if step % 100 == 0:
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: batch_images, y: batch_labels, keep_prob: 1.0})
                print('Step %d, training accuracy %g' % (step, train_accuracy))

            sess.run(optimizer,
                     feed_dict={x: batch_images, y: batch_labels, keep_prob: 0.5})

        test_accuracy = sess.run(accuracy,
                                 feed_dict={x: test.images, y: test.labels, keep_prob: 1.0})
        print('Test accuracy %g' % test_accuracy)

        conv_net.save(sess, 'conv_neural_net.pkl')


####################################################

if __name__ == '__main__':
    main(max_train_steps=1000, batch_size=50)
