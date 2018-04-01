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
import keras


####################################################


def ConvolutionalNeuralNetwork(input_shape, dropout_rate):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X = keras.layers.Input(input_shape)

    # Convolutional layer 1: CONV -> RELU -> MAXPOOL
    Y = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv1')(X)
    Y = keras.layers.Activation('relu')(Y)
    Y = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool1')(Y)

    # Convolutional layer 2: CONV -> RELU -> MAXPOOL
    Y = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv2')(Y)
    Y = keras.layers.Activation('relu')(Y)
    Y = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2')(Y)

    # Fully connected layer (hidden): FLATTEN Y (means convert it to a vector) + FULLYCONNECTED
    Y = keras.layers.Flatten()(Y)
    Y = keras.layers.Dense(units=256, activation='relu', name='fc_hidden')(Y)

    # Dropout layer:
    Y = keras.layers.Dropout(rate=dropout_rate)(Y)

    # Fully connected layer (output): FULLYCONNECTED
    Y = keras.layers.Dense(units=3, activation='softmax', name='fc_output')(Y)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = keras.models.Model(inputs=X, outputs=Y, name='Model')

    return model


####################################################

def main(epochs=10, batch_size=50):
    ## Import the datasets.
    train_images, train_labels, validation_images, validation_labels, \
    test_images, test_labels, info = ImportDatasets.import_datasets()

    # print(train_images.shape)  # <class 'numpy.ndarray'>, (1189, 40, 60, 3)
    # print(train_labels.shape)  # <class 'numpy.ndarray'>, (1189, 3)
    # print(validation_images.shape)  # <class 'numpy.ndarray'>, (275, 40, 60, 3)
    # print(validation_labels.shape)  # <class 'numpy.ndarray'>, (275, 3)
    # print(test_images.shape)  # <class 'numpy.ndarray'>, (366, 40, 60, 3)
    # print(test_labels.shape)  # <class 'numpy.ndarray'>, (366, 3)

    ## Create the model
    model = ConvolutionalNeuralNetwork(
        input_shape=(info['height'], info['width'], info['channel']),
        dropout_rate=0.5)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    ## Train the model
    model.fit(x=train_images, y=train_labels, epochs=epochs, batch_size=batch_size)

    ## Test the model
    results = model.evaluate(x=test_images, y=test_labels)
    print("Loss = " + str(results[0]))
    print("Test Accuracy = " + str(results[1]))


####################################################

if __name__ == '__main__':
    main(epochs=5, batch_size=50)
