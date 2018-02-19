# NN.py
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
# 
# Implementation of deep neural networks.
# ==============================================================================

import numpy as np


class NeuralNetwork:
    def __init__(self, dim_layers, activation_layers):
        self.num_layers = len(dim_layers)
        self.dim_layers = dim_layers
        self.activation_layers = activation_layers
        self.parameters = {}
        self.grads = {}
        self.cache = {}

        for layer in range(1, self.num_layers):
            self.parameters['W' + str(layer)] = np.random.randn(dim_layers[layer], dim_layers[layer - 1]) \
                                                / np.sqrt(dim_layers[layer - 1])
            # For DNN, the initialization of W cannot be too small!
            self.parameters['b' + str(layer)] = np.zeros((dim_layers[layer], 1))

    def forward_onelayer(self, x, layer):
        self.cache['x' + str(layer - 1)] = x
        W = self.parameters['W' + str(layer)]
        b = self.parameters['b' + str(layer)]
        z = W.dot(x) + b
        self.cache['z' + str(layer)] = z

        activation = self.activation_layers[layer]
        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-z))
        elif activation == 'tanh':
            y = np.tanh(z)
        elif activation == 'softmax':
            y = np.exp(z)
            y = y / y.sum(axis=0)
        elif activation == "relu":
            y = np.maximum(z, 0)
        else:
            y = z.copy()

        return y

    def forward(self, x):
        for layer in range(1, self.num_layers):
            y = self.forward_onelayer(x, layer)
            x = y
        return y

    def backward_onelayer(self, dy, layer):
        M = dy.shape[1]
        z = self.cache['z' + str(layer)]
        activation = self.activation_layers[layer]
        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-z))
            dz = dy * y * (1 - y)
        elif activation == 'tanh':
            y = np.tanh(z)
            dz = dy * (1 - y ** 2)
        elif activation == 'softmax':
            y = np.exp(z)
            y = y / y.sum(axis=0)
            dz = dy * y * (1 - y)
        elif activation == "relu":
            dz = dy * (z > 0)
        else:
            dz = dy.copy()

        x = self.cache['x' + str(layer - 1)]
        dW = 1 / M * dz.dot(x.transpose())
        db = 1 / M * np.sum(dz, axis=1, keepdims=True)
        self.grads['dW' + str(layer)] = dW
        self.grads['db' + str(layer)] = db

        W = self.parameters['W' + str(layer)]
        dx = W.transpose().dot(dz)

        return dx

    def backward(self, dy):
        for layer in reversed(range(1, self.num_layers)):
            dx = self.backward_onelayer(dy, layer)
            dy = dx

    def loss(self, y, dataset_y):
        loss = - np.mean(dataset_y * np.log(y) + (1 - dataset_y) * np.log(1 - y))
        return loss

    def dloss(self, y, dataset_y):
        dloss = - dataset_y / y + (1 - dataset_y) / (1 - y)
        return dloss

    def update_onestep(self, learning_rate):
        for layer in range(1, self.num_layers):
            self.parameters['W' + str(layer)] -= learning_rate * self.grads['dW' + str(layer)]
            self.parameters['b' + str(layer)] -= learning_rate * self.grads['db' + str(layer)]

    def train_onestep(self, dataset_x, dataset_y, learning_rate=0.1):
        x = dataset_x.copy()
        y = self.forward(x)
        loss = self.loss(y, dataset_y)
        dloss = self.dloss(y, dataset_y)
        self.backward(dloss)
        self.update_onestep(learning_rate)
        return loss

    def predict(self, dataset_x):
        x = dataset_x.copy()
        y = self.forward(x)
        index = y >= 0.5
        y[index] = 1
        y[~index] = 0
        return y

    def evaluate_accuracy(self, dataset_x, dataset_y):
        y = self.predict(dataset_x)
        return np.sum(y == dataset_y) / len(y.ravel())
