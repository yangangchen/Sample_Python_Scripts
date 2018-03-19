# NN.py
#
# Author: Yangang Chen
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
# Implementation of recurrent neural networks.
# ==============================================================================

import numpy as np


class NeuralNetwork:
    def __init__(self, T, dim_layers, activation_layers):
        self.T = T
        self.num_layers = len(dim_layers)
        self.dim_layers = dim_layers
        self.activation_layers = activation_layers
        self.parameters = {}
        self.grads = {}
        self.cache = {}

        for layer in range(1, self.num_layers):
            if layer == 1:
                dim_input = dim_layers[layer - 1]  + dim_layers[-2]
            else:
                dim_input = dim_layers[layer - 1]
            dim_output = dim_layers[layer]
            self.parameters['W' + str(layer)] \
                = np.random.randn(dim_output, dim_input) / np.sqrt(dim_input)
            self.parameters['b' + str(layer)] = np.zeros((dim_output, 1))

    def forward_onetime_onelayer(self, xt, t, layer):
        self.cache['x' + str(layer - 1) + '-' + str(t)] = xt
        W = self.parameters['W' + str(layer)]
        b = self.parameters['b' + str(layer)]
        zt = W.dot(xt) + b
        self.cache['z' + str(layer) + '-' + str(t)] = zt

        activation = self.activation_layers[layer]
        if activation == "sigmoid":
            yt = 1 / (1 + np.exp(-zt))
        elif activation == 'tanh':
            yt = np.tanh(zt)
        elif activation == 'softmax':
            yt = np.exp(zt)
            yt = yt / yt.sum(axis=0)
        elif activation == "relu":
            yt = np.maximum(zt, 0)
        else:
            yt = zt.copy()

        return yt

    def forward(self, x, a0=None):
        M = x.shape[1]
        y = np.zeros((self.dim_layers[-1], M, self.T))
        a = np.zeros((self.dim_layers[-2], M, self.T))
        at = np.zeros((self.dim_layers[-2], M)) if a0 is None else a0.copy()
        for t in range(self.T):
            xt = x[:, :, t]
            at = np.vstack([at, xt])
            for layer in range(1, self.num_layers - 1):
                at = self.forward_onetime_onelayer(at, t, layer)
            yt = self.forward_onetime_onelayer(at, t, self.num_layers - 1)
            a[:, :, t] = at
            y[:, :, t] = yt
        return y, a

    def backward_onetime_onelayer(self, dyt, t, layer):
        M = dyt.shape[1]
        zt = self.cache['z' + str(layer) + '-' + str(t)]
        activation = self.activation_layers[layer]
        if activation == "sigmoid":
            yt = 1 / (1 + np.exp(-zt))
            dzt = dyt * yt * (1 - yt)
        elif activation == 'tanh':
            yt = np.tanh(zt)
            dzt = dyt * (1 - yt ** 2)
        elif activation == 'softmax':
            yt = np.exp(zt)
            yt = yt / yt.sum(axis=0)
            dzt = dyt * yt * (1 - yt)
        elif activation == "relu":
            dzt = dyt * (zt > 0)
        else:
            dzt = dyt.copy()
        # print(dzt)
        # print('check dzt')

        xt = self.cache['x' + str(layer - 1) + '-' + str(t)]
        dW = 1 / M * dzt.dot(xt.transpose())
        db = 1 / M * np.sum(dzt, axis=1, keepdims=True)
        # print(db)
        # print('check db')
        if t == self.T - 1:
            self.grads['dW' + str(layer)] = dW
            self.grads['db' + str(layer)] = db
        else:
            self.grads['dW' + str(layer)] += dW
            self.grads['db' + str(layer)] += db

        W = self.parameters['W' + str(layer)]
        dxt = W.transpose().dot(dzt)

        return dxt

    def backward(self, dy, daT=None):
        M = dy.shape[1]
        dat = np.zeros((self.dim_layers[-2], M)) if daT is None else daT.copy()
        for t in reversed(range(self.T)):
            dyt = dy[:, :, t]
            dyt = self.backward_onetime_onelayer(dyt, t, self.num_layers - 1)
            dat += dyt
            for layer in reversed(range(1, self.num_layers - 1)):
                dat = self.backward_onetime_onelayer(dat, t, layer)
            dat = dat[:self.dim_layers[self.num_layers - 2], :]

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
        y, _ = self.forward(x)
        loss = self.loss(y, dataset_y)
        dloss = self.dloss(y, dataset_y)
        self.backward(dloss)
        self.update_onestep(learning_rate)
        return loss

    def predict(self, dataset_x):
        x = dataset_x.copy()
        y, _ = self.forward(x)
        index = y >= 0.5
        y[index] = 1
        y[~index] = 0
        return y

    def evaluate_accuracy(self, dataset_x, dataset_y):
        y = self.predict(dataset_x)
        return np.sum(y == dataset_y) / len(y.ravel())
