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
    ########## Initialization ##########
    def __init__(self, learning_task, T, dim_layers,
                 activation_hidden='tanh', activation_output='default', loss_function='default'):
        self.learning_task = learning_task

        self.T = T
        self.num_layers = len(dim_layers)
        self.dim_layers = dim_layers

        self.activation_hidden = activation_hidden

        if activation_output == 'default':
            if self.learning_task == 'regression':
                self.activation_output = 'identity'
            elif self.learning_task == 'classification-two-classes':
                self.activation_output = 'sigmoid'
            elif self.learning_task == 'classification':
                self.activation_output = 'softmax'
        else:
            self.activation_output = activation_output

        if loss_function == 'default':
            if self.learning_task == 'regression':
                self.loss_function = 'mean-squared-error'
            elif self.learning_task == 'classification-two-classes':
                self.loss_function = 'cross-entropy-two-classes'
            elif self.learning_task == 'classification':
                self.loss_function = 'cross-entropy'
        else:
            self.loss_function = loss_function

        self.parameters = {}
        self.grads = {}
        self.cache = {}

        for layer in range(1, self.num_layers):
            if layer == 1:
                dim_input = dim_layers[layer - 1] + dim_layers[-2]
            else:
                dim_input = dim_layers[layer - 1]
            dim_output = dim_layers[layer]
            self.parameters['W' + str(layer)] = \
                np.random.randn(dim_input, dim_output) / np.sqrt(dim_input)
            # For DNN, the initialization of W cannot be too small!
            self.parameters['b' + str(layer)] = np.zeros((1, dim_output))

    ########## Forward Algorithm ##########
    def forward_onetime_onelayer(self, xt, t, layer):
        self.cache['x' + str(layer - 1) + '-' + str(t)] = xt
        W = self.parameters['W' + str(layer)]
        b = self.parameters['b' + str(layer)]
        zt = xt.dot(W) + b
        self.cache['z' + str(layer) + '-' + str(t)] = zt

        activation = self.activation_output if layer == self.num_layers - 1 else self.activation_hidden
        if activation == "sigmoid":
            yt = 1 / (1 + np.exp(-zt))
        elif activation == 'tanh':
            yt = np.tanh(zt)
        elif activation == 'softmax':
            yt = np.exp(zt)
            yt = yt / yt.sum(axis=1, keepdims=True)
        elif activation == "relu":
            yt = np.maximum(zt, 0)
        elif activation == 'identity':
            yt = zt.copy()

        return yt

    def forward(self, x, a0=None):
        M = x.shape[0]
        y = np.zeros((M, self.dim_layers[-1], self.T))
        a = np.zeros((M, self.dim_layers[-2], self.T))
        at = np.zeros((M, self.dim_layers[-2])) if a0 is None else a0.copy()
        for t in range(self.T):
            xt = x[:, :, t]
            at = np.hstack([at, xt])
            for layer in range(1, self.num_layers - 1):
                at = self.forward_onetime_onelayer(at, t, layer)
            yt = self.forward_onetime_onelayer(at, t, self.num_layers - 1)
            a[:, :, t] = at
            y[:, :, t] = yt
        return y, a

    ########## Backward Algorithm ##########
    def backward_onetime_onelayer(self, dyt, t, layer):
        zt = self.cache['z' + str(layer) + '-' + str(t)]
        activation = self.activation_output if layer == self.num_layers - 1 else self.activation_hidden
        if activation == "sigmoid":
            yt = 1 / (1 + np.exp(-zt))
            dzt = dyt * yt * (1 - yt)
        elif activation == 'tanh':
            yt = np.tanh(zt)
            dzt = dyt * (1 - yt ** 2)
        elif activation == 'softmax':
            yt = np.exp(zt)
            yt = yt / yt.sum(axis=1, keepdims=True)
            dzt = dyt * yt * (1 - yt)
        elif activation == "relu":
            dzt = dyt * (zt > 0)
        else:
            dzt = dyt.copy()

        M = dyt.shape[0]
        W = self.parameters['W' + str(layer)]
        xt = self.cache['x' + str(layer - 1) + '-' + str(t)]
        dxt = dzt.dot(W.transpose())

        dW = 1 / M * xt.transpose().dot(dzt)
        db = 1 / M * np.sum(dzt, axis=0, keepdims=True)
        if t == self.T - 1:
            self.grads['dW' + str(layer)] = dW
            self.grads['db' + str(layer)] = db
        else:
            self.grads['dW' + str(layer)] += dW
            self.grads['db' + str(layer)] += db

        return dxt

    def backward(self, dy, daT=None):
        M = dy.shape[0]
        dat = np.zeros((M, self.dim_layers[-2])) if daT is None else daT.copy()
        for t in reversed(range(self.T)):
            dyt = dy[:, :, t]
            dyt = self.backward_onetime_onelayer(dyt, t, self.num_layers - 1)
            dat += dyt
            for layer in reversed(range(1, self.num_layers - 1)):
                dat = self.backward_onetime_onelayer(dat, t, layer)
            dat = dat[:, :self.dim_layers[self.num_layers - 2]]

    ########## Loss function ##########
    def loss(self, y, dataset_y):
        assert y.shape == dataset_y.shape
        M = y.shape[0]
        if self.loss_function == 'mean-squared-error':
            loss = 1 / (2 * M) * np.sum((y - dataset_y) ** 2)
        elif self.loss_function == 'cross-entropy-two-classes':
            assert y.shape[1] == 1
            loss = - 1 / M * np.sum(dataset_y * np.log(y) + (1 - dataset_y) * np.log(1 - y))
        elif self.loss_function == 'cross-entropy':
            loss = - 1 / M * np.sum(dataset_y * np.log(y))
        return loss

    def dloss(self, y, dataset_y):
        assert y.shape == dataset_y.shape
        if self.loss_function == 'mean-squared-error':
            dloss = y - dataset_y
        elif self.loss_function == 'cross-entropy-two-classes':
            dloss = - dataset_y / y + (1 - dataset_y) / (1 - y)
        elif self.loss_function == 'cross-entropy':
            dloss = - dataset_y / y
        return dloss

    ########## Training ##########
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

    ########## Testing ##########
    def predict(self, dataset_x):
        x = dataset_x.copy()
        y, _ = self.forward(x)
        if self.learning_task == 'classification-two-classes':
            y = (y >= 0.5).astype(int)
        elif self.learning_task == 'classification':
            y_max = np.amax(y, axis=1, keepdims=True)
            y = (y > y_max - 1e-6).astype(int)
        return y

    def evaluate_accuracy(self, dataset_x, dataset_y):
        y = self.predict(dataset_x)
        if self.learning_task == 'regression':
            score = - self.loss(y, dataset_y)
        elif self.learning_task == 'classification-two-classes' or self.learning_task == 'classification-two-classes':
            score = np.sum(y == dataset_y) / len(y.ravel())
        return score