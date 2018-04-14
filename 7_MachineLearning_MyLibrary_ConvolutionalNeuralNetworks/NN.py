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
# Implementation of convolutional neural networks.
# ==============================================================================

import numpy as np


class NeuralNetwork:
    ########## Initialization ##########
    def __init__(self, learning_task,
                 dim_input,
                 dim_conv_kernel, dim_conv_stride, dim_conv_channel,
                 dim_maxpool_kernel, dim_maxpool_stride,
                 dim_fc,
                 pad_conv=True,
                 activation_conv='relu', activation_fc_hidden='relu', activation_fc_output='default',
                 loss_function='default'):
        self.learning_task = learning_task

        self.dim_input = dim_input
        assert self.dim_input[0] is None
        self.num_conv_layers = len(dim_conv_stride)
        self.dim_conv_kernel = dim_conv_kernel
        self.dim_conv_stride = dim_conv_stride
        self.dim_conv_channel = dim_conv_channel
        assert self.dim_conv_kernel[0] is None
        assert self.dim_conv_stride[0] is None
        assert self.dim_conv_channel[0] is None
        self.dim_conv_channel[0] = self.dim_input[3]
        self.pad_conv = pad_conv
        if self.pad_conv:
            self.dim_conv_pad = [None]
            for d_k in self.dim_conv_kernel[1:]:
                self.dim_conv_pad.append(np.array(d_k) // 2)
        self.activation_conv = activation_conv

        self.dim_maxpool_kernel = dim_maxpool_kernel
        self.dim_maxpool_stride = dim_maxpool_stride
        assert self.dim_maxpool_kernel[0] is None
        assert self.dim_maxpool_stride[0] is None

        self.dim_conv_maxpool = [dim_input[1:]]
        d_h, d_w = dim_input[1], dim_input[2]
        for layer in range(1, self.num_conv_layers):
            k_h, k_w = self.dim_conv_kernel[layer]
            s_h, s_w = self.dim_conv_stride[layer]
            if self.pad_conv:
                p_h, p_w = self.dim_conv_pad[layer]
                d_h = (d_h - k_h + 2 * p_h) // s_h + 1
                d_w = (d_w - k_w + 2 * p_w) // s_w + 1
            else:
                d_h = (d_h - k_h) // s_h + 1
                d_w = (d_w - k_w) // s_w + 1
            self.dim_conv_maxpool.append((d_h, d_w, self.dim_conv_channel[layer]))
            k_h, k_w = self.dim_maxpool_kernel[layer]
            s_h, s_w = self.dim_maxpool_stride[layer]
            d_h = (d_h - k_h) // s_h + 1
            d_w = (d_w - k_w) // s_w + 1
            self.dim_conv_maxpool.append((d_h, d_w, self.dim_conv_channel[layer]))

        self.num_fc_layers = len(dim_fc)
        self.dim_fc = dim_fc
        assert self.dim_fc[0] is None
        self.dim_fc[0] = d_h * d_w * self.dim_conv_channel[-1]
        self.activation_fc_hidden = activation_fc_hidden

        print('Dimensions of the convolutional neural network: \n' + str(self.dim_conv_maxpool + self.dim_fc))

        if activation_fc_output == 'default':
            if self.learning_task == 'regression':
                self.activation_fc_output = 'identity'
            elif self.learning_task == 'classification-two-classes':
                self.activation_fc_output = 'sigmoid'
            elif self.learning_task == 'classification':
                self.activation_fc_output = 'softmax'
        else:
            self.activation_fc_output = activation_fc_output

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

        for layer in range(1, self.num_conv_layers):
            self.parameters['W_conv' + str(layer)] = \
                np.random.randn(dim_conv_kernel[layer][0], dim_conv_kernel[layer][1],
                                dim_conv_channel[layer - 1], dim_conv_channel[layer]) \
                / np.sqrt(dim_conv_kernel[layer][0] * dim_conv_kernel[layer][1] * dim_conv_channel[layer - 1])
            # For DNN, the initialization of W cannot be too small!
            self.parameters['b_conv' + str(layer)] = np.zeros(dim_conv_channel[layer])

        for layer in range(1, self.num_fc_layers):
            self.parameters['W_fc' + str(layer)] = \
                np.random.randn(dim_fc[layer - 1], dim_fc[layer]) / np.sqrt(dim_fc[layer - 1])
            # For DNN, the initialization of W cannot be too small!
            self.parameters['b_fc' + str(layer)] = np.zeros((1, dim_fc[layer]))

    ########## Forward Algorithm ##########
    def forward_conv_onelayer(self, x, layer):
        self.cache['x_conv' + str(layer - 1)] = x
        assert x.shape[3] == self.dim_conv_channel[layer - 1]
        W = self.parameters['W_conv' + str(layer)]
        b = self.parameters['b_conv' + str(layer)]
        M = x.shape[0]
        d_h, d_w, d_c = x.shape[1], x.shape[2], self.dim_conv_channel[layer]
        k_h, k_w = self.dim_conv_kernel[layer]
        s_h, s_w = self.dim_conv_stride[layer]
        if self.pad_conv:
            p_h, p_w = self.dim_conv_pad[layer]
            d_h = (d_h - k_h + 2 * p_h) // s_h + 1
            d_w = (d_w - k_w + 2 * p_w) // s_w + 1
            x_pad = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            d_h = (d_h - k_h) // s_h + 1
            d_w = (d_w - k_w) // s_w + 1
            x_pad = x.copy()
        z = np.zeros((M, d_h, d_w, d_c))
        for h in range(d_h):
            for w in range(d_w):
                z[:, h, w, :] = np.tensordot(
                    x_pad[:, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w, :], W, axes=([1, 2, 3], [0, 1, 2])
                ) + np.expand_dims(b, axis=0)
        self.cache['z_conv' + str(layer)] = z

        if self.activation_conv == "sigmoid":
            y = 1 / (1 + np.exp(-z))
        elif self.activation_conv == 'tanh':
            y = np.tanh(z)
        elif self.activation_conv == 'softmax':
            y = np.exp(z)
            y = y / y.sum(axis=(1, 2, 3), keepdims=True)
        elif self.activation_conv == "relu":
            y = np.maximum(z, 0)
        elif self.activation_conv == 'identity':
            y = z.copy()

        return y

    def forward_maxpool_onelayer(self, x, layer):
        self.cache['x_maxpool' + str(layer)] = x
        M = x.shape[0]
        d_h, d_w, d_c = x.shape[1], x.shape[2], self.dim_conv_channel[layer]
        k_h, k_w = self.dim_maxpool_kernel[layer]
        s_h, s_w = self.dim_maxpool_stride[layer]
        d_h = (d_h - k_h) // s_h + 1
        d_w = (d_w - k_w) // s_w + 1
        y = np.zeros((M, d_h, d_w, d_c))
        for h in range(d_h):
            for w in range(d_w):
                y[:, h, w, :] = np.max(x[:, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w, :], axis=(1, 2))
        return y

    def forward_fc_onelayer(self, x, layer):
        self.cache['x_fc' + str(layer - 1)] = x
        W = self.parameters['W_fc' + str(layer)]
        b = self.parameters['b_fc' + str(layer)]
        z = x.dot(W) + b
        self.cache['z_fc' + str(layer)] = z

        activation_fc = self.activation_fc_output if layer == self.num_fc_layers - 1 else self.activation_fc_hidden
        if activation_fc == "sigmoid":
            y = 1 / (1 + np.exp(-z))
        elif activation_fc == 'tanh':
            y = np.tanh(z)
        elif activation_fc == 'softmax':
            y = np.exp(z)
            y = y / y.sum(axis=1, keepdims=True)
        elif activation_fc == "relu":
            y = np.maximum(z, 0)
        elif activation_fc == 'identity':
            y = z.copy()

        return y

    def forward(self, x):
        for layer in range(1, self.num_conv_layers):
            y = self.forward_conv_onelayer(x, layer)
            y = self.forward_maxpool_onelayer(y, layer)
            x = y
        x = x.reshape((x.shape[0], -1))
        for layer in range(1, self.num_fc_layers):
            y = self.forward_fc_onelayer(x, layer)
            x = y
        return y

    ########## Backward Algorithm ##########
    def backward_conv_onelayer(self, dy, layer):
        z = self.cache['z_conv' + str(layer)]
        if self.activation_conv == "sigmoid":
            y = 1 / (1 + np.exp(-z))
            dz = dy * y * (1 - y)
        elif self.activation_conv == 'tanh':
            y = np.tanh(z)
            dz = dy * (1 - y ** 2)
        elif self.activation_conv == 'softmax':
            y = np.exp(z)
            y = y / y.sum(axis=(1, 2, 3), keepdims=True)
            dz = dy * y * (1 - y)
        elif self.activation_conv == "relu":
            dz = dy * (z > 0)
        elif self.activation_conv == 'identity':
            dz = dy.copy()

        M = dy.shape[0]
        d_h, d_w = dz.shape[1], dz.shape[2]
        k_h, k_w = self.dim_conv_kernel[layer]
        s_h, s_w = self.dim_conv_stride[layer]
        W = self.parameters['W_conv' + str(layer)]
        x = self.cache['x_conv' + str(layer - 1)]
        if self.pad_conv:
            p_h, p_w = self.dim_conv_pad[layer]
            x_pad = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            x_pad = x.copy()
        dx = np.zeros(x_pad.shape)
        for h in range(d_h):
            for w in range(d_w):
                dx[:, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w, :] += np.tensordot(
                    dz[:, h, w, :], W, axes=(1, 3))
        if self.pad_conv:
            dx = dx[:, p_h:-p_h, p_w:-p_w, :]

        dW = np.zeros(W.shape)
        for h in range(d_h):
            for w in range(d_w):
                dW += np.tensordot(
                    x_pad[:, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w, :], dz[:, h, w, :],
                    axes=(0, 0))
        dW /= M
        db = np.sum(dz, axis=(0, 1, 2), keepdims=False)
        db /= M
        self.grads['dW_conv' + str(layer)] = dW
        self.grads['db_conv' + str(layer)] = db

        return dx

    def backward_maxpool_onelayer(self, dy, layer):
        d_h, d_w = dy.shape[1], dy.shape[2]
        k_h, k_w = self.dim_maxpool_kernel[layer]
        s_h, s_w = self.dim_maxpool_stride[layer]
        x = self.cache['x_maxpool' + str(layer)]
        dx = np.zeros(x.shape)
        for h in range(d_h):
            for w in range(d_w):
                x0 = x[:, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w, :]
                x0_max = np.amax(x0, axis=(1, 2), keepdims=True)
                x0_mask = (x0 > x0_max - 1e-6)
                dx[:, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w, :] += \
                    x0_mask * dy[:, [h], :, :][:, :, [w], :]
        return dx

    def backward_fc_onelayer(self, dy, layer):
        z = self.cache['z_fc' + str(layer)]
        activation_fc = self.activation_fc_output if layer == self.num_fc_layers - 1 else self.activation_fc_hidden
        if activation_fc == "sigmoid":
            y = 1 / (1 + np.exp(-z))
            dz = dy * y * (1 - y)
        elif activation_fc == 'tanh':
            y = np.tanh(z)
            dz = dy * (1 - y ** 2)
        elif activation_fc == 'softmax':
            y = np.exp(z)
            y = y / y.sum(axis=1, keepdims=True)
            dz = dy * y * (1 - y)
        elif activation_fc == "relu":
            dz = dy * (z > 0)
        elif activation_fc == 'identity':
            dz = dy.copy()

        M = dy.shape[0]
        W = self.parameters['W_fc' + str(layer)]
        x = self.cache['x_fc' + str(layer - 1)]
        dx = dz.dot(W.transpose())

        dW = 1 / M * x.transpose().dot(dz)
        db = 1 / M * np.sum(dz, axis=0, keepdims=True)
        self.grads['dW_fc' + str(layer)] = dW
        self.grads['db_fc' + str(layer)] = db

        return dx

    def backward(self, dy):
        for layer in reversed(range(1, self.num_fc_layers)):
            dx = self.backward_fc_onelayer(dy, layer)
            dy = dx
        dy = dy.reshape((dy.shape[0],) + self.dim_conv_maxpool[-1])
        for layer in reversed(range(1, self.num_conv_layers)):
            dy = self.backward_maxpool_onelayer(dy, layer)
            dx = self.backward_conv_onelayer(dy, layer)
            dy = dx

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
        for layer in range(1, self.num_conv_layers):
            self.parameters['W_conv' + str(layer)] -= learning_rate * self.grads['dW_conv' + str(layer)]
            self.parameters['b_conv' + str(layer)] -= learning_rate * self.grads['db_conv' + str(layer)]
        for layer in range(1, self.num_fc_layers):
            self.parameters['W_fc' + str(layer)] -= learning_rate * self.grads['dW_fc' + str(layer)]
            self.parameters['b_fc' + str(layer)] -= learning_rate * self.grads['db_fc' + str(layer)]

    def train_onestep(self, dataset_x, dataset_y, learning_rate=0.1):
        x = dataset_x.copy()
        y = self.forward(x)
        loss = self.loss(y, dataset_y)
        dloss = self.dloss(y, dataset_y)
        self.backward(dloss)
        self.update_onestep(learning_rate)
        return loss

    ########## Testing ##########
    def predict(self, dataset_x):
        x = dataset_x.copy()
        y = self.forward(x)
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
