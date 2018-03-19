# main.py
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
import matplotlib.pyplot as plt
from NN import *

np.random.seed(1)
np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)


def generate_data():
    T = 20
    M = 2000
    dim_x = 15
    dim_a = 30
    np.random.seed(1)
    NN = NeuralNetwork(T, [dim_x, dim_a, 1], [None, 'tanh', 'sigmoid'])
    # print("W1: ")
    # print(NN.parameters["W1"])
    print("W2: ")
    print(NN.parameters["W2"])
    # print("b1: ")
    # print(NN.parameters["b1"])
    # print("b2: ")
    # print(NN.parameters["b2"])
    train_x = np.random.randn(dim_x, M, T)
    train_y = NN.predict(train_x)
    test_x = np.random.randn(dim_x, M // 4, T)
    test_y = NN.predict(test_x)
    return train_x, train_y, test_x, test_y


def main():
    train_x, train_y, test_x, test_y = generate_data()
    # print(train_x.shape)  # (5, 1000, 10)
    # print(train_y.shape)  # (1, 1000, 10)
    # print(test_x.shape)  # (5, 250, 10)
    # print(test_y.shape)  # (1, 250, 10)

    T = 20
    dim_x = 15
    dim_a = 20  # Changed!
    np.random.seed(100)  # Changed!
    NN = NeuralNetwork(T, [dim_x, dim_a, 1], [None, 'tanh', 'sigmoid'])

    score = NN.evaluate_accuracy(train_x, train_y)
    print("Training accuracy is: " + str(score))

    # print("W1: ")
    # print(NN.parameters["W1"])
    print("W2: ")
    print(NN.parameters["W2"])
    # print("b1: ")
    # print(NN.parameters["b1"])
    # print("b2: ")
    # print(NN.parameters["b2"])

    step_array = []
    loss_array = []

    for step in range(3000 + 1):
        loss = NN.train_onestep(train_x, train_y, 0.0075)

        if step % 100 == 0:
            print("step: " + str(step) + ", loss: " + str(loss))
            step_array.append(step)
            loss_array.append(loss)

    # print([i for i in NN.parameters])
    # print([NN.parameters[i].shape for i in NN.parameters])
    # print([i for i in NN.cache])
    # print([NN.cache[i].shape for i in NN.cache])

    fig = plt.figure()
    plt.plot(np.array(step_array), np.array(loss_array))
    plt.show()
    fig.savefig('training_loss.png', bbox_inches='tight')
    plt.close(fig)

    score = NN.evaluate_accuracy(train_x, train_y)
    print("Training accuracy is: " + str(score))

    score = NN.evaluate_accuracy(test_x, test_y)
    print("Test accuracy is: " + str(score))

    # print("W1: ")
    # print(NN.parameters["W1"])
    print("W2: ")
    print(NN.parameters["W2"])
    # print("b1: ")
    # print(NN.parameters["b1"])
    # print("b2: ")
    # print(NN.parameters["b2"])


if __name__ == '__main__':
    main()
