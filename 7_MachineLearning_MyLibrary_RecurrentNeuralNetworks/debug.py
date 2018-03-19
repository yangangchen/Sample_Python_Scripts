import numpy as np
import matplotlib.pyplot as plt
from NN import *

np.random.seed(1)
np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)


# X: [d, None, T]
# y: [2, None, T]
T = 4
M = 10
NN = NeuralNetwork(T, [3, 5, 2], [None, 'tanh', 'softmax'])
np.random.seed(1)
x = np.random.randn(3, M, T)
a0 = np.random.randn(5, M)
w2 = np.random.randn(5, 5)
w1 = np.random.randn(5, 3)
NN.parameters['W1'] = np.hstack([w2, w1])
NN.parameters['W2'] = np.random.randn(2, 5)
NN.parameters['b1'] = np.random.randn(5, 1)
NN.parameters['b2'] = np.random.randn(2, 1)

y, a = NN.forward(x, a0)
print(a[4][1])
print(a.shape)
print(y[1][3])
print(y.shape)
print(x[1, 3, :])

input('Check')  # The same as Coursera's answer

#################

T = 1
M = 10
NN = NeuralNetwork(T, [3, 5, 2], [None, 'tanh', 'softmax'])
np.random.seed(1)
# x = np.random.randn(3, M, T)
x = np.expand_dims(np.random.randn(3, M), axis=2)
a0 = np.random.randn(5, M)
w1 = np.random.randn(5, 3)
w2 = np.random.randn(5, 5)
NN.parameters['W1'] = np.hstack([w2, w1])
NN.parameters['W2'] = np.random.randn(2, 5)
NN.parameters['b1'] = np.random.randn(5, 1)
NN.parameters['b2'] = np.random.randn(2, 1)

_, _ = NN.forward(x, a0)
# da = np.random.randn(5, M, T)
da = np.expand_dims(np.random.randn(5, M), axis=2)
dat = 0
for t in reversed(range(NN.T)):
    dat += da[:, :, t]
    for layer in reversed(range(1, NN.num_layers - 1)):
        dat = NN.backward_onetime_onelayer(dat, t, layer)
    dat = dat[:NN.dim_layers[NN.num_layers - 2], :]
print(NN.grads["dW1"])
print(NN.grads["db1"])

input('Check')  # Not the same as Coursera's answer

#################

T = 4
M = 10
NN = NeuralNetwork(T, [3, 5, 2], [None, 'tanh', 'softmax'])
np.random.seed(1)
x = np.random.randn(3, M, T)
a0 = np.random.randn(5, M)
w1 = np.random.randn(5, 3)
w2 = np.random.randn(5, 5)
NN.parameters['W1'] = np.hstack([w2, w1])
NN.parameters['W2'] = np.random.randn(2, 5)
NN.parameters['b1'] = np.random.randn(5, 1)
NN.parameters['b2'] = np.random.randn(2, 1)

_, _ = NN.forward(x, a0)
da = np.random.randn(5, M, T)
dat = 0
for t in reversed(range(NN.T)):
    dat += da[:, :, t]
    for layer in reversed(range(1, NN.num_layers - 1)):
        dat = NN.backward_onetime_onelayer(dat, t, layer)
    dat = dat[:NN.dim_layers[NN.num_layers - 2], :]

print(NN.grads["dW1"])
print(NN.grads["db1"])

input('Check')  # The same as Coursera's answer

#################

T = 4
M = 10
np.random.seed(1)
x = np.random.randn(3, M, T)
y = np.random.rand(2, M, T)
NN = NeuralNetwork(T, [3, 5, 2], [None, 'tanh', 'softmax'])
NN.train_onestep(x, y, learning_rate=0.1)

print(NN.grads["dW1"])
print(NN.grads["db1"])

input('Check')
