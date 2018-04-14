import numpy as np
import matplotlib.pyplot as plt
from NN import *

np.random.seed(1)
np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)


# X: [None, d, T]
# y: [None, 2, T]
T = 4
M = 10
NN = NeuralNetwork(learning_task='classification', T=T, dim_layers=[3, 5, 2])
np.random.seed(1)
x = np.random.randn(3, M, T).transpose((1, 0, 2))
a0 = np.random.randn(5, M).transpose()
w2 = np.random.randn(5, 5)
w1 = np.random.randn(5, 3)
NN.parameters['W1'] = np.hstack([w2, w1]).transpose()
NN.parameters['W2'] = np.random.randn(2, 5).transpose()
NN.parameters['b1'] = np.random.randn(1, 5)
NN.parameters['b2'] = np.random.randn(1, 2)

y, a = NN.forward(x, a0)

print(a[1, 4])
# Answer: [-0.99999375 0.77911235 -0.99861469 -0.99833267]
print(a.shape)
# Answer: (10, 5, 4)
print(y[3, 1])
# Answer: [ 0.79560373 0.86224861 0.11118257 0.81515947]
print(y.shape)
# Answer: (10, 2, 4)
print(x[3, 1, :])
# Answer: [-1.1425182 -0.34934272 -0.20889423 0.58662319]
input('Check')

#################

T = 1
M = 10
NN = NeuralNetwork(learning_task='classification', T=T, dim_layers=[3, 5, 2])
np.random.seed(1)
# x = np.random.randn(3, M, T).transpose((1, 0, 2))
x = np.expand_dims(np.random.randn(3, M).transpose(), axis=2)
a0 = np.random.randn(5, M).transpose()
w1 = np.random.randn(5, 3)
w2 = np.random.randn(5, 5)
NN.parameters['W1'] = np.hstack([w2, w1]).transpose()
NN.parameters['W2'] = np.random.randn(2, 5).transpose()
NN.parameters['b1'] = np.random.randn(1, 5)
NN.parameters['b2'] = np.random.randn(1, 2)

_, _ = NN.forward(x, a0)
# da = np.random.randn(5, M, T).transpose((1, 0, 2))
da = np.expand_dims(np.random.randn(5, M).transpose(), axis=2)
dat = 0
for t in reversed(range(NN.T)):
    dat += da[:, :, t]
    for layer in reversed(range(1, NN.num_layers - 1)):
        dat = NN.backward_onetime_onelayer(dat, t, layer)
    dat = dat[:, :NN.dim_layers[NN.num_layers - 2]]

print(NN.grads["dW1"])
# Answer:
print(NN.grads["db1"])
# Answer:
input('Check')

#################

T = 4
M = 10
NN = NeuralNetwork(learning_task='classification', T=T, dim_layers=[3, 5, 2])
np.random.seed(1)
x = np.random.randn(3, M, T).transpose((1, 0, 2))
a0 = np.random.randn(5, M).transpose()
w1 = np.random.randn(5, 3)
w2 = np.random.randn(5, 5)
NN.parameters['W1'] = np.hstack([w2, w1]).transpose()
NN.parameters['W2'] = np.random.randn(2, 5).transpose()
NN.parameters['b1'] = np.random.randn(1, 5)
NN.parameters['b2'] = np.random.randn(1, 2)

_, _ = NN.forward(x, a0)
da = np.random.randn(5, M, T).transpose((1, 0, 2))
dat = 0
for t in reversed(range(NN.T)):
    dat += da[:, :, t]
    for layer in reversed(range(1, NN.num_layers - 1)):
        dat = NN.backward_onetime_onelayer(dat, t, layer)
    dat = dat[:, :NN.dim_layers[NN.num_layers - 2]]

print(dat)
# Answer:
print(dat[3, 2])
# Answer: -0.314942375127
print(NN.grads["dW1"])
# Answer:
print(NN.grads["dW1"][5 + 1, 3])
# Answer: 1.12641044965
print(NN.grads["dW1"][2, 1])
# Answer: 0.230333312658
print(NN.grads["db1"])
# Answer:
print(NN.grads["db1"][0, 4])
# Answer: -0.074747722
input('Check')

#################

T = 4
M = 10
np.random.seed(1)
x = np.random.randn(M, 3, T)
y = np.random.randn(M, 2, T)
NN = NeuralNetwork(learning_task='classification', T=T, dim_layers=[3, 5, 2])
NN.train_onestep(x, y, learning_rate=0.1)

print(NN.grads["dW1"])
# Answer:
print(NN.grads["db1"])
# Answer:
input('Check')