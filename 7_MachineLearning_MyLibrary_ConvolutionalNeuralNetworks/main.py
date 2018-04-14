import numpy as np
import h5py
import matplotlib.pyplot as plt
from NN import *

np.random.seed(1)
np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf, suppress=True)


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_y = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_y = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_x = train_x / 255.
    test_x = test_x / 255.
    train_y = np.expand_dims(train_y, axis=1)
    test_y = np.expand_dims(test_y, axis=1)

    return train_x, train_y, test_x, test_y, classes


def main():
    train_x, train_y, test_x, test_y, classes = load_data()
    # print(train_x.shape)  # (209, 64, 64, 3)
    # print(train_y.shape)  # (209, 1)
    # print(test_x.shape)  # (50, 64, 64, 3)
    # print(test_y.shape)  # (50, 1)

    NN = NeuralNetwork(learning_task='classification-two-classes',
                       dim_input=(None, 64, 64, 3),
                       dim_conv_kernel=[None, (7, 7), (5, 5)],
                       dim_conv_stride=[None, (3, 3), (2, 2)],
                       dim_conv_channel=[None, 16, 32],
                       dim_maxpool_kernel=[None, (2, 2), (2, 2)],
                       dim_maxpool_stride=[None, (2, 2), (2, 2)],
                       dim_fc=[None, 7, 5, 1])

    score = NN.evaluate_accuracy(train_x, train_y)
    print("Training accuracy (before training) is: " + str(score))

    step_array = []
    loss_array = []

    for step in range(3000 + 1):
        loss = NN.train_onestep(train_x, train_y, 0.01)

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
    print("Training accuracy (after training) is: " + str(score))

    score = NN.evaluate_accuracy(test_x, test_y)
    print("Test accuracy is: " + str(score))


if __name__ == '__main__':
    main()
