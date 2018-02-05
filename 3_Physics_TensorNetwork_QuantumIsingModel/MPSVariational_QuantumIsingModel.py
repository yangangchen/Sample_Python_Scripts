# MPSVariational_QuantumIsingModel.py
#
# Author: Yangang Chen
#
# Copyright (C) 2017  Yangang Chen
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

"""Use MPS variational algorithm to compute the ground state energy and magnatization
of the 1D finite Quantum Ising Model. See

https://arxiv.org/pdf/1008.3477.pdf

for a description. The code is only implemented by numpy and scipy library."""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

import numpy as np

spin_dim = 2
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
si = np.identity(spin_dim)


def hamiltonian(h):
    H = np.zeros((3, 3, spin_dim, spin_dim))
    H[0, 0, :, :] = si
    H[1, 0, :, :] = sz
    H[2, 0, :, :] = h * sx
    H[2, 1, :, :] = sz
    H[2, 2, :, :] = si
    return H


def magnatization(direction):
    M = np.zeros((2, 2, spin_dim, spin_dim))
    M[0, 0, :, :] = si
    if direction == 'x':
        M[1, 0, :, :] = sx
    elif direction == 'z':
        M[1, 0, :, :] = sz
    M[1, 1, :, :] = si
    return M


def mpo(op, L):
    oplist = []
    for l in range(L):
        if l == 0:
            oplist.append(np.expand_dims(op[-1, :, :, :], axis=0))
        elif l == L - 1:
            oplist.append(np.expand_dims(op[:, 0, :, :], axis=1))
        else:
            oplist.append(op)
    return oplist


class TensorNetwork:
    def __init__(self, L, bond_dim, spin_dim):
        self.bond_dim = bond_dim
        self.spin_dim = spin_dim
        self.L = L

        self._initiate_mps()
        self._normalize_mps()

    def _initiate_mps(self):
        self.Alist = []
        for l in range(self.L):
            if l == self.L - 1:
                self.Alist.append(np.random.randn(min(self.spin_dim ** (l + 1), self.bond_dim),
                                                  1,
                                                  self.spin_dim))
            else:
                self.Alist.append(np.random.randn(min(self.spin_dim ** l, self.bond_dim),
                                                  min(self.spin_dim ** (l + 1), self.bond_dim),
                                                  self.spin_dim))

    def _normalize_onesite(self, A, A_next):
        A_mat = np.transpose(A, (0, 2, 1)).reshape((-1, A.shape[1]))
        U, S, V = np.linalg.svd(A_mat, full_matrices=False)
        A = np.transpose(U.reshape((A.shape[0], A.shape[2], A.shape[1])), (0, 2, 1))
        A_next = np.tensordot(np.diag(S).dot(V), A_next, axes=(1, 0))
        return A, A_next

    def _normalize_mps(self):
        for l in range(self.L):
            if l == self.L - 1:
                self.Alist[l] = self.Alist[l] / np.linalg.norm(self.Alist[l].ravel())
            else:
                self.Alist[l], self.Alist[l + 1] = self._normalize_onesite(self.Alist[l], self.Alist[l + 1])

    def _contract_AA(self, A):
        AA = np.tensordot(A, np.conj(A), axes=(2, 2))
        AA = np.transpose(AA, (0, 2, 1, 3))
        return AA

    def _contract_AHA(self, A, H):
        AH = np.tensordot(A, H, axes=(2, 2))
        AHA = np.tensordot(AH, np.conj(A), axes=(4, 2))
        AHA = np.transpose(AHA, (0, 2, 4, 1, 3, 5))
        return AHA

    def compute_norm(self):
        Anorm = np.array([[1]])
        for l in range(self.L):
            AA = self._contract_AA(self.Alist[l])
            Anorm = np.tensordot(Anorm, AA, axes=((0, 1), (0, 1)))
        return Anorm[0, 0]

    def compute_expectation(self, oplist):
        value = np.array([[[1]]])
        for l in range(self.L):
            AopA = self._contract_AHA(self.Alist[l], oplist[l])
            value = np.tensordot(value, AopA, axes=((0, 1, 2), (0, 1, 2)))
        return value[0, 0, 0] / self.L

    def _AHA_left_right_initial(self, Hlist):
        AHAlist = []
        for l in range(self.L):
            AHAlist.append(self._contract_AHA(self.Alist[l], Hlist[l]))
        Leftlist = [np.array([[[1]]])]
        for l in range(self.L):
            Leftlist.append(np.tensordot(
                Leftlist[-1], AHAlist[l], axes=((0, 1, 2), (0, 1, 2))))
        Rightlist = [np.array([[[1]]])]
        for l in reversed(range(self.L)):
            Rightlist.append(np.tensordot(
                AHAlist[l], Rightlist[-1], axes=((3, 4, 5), (0, 1, 2))))
        Rightlist = Rightlist[::-1]
        return AHAlist, Leftlist, Rightlist

    def _AHA_left_right_onesite(self, Hlist, AHAlist, Leftlist, Rightlist, l):
        AHAlist[l] = self._contract_AHA(self.Alist[l], Hlist[l])
        Leftlist[l + 1] = np.tensordot(Leftlist[l], AHAlist[l], axes=((0, 1, 2), (0, 1, 2)))
        Rightlist[l] = np.tensordot(AHAlist[l], Rightlist[l + 1], axes=((3, 4, 5), (0, 1, 2)))
        return AHAlist, Leftlist, Rightlist

    def _environment_onesite(self, Hlist, Leftlist, Rightlist, l):
        Env = np.tensordot(Leftlist[l], Hlist[l], axes=(1, 0))
        Env = np.tensordot(Env, Rightlist[l + 1], axes=(2, 1))
        Env = np.transpose(Env, (0, 4, 2, 1, 5, 3))
        return Env

    def _varitional_onesite(self, Hlist, AHAlist, Leftlist, Rightlist, l):
        Env = self._environment_onesite(Hlist, Leftlist, Rightlist, l)
        Env_mat = Env.reshape((Env.shape[0] * Env.shape[1] * Env.shape[2],
                               Env.shape[3] * Env.shape[4] * Env.shape[5]))
        D, V = np.linalg.eigh(Env_mat)
        Anew = V[:, 0].reshape(self.Alist[l].shape)
        if l == self.L - 1:
            self.Alist[l] = Anew / np.linalg.norm(Anew.ravel())
        else:
            self.Alist[l], self.Alist[l + 1] = self._normalize_onesite(Anew, self.Alist[l + 1])
        AHAlist, Leftlist, Rightlist = self._AHA_left_right_onesite(Hlist, AHAlist, Leftlist, Rightlist, l)
        return AHAlist, Leftlist, Rightlist

    def _varitional_oneiter(self, Hlist, AHAlist, Leftlist, Rightlist):
        for l in range(self.L):
            AHAlist, Leftlist, Rightlist = self._varitional_onesite(Hlist, AHAlist, Leftlist, Rightlist, l)
        return AHAlist, Leftlist, Rightlist

    def variational_algorithm(self, Hlist, max_iter):
        AHAlist, Leftlist, Rightlist = self._AHA_left_right_initial(Hlist)
        for iter in range(max_iter):
            AHAlist, Leftlist, Rightlist = self._varitional_oneiter(Hlist, AHAlist, Leftlist, Rightlist)


def main():
    L = 40
    max_iter = 20
    bond_dim = 10
    h_array = np.linspace(0, 2, 41)
    energy_array = np.zeros(len(h_array))
    magx_array = np.zeros(len(h_array))

    Mxlist = mpo(magnatization(direction='x'), L)

    for i in range(len(h_array)):
        print(i)
        h = h_array[i]
        Hlist = mpo(hamiltonian(h), L)

        TN = TensorNetwork(L=L, bond_dim=bond_dim, spin_dim=spin_dim)

        TN.variational_algorithm(Hlist=Hlist, max_iter=max_iter)

        energy_array[i] = TN.compute_expectation(Hlist)
        magx_array[i] = TN.compute_expectation(Mxlist)

    result = np.vstack([h_array, energy_array, magx_array])

    fig = plt.figure()
    plt.plot(h_array, energy_array, 'o-')
    plt.xlabel('h')
    plt.ylabel('energy')
    plt.title('Energy')
    plt.show()
    fig.savefig('MPSVariational_QuantumIsingModel_energy.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(h_array, magx_array, 'o-')
    plt.xlabel('h')
    plt.ylabel('magx')
    plt.title('Magnatization along x direction')
    plt.show()
    fig.savefig('MPSVariational_QuantumIsingModel_magx.pdf', bbox_inches='tight')
    plt.close(fig)

    np.save('MPSVariational_QuantumIsingModel_result.npy', result)


####################################################

if __name__ == '__main__':
    main()
