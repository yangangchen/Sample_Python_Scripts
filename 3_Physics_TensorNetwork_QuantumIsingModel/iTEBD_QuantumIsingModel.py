# iTEBD_QuantumIsingModel.py
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

"""Use iTEBD algorithm to compute the ground state energy and magnatization
of the 1D infinite Quantum Ising Model. See

http://uni10-tutorials.readthedocs.io/en/latest/lecture1.html

for a description. The code is only implemented by numpy and scipy library."""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

np.random.seed(0)

spin_dim = 2
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
si = np.identity(spin_dim)


def hamiltonian(h):
    H = np.tensordot(sz, sz, axes=0) + h * 1 / 2 * (
        np.tensordot(sx, si, axes=0) + np.tensordot(si, sx, axes=0))
    H = np.transpose(H, (0, 2, 1, 3))
    return H


def magnatization(pattern):
    if pattern == 'xx':
        M = 1 / 2 * (
            np.tensordot(sx, si, axes=0) + np.tensordot(si, sx, axes=0))
    elif pattern == 'xi':
        M = np.tensordot(sx, si, axes=0)
    elif pattern == 'ix':
        M = np.tensordot(si, sx, axes=0)
    elif pattern == 'zz':
        M = 1 / 2 * (
            np.tensordot(sz, si, axes=0) + np.tensordot(si, sz, axes=0))
    elif pattern == 'zi':
        M = np.tensordot(sz, si, axes=0)
    elif pattern == 'iz':
        M = np.tensordot(si, sz, axes=0)
    M = np.transpose(M, (0, 2, 1, 3))
    return M


def unitary(dt, H):
    return expm(-dt * H.reshape((spin_dim ** 2, spin_dim ** 2)))


class TensorNetwork:
    def __init__(self, bond_dim, spin_dim):
        self.bond_dim = bond_dim
        self.spin_dim = spin_dim

        self.GammaA = self._initialize_Gamma()
        self.GammaB = self._initialize_Gamma()
        self.LambdaA = self._initialize_Lambda()
        self.LambdaB = self._initialize_Lambda()

    def _initialize_Gamma(self):
        GammaA = np.random.randn(self.bond_dim, self.bond_dim, self.spin_dim)
        # GammaA, _ = np.linalg.qr(np.random.randn(self.bond_dim * self.spin_dim, self.bond_dim))
        # GammaA = GammaA.reshape((self.bond_dim, self.spin_dim, self.bond_dim))
        # GammaA = np.transpose(GammaA, (0, 2, 1))
        return GammaA

    def _initialize_Lambda(self):
        LambdaA = np.random.randn(self.bond_dim)
        LambdaA *= 1 / np.linalg.norm(LambdaA)
        return LambdaA

    def _contract_AB(self, GammaA, GammaB, LambdaA, LambdaB):
        A = np.tensordot(GammaA, np.diag(LambdaA), axes=(1, 0))
        A = np.transpose(A, (0, 2, 1))
        B = np.tensordot(GammaB, np.diag(LambdaB), axes=(1, 0))
        B = np.transpose(B, (0, 2, 1))
        AB = np.tensordot(A, B, axes=(1, 0))
        AB = np.transpose(AB, (0, 2, 1, 3))
        return AB

    def _left_right_eigs(self, AB):
        ABAB = np.tensordot(AB, np.conj(AB), axes=((2, 3), (2, 3)))
        ABAB = np.transpose(ABAB, (0, 2, 1, 3))
        ABAB_mat = ABAB.reshape((self.bond_dim ** 2, self.bond_dim ** 2))
        D1, U1 = np.linalg.eig(ABAB_mat)
        # Pick the eigenvalue that is equal to 1
        ind1 = (abs(np.imag(D1)) < 1e-8) & (np.real(D1) > 1 - 1e-4) & (np.real(D1) < 1 + 1e-4)
        flag1 = sum(ind1) == 1
        D2, U2 = np.linalg.eig(ABAB_mat.transpose())
        # Pick the eigenvalue that is equal to 1
        ind2 = (abs(np.imag(D2)) < 1e-8) & (np.real(D2) > 1 - 1e-4) & (np.real(D2) < 1 + 1e-4)
        flag2 = sum(ind2) == 1
        flag = flag1 * flag2
        if flag:
            Right = U1[:, ind1].ravel()
            Left = U2[:, ind2].ravel()
            norm = Left.dot(Right)
            if abs(np.imag(norm)) > 1e-8:  # The norm must be a real positive number
                flag = 0
            else:
                if np.real(norm) < 0:
                    Right = -Right
                    norm = -norm
            Left = Left.reshape((self.bond_dim, self.bond_dim))
            Right = Right.reshape((self.bond_dim, self.bond_dim))
        else:
            Right = None
            Left = None
            norm = None
        return Left, Right, norm, flag

    def evaluate_tensors(self):
        self.AB = self._contract_AB(self.GammaA, self.GammaB, self.LambdaA, self.LambdaB)
        self.AB_Left, self.AB_Right, self.norm, flag1 = self._left_right_eigs(self.AB)
        self.BA = self._contract_AB(self.GammaB, self.GammaA, self.LambdaB, self.LambdaA)
        self.BA_Left, self.BA_Right, self.norm, flag2 = self._left_right_eigs(self.BA)
        flag = flag1 * flag2
        return flag

    def _apply_unitary(self, AB, LambdaB, U):
        AB1 = np.tensordot(np.diag(LambdaB), AB, axes=(1, 0))
        AB1_mat = AB1.reshape((self.bond_dim ** 2, self.spin_dim ** 2))
        Theta = AB1_mat.dot(U)
        Theta = Theta.reshape((self.bond_dim, self.bond_dim, self.spin_dim, self.spin_dim))
        return Theta

    def _tensor_svd(self, Theta):
        Theta1 = np.transpose(Theta, (0, 2, 1, 3))
        Theta1_mat = Theta1.reshape((self.bond_dim * self.spin_dim, self.bond_dim * self.spin_dim))
        X, LambdaA, Y = np.linalg.svd(Theta1_mat)
        X = X[:, :self.bond_dim]
        X = X.reshape((self.bond_dim, self.spin_dim, self.bond_dim))
        X = np.transpose(X, (0, 2, 1))
        Y = Y[:self.bond_dim, :]
        Y = Y.reshape((self.bond_dim, self.bond_dim, self.spin_dim))
        LambdaA = LambdaA[:self.bond_dim]
        LambdaA *= 1 / np.linalg.norm(LambdaA)
        return X, Y, LambdaA

    def _renormalize_Gamma(self, X, Y, LambdaB):
        GammaA = np.tensordot(np.diag(1 / LambdaB), X, axes=(1, 0))
        GammaB = np.tensordot(Y, np.diag(1 / LambdaB), axes=(1, 0))
        GammaB = np.transpose(GammaB, (0, 2, 1))
        return GammaA, GammaB

    def _iTEBD_onestep(self, GammaA, GammaB, LambdaA, LambdaB, U):
        AB = self._contract_AB(GammaA, GammaB, LambdaA, LambdaB)
        Theta = self._apply_unitary(AB, LambdaB, U)
        X, Y, LambdaA = self._tensor_svd(Theta)
        GammaA, GammaB = self._renormalize_Gamma(X, Y, LambdaB)
        return GammaA, GammaB, LambdaA, LambdaB

    def iTEBD(self, U, N):
        for n in range(N + 1):
            self.GammaA, self.GammaB, self.LambdaA, self.LambdaB = \
                self._iTEBD_onestep(self.GammaA, self.GammaB, self.LambdaA, self.LambdaB, U)
            self.GammaB, self.GammaA, self.LambdaB, self.LambdaA = \
                self._iTEBD_onestep(self.GammaB, self.GammaA, self.LambdaB, self.LambdaA, U)

    def _operator_expectation(self, op, AB, Left, Right):
        Left = Left.ravel()
        Right = Right.ravel()
        Tensor = np.tensordot(AB, op, axes=((2, 3), (0, 1)))
        Tensor = np.tensordot(Tensor, AB, axes=((2, 3), (2, 3)))
        Tensor = np.transpose(Tensor, (0, 2, 1, 3))
        Matrix = Tensor.reshape((self.bond_dim ** 2, self.bond_dim ** 2))
        value = Left.dot(Matrix.dot(Right))
        value /= self.norm
        return value

    def operator_expectation(self, op, pattern='AB-BA'):
        if pattern == 'AB':
            value = self._operator_expectation(op, self.AB, self.AB_Left, self.AB_Right)
        elif pattern == 'BA':
            value = self._operator_expectation(op, self.BA, self.BA_Left, self.BA_Right)
        elif pattern == 'AB-BA':
            value1 = self._operator_expectation(op, self.AB, self.AB_Left, self.AB_Right)
            value2 = self._operator_expectation(op, self.BA, self.BA_Left, self.BA_Right)
            value = 1 / 2 * (value1 + value2)
        return value


def main():
    dt = 0.01
    N = 10000
    bond_dim = 20
    h_array = np.linspace(0, 2, 41)
    energy_array = np.zeros(len(h_array))
    magx_array = np.zeros(len(h_array))
    magz_array = np.zeros(len(h_array))

    Mx = magnatization(pattern='xx')
    Mz1 = magnatization(pattern='zi')
    Mz2 = magnatization(pattern='iz')

    for i in range(len(h_array)):
        print(i)
        h = h_array[i]
        H = hamiltonian(h)
        U = unitary(dt, H)

        while 1:
            TN = TensorNetwork(bond_dim=bond_dim, spin_dim=spin_dim)

            TN.iTEBD(U, N)

            flag = TN.evaluate_tensors()

            if flag:
                energy_array[i] = TN.operator_expectation(H)
                magx_array[i] = TN.operator_expectation(Mx)
                magz_array[i] = 1 / 2 * (abs(TN.operator_expectation(Mz1, pattern='AB'))
                                         + abs(TN.operator_expectation(Mz2, pattern='AB')))
                print('Succeed!')
                break
            else:
                print('Restart evaluation!')

    result = np.vstack([h_array, energy_array, magx_array, magz_array])

    fig = plt.figure()
    plt.plot(h_array, energy_array, 'o-')
    plt.xlabel('h')
    plt.ylabel('energy')
    plt.title('Energy')
    plt.show()
    fig.savefig('iTEBD_QuantumIsingModel_energy.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(h_array, magx_array, 'o-')
    plt.xlabel('h')
    plt.ylabel('magx')
    plt.title('Magnatization along x direction')
    plt.show()
    fig.savefig('iTEBD_QuantumIsingModel_magx.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(h_array, magz_array, 'o-')
    plt.xlabel('h')
    plt.ylabel('magz')
    plt.title('Magnatization along z direction')
    plt.show()
    fig.savefig('iTEBD_QuantumIsingModel_magz.pdf', bbox_inches='tight')
    plt.close(fig)

    np.save('iTEBD_QuantumIsingModel_result.npy', result)


####################################################

if __name__ == '__main__':
    main()
