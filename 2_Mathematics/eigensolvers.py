# eigensolvers.py
# 
# Copyright (C) 2017  Yangang Chen
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# 
# 
# 
# Multiple QR factorization algorithms, eigensolvers and SVD solvers.
# 
################################

import math
import numpy as np


##############################

def qr_gram_schmidt(A):
    m, n = A.shape
    Q = A.copy()
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


class qr_householder:
    def __init__(self, A):
        self.m, self.n = A.shape
        self.V = np.zeros((self.m, self.n))
        R = A.copy()
        for k in range(self.n):
            # Compute householder reflector v
            x = R[k:, k]
            v = x.copy()
            v[0] += np.sign(x[0]) * np.linalg.norm(x)
            v = v / np.linalg.norm(v)
            self.V[k:, k] = v
            # Apply householder reflector on each column vector
            for j in range(k, self.n):
                R[k:, j] -= 2 * v * (v.dot(R[k:, j]))
        self.R = R[:self.n, :]

    def Qmatvec(self, x, transpose=False):
        if transpose:
            for k in range(self.n):
                v = self.V[k:, k]
                x[k:] -= 2 * v * (v.dot(x[k:]))
        else:
            for k in reversed(range(self.n)):
                v = self.V[k:, k]
                x[k:] -= 2 * v * (v.dot(x[k:]))
        return x

    def Qmatmat(self, X, transpose=False):
        if transpose:
            for k in range(self.n):
                v = self.V[k:, k]
                for j in range(X.shape[1]):
                    X[k:, j] -= 2 * v * (v.dot(X[k:, j]))
        else:
            for k in reversed(range(self.n)):
                v = self.V[k:, k]
                for j in range(X.shape[1]):
                    X[k:, j] -= 2 * v * (v.dot(X[k:, j]))
        return X

    def Q(self):
        return self.Qmatmat(np.identity(self.m))


class qr_givens_rotation:
    def __init__(self, H):
        self.R = H.copy()
        self.n = H.shape[0]
        self.csarray = np.zeros((self.n - 1, 2))
        for i in range(self.n - 1):
            R0 = self.R[i, i:].copy()  # Note: must be a copy assignment!
            R1 = self.R[i + 1, i:].copy()  # Note: must be a copy assignment!
            c = R0[0]
            s = -R1[0]
            nR = math.sqrt(c ** 2 + s ** 2)
            c = c / nR
            s = s / nR
            self.csarray[i, 0] = c
            self.csarray[i, 1] = s
            self.R[i, i:] = c * R0 - s * R1
            self.R[i + 1, i:] = s * R0 + c * R1

    def Q(self):
        Q = np.identity(self.n)
        for i in reversed(range(self.n - 1)):
            c = self.csarray[i, 0]
            s = self.csarray[i, 1]
            Q0 = Q[i, :].copy()
            Q1 = Q[i + 1, :].copy()
            Q[i, :] = c * Q0 + s * Q1
            Q[i + 1, :] = -s * Q0 + c * Q1
        # Q = np.identity(self.n)
        # for i in reversed(range(self.n - 1)):
        #     Qiter = np.identity(self.n)
        #     c = self.csarray[i, 0]
        #     s = self.csarray[i, 1]
        #     Qiter[i, i] = c
        #     Qiter[i + 1, i + 1] = c
        #     Qiter[i, i + 1] = s
        #     Qiter[i + 1, i] = -s
        #     Q = Qiter.dot(Q)
        return Q


##############################

def power_iter(A, q, maxiter=10000):
    for iter in range(maxiter + 1):
        q = q / np.linalg.norm(q)
        l = q.dot(A.dot(q))
        if np.linalg.norm(A.dot(q) - l * q) < 1e-6:
            break
        q = A.dot(q)
    print('\nPower iteration: ')
    print('Number of iterations: ' + str(iter))
    print('Eigenvalue: ' + str(l))
    print('Eigenvector: ' + str(q))
    return l, q


def inverse_power_iter(A, mu, q, maxiter=10000):
    n = A.shape[0]
    for iter in range(maxiter + 1):
        q = q / np.linalg.norm(q)
        l = q.dot(A.dot(q))
        if np.linalg.norm(A.dot(q) - l * q) < 1e-6:
            break
        q = np.linalg.solve(A - mu * np.identity(n), q)
    print('\nInverse power iteration: ')
    print('Number of iterations: ' + str(iter))
    print('Eigenvalue: ' + str(l))
    print('Eigenvector: ' + str(q))
    return l, q


def rayleigh_quotient_iter(A, q, maxiter=10000):
    n = A.shape[0]
    for iter in range(maxiter + 1):
        q = q / np.linalg.norm(q)
        l = q.dot(A.dot(q))
        if np.linalg.norm(A.dot(q) - l * q) < 1e-6:
            break
        q = np.linalg.solve(A - l * np.identity(n), q)
    print('\nRayleigh quotient iteration: ')
    print('Number of iterations: ' + str(iter))
    print('Eigenvalue: ' + str(l))
    print('Eigenvector: ' + str(q))
    return l, q


##############################

def block_power_iter(A, Q, maxiter=10000):
    Z = Q.copy()
    for iter in range(maxiter + 1):
        QR = qr_householder(Z)
        Q = QR.Q()
        Z = A.dot(Q)
        Lambda = Q.transpose().dot(Z)
        if np.linalg.norm(A.dot(Q) - Q.dot(np.triu(Lambda, 0))) < 1e-6:
            break
    print('Number of iterations: ' + str(iter))
    return Lambda, Q


def QR_iter(A, maxiter=10000):
    n = A.shape[0]
    Lambda = A.copy()
    Q = np.identity(n)
    for iter in range(maxiter + 1):
        QR = qr_householder(Lambda)
        Qiter = QR.Q()
        Lambda = QR.R.dot(Qiter)
        Q = Q.dot(Qiter)
        if np.linalg.norm(A.dot(Q) - Q.dot(np.triu(Lambda, 0))) < 1e-6:
            break
    print('Number of iterations: ' + str(iter))
    return Lambda, Q


def QR_iter_shift(A, maxiter=10000, sort=True):
    n = A.shape[0]
    Lambda = A.copy()
    Q = np.identity(n)
    k = n
    for iter in range(maxiter + 1):
        subLambda = Lambda[:k, :k]
        a0 = subLambda[-1, -1]
        a1 = subLambda[-1, -2]
        d = (a1 - a0) / 2
        sign = 1 if d >= 0 else -1
        mu = a0 - sign * a1 ** 2 / (abs(d) + math.sqrt(d ** 2 + a1 ** 2))

        subLambda = subLambda - mu * np.identity(k)
        subQR = qr_householder(subLambda)
        subQiter = subQR.Q()
        subLambda = subQR.R.dot(subQiter)
        l = abs(subLambda[-1, -1])
        subLambda = subLambda + mu * np.identity(k)
        Lambda[:k, :k] = subLambda
        Qiter = np.identity(n)
        Qiter[:k, :k] = subQiter
        Q = Q.dot(Qiter)
        if l < 1e-6:
            if k == 2:
                break
            else:
                k -= 1
    if sort:
        sortindex = abs(Lambda.diagonal()).argsort()[::-1]
        Lambda = Lambda[sortindex, :][:, sortindex]
        Q = Q[:, sortindex]
    print('Number of iterations: ' + str(iter))
    return Lambda, Q


##############################

class hessenberg_reduction:
    def __init__(self, A):
        self.n = A.shape[0]
        self.V = np.zeros((self.n, self.n - 1))
        self.H = A.copy()
        for k in range(self.n - 1):
            # Compute householder reflector v
            x = self.H[k + 1:, k]
            v = x.copy()
            v[0] += np.sign(x[0]) * np.linalg.norm(x)
            v = v / np.linalg.norm(v)
            self.V[k + 1:, k] = v
            # Apply householder reflector on each column vector
            for j in range(k, self.n):
                self.H[k + 1:, j] -= 2 * v * (v.dot(self.H[k + 1:, j]))
            # Apply householder reflector on each row vector
            for i in range(self.n):
                self.H[i, k + 1:] -= 2 * v * (self.H[i, k + 1:].dot(v))

    def Qmatvec(self, x, transpose=False):
        if transpose:
            for k in range(self.n - 1):
                v = self.V[k + 1:, k]
                x[k + 1:] -= 2 * v * (v.dot(x[k + 1:]))
        else:
            for k in reversed(range(self.n - 1)):
                v = self.V[k + 1:, k]
                x[k + 1:] -= 2 * v * (v.dot(x[k + 1:]))
        return x

    def Qmatmat(self, X, transpose=False):
        if transpose:
            for k in range(self.n - 1):
                v = self.V[k + 1:, k]
                for j in range(X.shape[1]):
                    X[k + 1:, j] -= 2 * v * (v.dot(X[k + 1:, j]))
        else:
            for k in reversed(range(self.n - 1)):
                v = self.V[k + 1:, k]
                for j in range(X.shape[1]):
                    X[k + 1:, j] -= 2 * v * (v.dot(X[k + 1:, j]))
        return X

    def Q(self):
        return self.Qmatmat(np.identity(self.n))


##############################

def eigen(A, maxiter=10000, sort=True):
    n = A.shape[0]
    Hessenberg = hessenberg_reduction(A)
    print('Hessenberg reduction completed!')
    Lambda = Hessenberg.H.copy()
    Q = Hessenberg.Q()
    k = n
    for iter in range(maxiter + 1):
        subLambda = Lambda[:k, :k]
        a0 = subLambda[-1, -1]
        a1 = subLambda[-1, -2]
        d = (a1 - a0) / 2
        sign = 1 if d >= 0 else -1
        mu = a0 - sign * a1 ** 2 / (abs(d) + math.sqrt(d ** 2 + a1 ** 2))

        subLambda = subLambda - mu * np.identity(k)
        subQR = qr_givens_rotation(subLambda)
        # subQR = qr_householder(subLambda)
        subQiter = subQR.Q()
        subLambda = subQR.R.dot(subQiter)
        l = abs(subLambda[-1, -1])
        subLambda = subLambda + mu * np.identity(k)
        Lambda[:k, :k] = subLambda
        Qiter = np.identity(n)
        Qiter[:k, :k] = subQiter
        Q = Q.dot(Qiter)
        if l < 1e-6:
            if k == 2:
                break
            else:
                k -= 1
    if sort:
        sortindex = abs(Lambda.diagonal()).argsort()[::-1]
        Lambda = Lambda[sortindex, :][:, sortindex]
        Q = Q[:, sortindex]
    print('Number of iterations: ' + str(iter))
    return Lambda, Q


##############################

class bidiagonalization:
    def __init__(self, A):
        self.m, self.n = A.shape
        self.U = np.zeros((self.m, self.n))
        self.V = np.zeros((self.n - 2, self.n))
        self.A = A.copy()
        for k in range(self.n):
            # Compute householder reflector u
            x = self.A[k:, k]
            u = x.copy()
            u[0] += np.sign(x[0]) * np.linalg.norm(x)
            u = u / np.linalg.norm(u)
            self.U[k:, k] = u
            # Apply householder reflector on each column vector
            for j in range(k, self.n):
                self.A[k:, j] -= 2 * u * (u.dot(self.A[k:, j]))

            if k < self.n - 2:
                # Compute householder reflector v
                x = self.A[k, k + 1:]
                v = x.copy()
                v[0] += np.sign(x[0]) * np.linalg.norm(x)
                v = v / np.linalg.norm(v)
                self.V[k, k + 1:] = v
                # Apply householder reflector on each row vector
                for i in range(k, self.m):
                    self.A[i, k + 1:] -= 2 * v * (self.A[i, k + 1:].dot(v))

    def Umatvec(self, x, transpose=False):
        if transpose:
            for k in range(self.n):
                u = self.U[k:, k]
                x[k:] -= 2 * u * (u.dot(x[k:]))
        else:
            for k in reversed(range(self.n)):
                u = self.U[k:, k]
                x[k:] -= 2 * u * (u.dot(x[k:]))
        return x

    def Umatmat(self, X, transpose=False):
        if transpose:
            for k in range(self.n):
                u = self.U[k:, k]
                for j in range(X.shape[1]):
                    X[k:, j] -= 2 * u * (u.dot(X[k:, j]))
        else:
            for k in reversed(range(self.n)):
                u = self.U[k:, k]
                for j in range(X.shape[1]):
                    X[k:, j] -= 2 * u * (u.dot(X[k:, j]))
        return X

    def U(self):
        return self.Umatmat(np.identity(self.m))

    def Vmatvec(self, x, transpose=False):
        if transpose:
            for k in range(self.n - 2):
                v = self.V[k, k + 1:]
                x[k + 1:] -= 2 * v * (x[k + 1:].dot(v))
        else:
            for k in reversed(range(self.n - 2)):
                v = self.V[k, k + 1:]
                x[k + 1:] -= 2 * v * (x[k + 1:].dot(v))
        return x

    def Vmatmat(self, X, transpose=False):
        if transpose:
            for k in range(self.n - 2):
                v = self.V[k, k + 1:]
                for i in range(X.shape[0]):
                    X[i, k + 1:] -= 2 * v * (X[i, k + 1:].dot(v))
        else:
            for k in reversed(range(self.n - 2)):
                v = self.V[k, k + 1:]
                for i in range(X.shape[0]):
                    X[i, k + 1:] -= 2 * v * (X[i, k + 1:].dot(v))
        return X

    def V(self):
        return self.Vmatmat(np.identity(self.n))

def svd(A, maxiter=100):
    m, n = A.shape
    SVD = bidiagonalization(A)
    U = SVD.Umatmat(np.identity(m))
    V = SVD.Vmatmat(np.identity(n)).transpose()

    H = np.zeros((2 * n, 2 * n))
    H[n:, :n] = SVD.A[:n, :]
    H[:n, n:] = SVD.A[:n, :].transpose()
    permindices = np.vstack([range(n), range(n, 2 * n)]).transpose().reshape(2 * n)
    # print(permindices)
    # print(permindices.argsort())
    H = H[permindices, :][:, permindices]
    # print(H)

    Lambda = H.copy()
    Q = np.identity(2 * n)
    k = 2 * n
    for iter in range(maxiter + 1):
        subLambda = Lambda[:k, :k]
        a0 = subLambda[-1, -1]
        a1 = subLambda[-1, -2]
        d = (a1 - a0) / 2
        sign = 1 if d >= 0 else -1
        mu = a0 - sign * a1 ** 2 / (abs(d) + math.sqrt(d ** 2 + a1 ** 2))

        subLambda = subLambda - mu * np.identity(k)
        subQR = qr_givens_rotation(subLambda)
        # subQR = qr_householder(subLambda)
        subQiter = subQR.Q()
        subLambda = subQR.R.dot(subQiter)
        l = abs(subLambda[-1, -1])
        subLambda = subLambda + mu * np.identity(k)
        Lambda[:k, :k] = subLambda
        Qiter = np.identity(2 * n)
        Qiter[:k, :k] = subQiter
        Q = Q.dot(Qiter)
        if l < 1e-6:
            if k == 2:
                break
            else:
                k -= 1
    sortindex = Lambda.diagonal().argsort()[::-1]
    sortindex[n:] = sortindex[2 * n - 1:n - 1:-1]
    Lambda = Lambda[sortindex, :][:, sortindex]
    Q = Q[:, sortindex]

    Q = Q[permindices.argsort(), :]
    # mysign = np.sign(Q[0, :n] * Q[0, n:])
    # Q[:, n:] = Q[:, n:].dot(np.diag(mysign))
    HU = np.zeros((m, n))
    HU[:n, :] = math.sqrt(2) * Q[n:, :n]
    HV = math.sqrt(2) * Q[:n, :n]

    S = np.diag(Lambda[:n, :n])
    U = U.dot(HU)
    V = V.dot(HV)
    # print(A)
    # print(U.dot(np.diag(S).dot(V.transpose())))

    print('Number of iterations: ' + str(iter))
    return U, S, V
