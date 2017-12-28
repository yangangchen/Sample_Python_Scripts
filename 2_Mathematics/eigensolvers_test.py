# eigensolvers_test.py
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
# This test script calls "eigensolvers.py".

################################

import eigensolvers

import numpy as np
from time import time

##############################

np.random.seed(100)
np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf, suppress=True)

print('\n +++++++ Test 1: Computational time +++++++ \n')

# Random example (non-symmetric)
n = 100
X = np.random.randn(n, n)
Lambda = np.diag(np.random.randn(n))
A = X.dot(Lambda.dot(np.linalg.inv(X)))
# Random example (symmetric)
# n = 100
# A = np.random.randn(n, n)
# A = 1 / 2 * (A + A.transpose())

# print('\n QR iteration with shift \n')
#
# t0 = time()
# Lambda, Q = eigensolvers.QR_iter_shift(A, sort=True)
# t1 = time()
# print(t1 - t0)

print('\n Ultimate eigen-solver \n')

t0 = time()
Lambda, Q = eigensolvers.eigen(A, sort=True)
t1 = time()
print(t1 - t0)

print('\n +++++++ Test 2: QR factorization (Gram-Schmidt, Householder) +++++++ \n')

# Example from my note
# m = 3
# n = 2
# A = np.array([[1, -4], [2, 3], [2, 2]]).astype(float)
# Random example
m = 10
n = 5
A = np.random.randn(m, n)

Q_exact, R_exact = np.linalg.qr(A)
print('\n Exact solution: \n')
print('\n Q: ')
print(Q_exact)
print('\n R: ')
print(R_exact)

Q, R = eigensolvers.qr_gram_schmidt(A)
print('\n Gram-Schmidt: \n')
print('\n Q: ')
print(Q)
print('\n R: ')
print(R)

QR = eigensolvers.qr_householder(A)
print('\n Householder: \n')
print('\n Q: ')
print(QR.Qmatmat(np.identity(m)))
print('\n Q^T: ')
print(QR.Qmatmat(np.identity(m), transpose=True))
print('\n R: ')
print(QR.R)

print('\n +++++++ Test 3: QR factorization (Givens rotation) +++++++ \n')

# Random example
n = 5
A = np.random.randn(n, n)
H = np.triu(A, -1)[:n, :n]
Q_exact, R_exact = np.linalg.qr(H)
print('\n Exact solution: \n')
print('\n Q: ')
print(Q_exact)
print('\n R: ')
print(R_exact)

QR = eigensolvers.qr_householder(H)
print('\n Householder: \n')
print('\n Q: ')
print(QR.Q())
print('\n R: ')
print(QR.R)

QR = eigensolvers.qr_givens_rotation(H)
print('\n Givens rotation: \n')
print('\n Q: ')
print(QR.Q())
print('\n R: ')
print(QR.R)

print('\n +++++++ Test 4: Single eigenvalue problem +++++++ \n')

# Example from my note (non-symmetric)
# n = 3
# A = np.array([[21, 7, -1], [5, 7, 7], [4, -4, 20]]).astype(float)
# Example from my note (symmetric)
# n = 3
# A = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 4]]).astype(float)
# Random example (non-symmetric)
n = 10
X = np.random.randn(n, n)
Lambda = np.diag(np.random.randn(n))
A = X.dot(Lambda.dot(np.linalg.inv(X)))
# Random example (symmetric)
# n = 10
# A = np.random.randn(n, n)
# A = 1 / 2 * (A + A.transpose())

l, q = eigensolvers.power_iter(A, q=np.ones(n))
l, q = eigensolvers.inverse_power_iter(A, mu=15, q=np.ones(n))
l, q = eigensolvers.rayleigh_quotient_iter(A, q=np.ones(n))

print('\n +++++++ Test 5: Full eigenvalue problem +++++++ \n')

print('\n Block power iteration \n')

Lambda, Q = eigensolvers.block_power_iter(A, Q=np.random.randn(n, n))
print('Eigenvalue: ')
print(Lambda)
print('Eigenvector: ')
print(Q)

print('\n QR iteration without shift \n')

Lambda, Q = eigensolvers.QR_iter(A)
print('Eigenvalue: ')
print(Lambda)
print('Eigenvector: ')
print(Q)

print('\n QR iteration with shift \n')

Lambda, Q = eigensolvers.QR_iter_shift(A, sort=False)
print('Eigenvalue: ')
print(Lambda)
print('Eigenvector: ')
print(Q)

Lambda, Q = eigensolvers.QR_iter_shift(A, sort=True)
print('Eigenvalue: ')
print(Lambda)
print('Eigenvector: ')
print(Q)

print('\n Ultimate eigen-solver \n')

Lambda, Q = eigensolvers.eigen(A, sort=False)
print('Eigenvalue: ')
print(Lambda)
print('Eigenvector: ')
print(Q)

Lambda, Q = eigensolvers.eigen(A, sort=True)
print('Eigenvalue: ')
print(Lambda)
print('Eigenvector: ')
print(Q)

print('\n +++++++ Test 6: Bidiagonalization +++++++ \n')

# Random example
m = 12
n = 8
A = np.random.randn(m, n)

SVD = eigensolvers.bidiagonalization(A)
U = SVD.Umatmat(np.identity(m))
V = SVD.Vmatmat(np.identity(n))
Anew = U.dot(SVD.A.dot(V))
print('A=')
print(A)
print('bidiag_A=')
print(SVD.A)
print('U=')
print(U)
print('V=')
print(V)
print('Error=')
print(A - Anew)

print('\n +++++++ Test 7: Singular value decomposition +++++++ \n')

U_exact, S_exact, V_exact = np.linalg.svd(A)
print('A=')
print(A)
print('S_exact=')
print(S_exact)
print('U_exact=')
print(U_exact)
print('V_exact=')
print(V_exact)

U, S, V = eigensolvers.svd(A)
Anew = U.dot(np.diag(S).dot(V.transpose()))
print('A=')
print(A)
print('S=')
print(S)
print('U=')
print(U)
print('V=')
print(V)
print('Error=')
print(A - Anew)
