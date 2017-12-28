# Finance_European_Option.py
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
# Compute the price of the European put option by solving the Black-Scholes equation
#     du/dt + 1 / 2 * sigma * S ** 2 * d^2u/dx^2 + r S du/dS - r u = 0

################################


import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

sigma = 0.4  # Volatility
r = 0.03  # Interest rate
T = 1  # Time to expiry
level = 4  # Higher level means the grid is finer. Lower level means the grid is coarser
N = 25 * (2 ** (level - 1))  # Number of timesteps
dt = T / N
K = 100  # Strike price


def Sgrid(level):
    # Construct S, which is the non-uniform grid of the underlying asset
    # Whenever the level is increased by 1, the number of the grid nodes is doubled
    S0 = np.hstack([np.arange(0, 0.4 * K, 0.1 * K), np.arange(0.45 * K, 0.8 * K, 0.05 * K),
                    np.arange(0.82 * K, 0.9 * K, 0.02 * K), np.arange(0.91 * K, 1.1 * K, 0.01 * K),
                    np.arange(1.12 * K, 1.2 * K, 0.02 * K), np.arange(1.25 * K, 1.6 * K, 0.05 * K),
                    np.arange(1.7 * K, 2 * K, 0.1 * K), np.array([2.2 * K, 2.4 * K, 2.8 * K,
                                                                  3.6 * K, 5 * K, 7.5 * K, 10 * K])])
    S = S0.copy()
    for l in range(1, level):
        S = np.vstack([S, np.hstack([1 / 2 * (S[:-1] + S[1:]), 0])]).transpose().ravel()[:-1]
    return S


def BlackScholesMatrix(S):
    # Constructing the Black - Scholes matrix
    J = len(S)
    A = sps.lil_matrix((J, J))
    A[0, 0] = r
    for j in range(1, J - 1):
        a = sigma ** 2 * S[j] ** 2 / ((S[j] - S[j - 1]) * (S[j + 1] - S[j - 1]))
        b = sigma ** 2 * S[j] ** 2 / ((S[j + 1] - S[j]) * (S[j + 1] - S[j - 1]))
        p = r * S[j] / (S[j + 1] - S[j - 1])
        pp = r * S[j] / (S[j + 1] - S[j])
        if a - p >= 0:  # Central differencing
            A[j, j] = r + a + b
            A[j, j - 1] = -(a - p)
            A[j, j + 1] = -(b + p)
        else:  # Upwinding
            A[j, j] = r + a + b + pp
            A[j, j - 1] = -a
            A[j, j + 1] = -(b + pp)

    return A


def timestepping(V0, A, scheme='implicit'):
    # Implement the implicit scheme or Crank-Nicolson scheme for the European option
    # Implicit scheme: ( I + dt * A ) V = V0
    # Crank-Nicolson scheme: ( I + dt * A /2) V = (I - dt * A /2) V0

    J = len(V0)
    if scheme == 'implicit':
        return spsl.spsolve(sps.eye(J) + dt * A, V0)
    else:
        return spsl.spsolve(sps.eye(J) + 1 / 2 * dt * A, V0 - 1 / 2 * dt * A.dot(V0))


############### Main program ###############

S = Sgrid(level)
V = np.maximum(K - S, 0)  # Payoff of the European put
A = BlackScholesMatrix(S)

# Rannacher timestepping scheme, second-order accurate
for n in range(N):
    scheme = 'implicit' if n < 2 else 'Crank-Nicolson'
    V = timestepping(V, A, scheme)

# Compute the greeks: delta and gamma
Delta = (V[2:] - V[:-2]) / (S[2:] - S[:-2])
Gamma = ((V[2:] - V[1:-1]) / (S[2:] - S[1:-1])
         - (V[1:-1] - V[:-2]) / (S[1:-1] - S[:-2])) / (
            (S[2:] - S[:-2]) / 2)

# Plot the option price and the greeks
index = (S[1:-1] >= 50) & (S[1:-1] <= 150)
plt.plot(S[1:-1][index], V[1:-1][index])
plt.title('Option price at t=0')
plt.show()
plt.plot(S[1:-1][index], Delta[index])
plt.title('Option delta at t=0')
plt.show()
plt.plot(S[1:-1][index], Gamma[index])
plt.title('Option gamma at t=0')
plt.show()
