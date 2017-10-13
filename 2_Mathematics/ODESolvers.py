# ODESolvers.py
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
# Solve the ordinary differential equation (ODE) y' = Ay using
# multistep ODE solvers. Here y is a vector, A is a matrix.

################################

import numpy as np
import matplotlib.pyplot as plt


class ODESolver:
    def __init__(self, A, T):
        self.A = A.copy()  # The matrix A
        self.T = T  # The total time T
        self.M = A.shape[0]  # The dimension of the problem

    def ABsolver(self, y0, N):
        # Use 2nd order Adams-Bashforth method to solve the ODE y' = Ay
        # y0: initial condition
        # N: number of the steps
        # y: solution at each time step
        # ny: norm of the solution at each time step
        h = self.T / N
        y = np.zeros((N + 1, self.M))
        normy = np.zeros(N + 1)

        y[0,] = y0
        normy[0] = np.linalg.norm(y[0,])

        for n in range(N):
            if n == 0:
                y1 = y0 + h * self.A.dot(y0)  # Forward Euler
                y[n + 1, :] = y1
            elif n == 1:
                y2 = y1 + h * (3 / 2 * self.A.dot(y1) - 1 / 2 * self.A.dot(y0))
                y[n + 1, :] = y2
            else:
                y0 = y1
                y1 = y2
                y2 = y1 + h * (3 / 2 * self.A.dot(y1) - 1 / 2 * self.A.dot(y0))
                y[n + 1, :] = y2
            normy[n] = np.linalg.norm(y[n + 1, :])

        return y, normy

    def BDFsolver(self, y0, N):
        # Use 2nd order Backward Differentiation Formulae method to solve the ODE y' = Ay
        # y0: initial condition
        # N: number of the steps
        # y: solution at each time step
        # ny: norm of the solution at each time step
        h = self.T / N
        y = np.zeros((N + 1, self.M))
        normy = np.zeros(N + 1)

        y[0, :] = y0
        normy[0] = np.linalg.norm(y[0,])

        for n in range(N):
            if n == 0:
                y1 = np.linalg.solve(np.eye(self.M) - h * self.A, y0)  # Backward Euler
                y[n + 1, :] = y1
            elif n == 1:
                y2 = np.linalg.solve(np.eye(self.M) - 2 / 3 * h * self.A, 4 / 3 * y1 - 1 / 3 * y0)
                y[n + 1, :] = y2
            else:
                y0 = y1
                y1 = y2
                y2 = np.linalg.solve(np.eye(self.M) - 2 / 3 * h * self.A, 4 / 3 * y1 - 1 / 3 * y0)
                y[n + 1, :] = y2
            normy[n] = np.linalg.norm(y[n + 1, :])

        return y, normy


################################

T = 10
A = np.array([[-20, 10, 0, 0], [10, -20, 10, 0.], [0, 10, -20, 10], [0, 0, 10, -20]])
y0 = np.array([1, 1, 1, 1])
N = 400

ode = ODESolver(A=A, T=T)
# y, ny = ode.ABsolver(y0=y0, N=N)
y, ny = ode.BDFsolver(y0=y0, N=N)
# plt.plot(np.linspace(0, T, N + 1), ny)
plt.semilogy(np.linspace(0, T, N + 1), ny)
plt.show()
