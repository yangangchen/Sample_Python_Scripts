# odesolvers_test.py
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
# This test script calls "odesolvers.py".

################################

import odesolvers

import numpy as np
import matplotlib.pyplot as plt

##############################

T = 10
A = np.array([[-20, 10, 0, 0], [10, -20, 10, 0.], [0, 10, -20, 10], [0, 0, 10, -20]])
y0 = np.array([1, 1, 1, 1])
N = 400

ode = odesolvers.ODESolvers(A=A, T=T)
# y, ny = ode.ABsolver(y0=y0, N=N)
y, ny = ode.BDFsolver(y0=y0, N=N)
# plt.plot(np.linspace(0, T, N + 1), ny)
plt.semilogy(np.linspace(0, T, N + 1), ny)
plt.show()
