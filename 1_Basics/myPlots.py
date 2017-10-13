# myPlots.py
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
# Produce the following plots with a single line:
# (1) 2-dimensional (x-y) plot
# (2) 2-dimensional (x-y) plot with legends
# (3) 3-dimensional (xy-z) plot

################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib import rc


def myPlot2D(X, Y, linestyle=None, linewidth=None, marker=None, markersize=None, color=None,
             title=None, xlabel=None, ylabel=None,
             xmin=None, xmax=None, ymin=None, ymax=None, figsize=(8, 6),
             set_num_xticks=None, set_num_yticks=None,
             title_font_size=24, axis_font_size=24, tick_font_size=18,
             filename=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.plot(X, Y,
             linestyle=linestyle, linewidth=linewidth,
             marker=marker, markersize=markersize, color=color)
    if title is not None:
        plt.title(title, fontsize=title_font_size)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=axis_font_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=axis_font_size)
    if xmin is not None and xmax is not None:
        plt.xlim((xmin, xmax))
    if ymin is not None and ymax is not None:
        plt.ylim((ymin, ymax))
    if set_num_xticks is not None:
        ax.xaxis.set_major_locator(LinearLocator(set_num_xticks))
    if set_num_yticks is not None:
        ax.yaxis.set_major_locator(LinearLocator(set_num_yticks))
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


def myPlot2DLegend(X, Y, linestylelist, markerlist, colorlist, legendlist,
                   linewidth=None, markersize=None,
                   title=None, xlabel=None, ylabel=None,
                   xmin=None, xmax=None, ymin=None, ymax=None, figsize=(8, 6),
                   set_num_xticks=None, set_num_yticks=None,
                   title_font_size=24, axis_font_size=24, tick_font_size=18,
                   legend_font_size=20, legend_loc=0,
                   filename=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    for n in range(Y.shape[0]):
        plt.plot(X, Y[n, :],
                 linestyle=linestylelist[n], linewidth=linewidth,
                 marker=markerlist[n], markersize=markersize,
                 color=colorlist[n], label=legendlist[n])
    if title is not None:
        plt.title(title, fontsize=title_font_size)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=axis_font_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=axis_font_size)
    if xmin is not None and xmax is not None:
        plt.xlim((xmin, xmax))
    if ymin is not None and ymax is not None:
        plt.ylim((ymin, ymax))
    if set_num_xticks is not None:
        ax.xaxis.set_major_locator(LinearLocator(set_num_xticks))
    if set_num_yticks is not None:
        ax.yaxis.set_major_locator(LinearLocator(set_num_yticks))
    plt.legend(loc=legend_loc, fontsize=legend_font_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


def myPlot3D(X, Y, Z, title=None, xlabel=None, ylabel=None, zlabel=None,
             xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
             show_colorbar=False, figsize=(8, 6),
             set_num_xticks=None, set_num_yticks=None, set_num_zticks=None,
             title_font_size=32, axis_font_size=32, tick_font_size=24,
             filename=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    if X.ndim == 2:
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.5)
        # ax.set_aspect(0.8)
    elif X.ndim == 1:
        surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
    else:
        print("Mistake: yangangFunctions.py, def myPlot3D. The input data must have dimension 1 or 2!")
    if title is not None:
        ax.set_title(title, fontsize=title_font_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_font_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=axis_font_size)
    if zlabel is not None:
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zlabel, fontsize=axis_font_size, rotation=0)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if zmin is not None and zmax is not None:
        ax.set_zlim(zmin, zmax)
    if show_colorbar:
        fig.colorbar(surf, shrink=0.7)
    if set_num_xticks is not None:
        ax.xaxis.set_major_locator(LinearLocator(set_num_xticks))
    if set_num_yticks is not None:
        ax.yaxis.set_major_locator(LinearLocator(set_num_yticks))
    if set_num_zticks is not None:
        ax.zaxis.set_major_locator(LinearLocator(set_num_zticks))
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
