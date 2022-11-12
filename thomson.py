# Copyright (c) 2022 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Author: Sylvain Gubian, PMP S.A.
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('QtAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.optimize import dual_annealing

def polar_to_cartesian(m):
    return np.array([
        np.sin(m[0]) * np.cos(m[1]),
        np.sin(m[0]) * np.sin(m[1]),
        np.cos(m[0])
    ])

def filter_indexes(x):
    return np.any(x[0] < x[1])


class Thomson(object):
    def __init__(self) -> None:
        self.nb_call = 0
        self.best_value = np.inf

    def objective(self, x):
        x_mat = np.array(x).reshape(len(x) // 2, 2, order='F')

        def rdist_fun(x):
            return np.array(
                1 / np.sqrt(np.sum(
                    (y_mat[x[0]] - y_mat[x[1]])** 2))
            )

        y_mat = np.apply_along_axis(polar_to_cartesian, axis=1, arr=x_mat)
        seq_vec = np.arange(0, x_mat.shape[0])
        indexes = np.array(np.meshgrid(seq_vec, seq_vec)).T.reshape(-1, 2)
        filter_vec = np.apply_along_axis(filter_indexes, axis=1, arr=indexes)
        indexes = indexes[filter_vec]
        rdist = np.apply_along_axis(rdist_fun, axis=1, arr=indexes)
        fvalue = np.sum(rdist)
        if fvalue < self.best_value:
            self.best_value = fvalue
            update_plot(x, fvalue, self.nb_call, self.best_value, better=True)
        else:
            update_plot(x, fvalue, self.nb_call, self.best_value, better=False)
        self.nb_call +=1 
        return fvalue


n_particles = 12
lw = np.array([0] * (n_particles * 2))
up = np.concatenate((
    np.repeat(np.pi, n_particles),
    np.repeat(2 * np.pi, n_particles)), axis=None)
bounds=list(zip(lw, up))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

def init_sphere():
    ax.plot_wireframe(x, y, z, color="grey", linewidth=0.2)
    ax.plot_surface(x, y, z, color="g", alpha=0.1)
    return fig,

def update_plot(x, f, nb_call, best_value, better):
    ax.view_init(elev=20, azim=nb_call % 360 )
    ax.set_title(f'Nb function call: {nb_call} Energy: {best_value:.6f}')
    if better:
        plt.cla()
        init_sphere()
        x_mat = np.array(x).reshape(len(x) // 2, 2, order='F')
        y_mat = np.apply_along_axis(polar_to_cartesian, axis=1, arr=x_mat)
        for i in range(n_particles):
            ax.scatter(y_mat[i, 0], y_mat[i, 1], y_mat[i, 2])
        seq_vec = np.arange(0, n_particles)
        indexes = np.array(np.meshgrid(seq_vec, seq_vec)).T.reshape(-1, 2)
        filter_vec = np.apply_along_axis(filter_indexes, axis=1, arr=indexes)
        indexes = indexes[filter_vec]
        for i in range(indexes.shape[0]):
            ax.plot(
                [
                    y_mat[indexes[i, 0], 0],
                    y_mat[indexes[i, 1], 0],
                ],
                [
                    y_mat[indexes[i, 0], 1],
                    y_mat[indexes[i, 1], 1],
                ],
                [
                    y_mat[indexes[i, 0], 2],
                    y_mat[indexes[i, 1], 2],
                ], linewidth=0.9,
            )

    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ion()
init_sphere()
ax.view_init(elev=20, azim=(90))
thomson = Thomson()
res = dual_annealing(thomson.objective, bounds=bounds)


