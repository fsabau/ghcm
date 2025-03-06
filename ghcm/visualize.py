from jax import Array
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import jax.numpy as jnp
from typing import Callable

X_color = '#1f77b4'
Y_color = '#2ca02c'
Z_color = '#ff7f0e'

def plot_sdes(x: Array, y: Array, z: Array, ts: Array | None = None, only_idx: int | None = None, alpha: float = 0.1, ax: Axes=None):
    if ax is None:
        fig, ax = plt.subplots()
    if ts is None:
        ts = jnp.linspace(0, 1, x.shape[1])
    x = x[:, :, 0]
    y = y[:, :, 0]
    z = z[:, :, 0]
    if only_idx is not None: 
        ax.plot(ts, x[only_idx], color = X_color)
        ax.plot(ts, y[only_idx], color = Y_color)
        ax.plot(ts, z[only_idx], color = Z_color)
    ax.plot(ts, x.T, color = X_color, alpha=alpha)
    ax.plot(ts, y.T, color = Y_color, alpha=alpha)
    ax.plot(ts, z.T, color = Z_color, alpha=alpha)
    return ax

def plot_causal_dag(adj: Array, ax: Axes = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.draw_networkx(G, pos={0: [-0.5, 0.33], 1: [0.0, -0.33], 2: [0.5, 0.33]}, labels = {0: 'X', 1: 'Y', 2: 'Z'}, node_color=[X_color, Y_color, Z_color], ax=ax)
    return ax

def plot_line_p_values(p_values: list[list[float]], metadata: list[dict], x_axis: Callable[[dict], float], ax: Axes=None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    x = jnp.array(list(map(x_axis, metadata)))
    y = jnp.array(p_values)
    y_mean = jnp.mean(y, axis=1)
    y_std = jnp.std(y, axis=1)

    ax.plot(x, y_mean)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    ax.set_ylim(0, 1)

    return ax
