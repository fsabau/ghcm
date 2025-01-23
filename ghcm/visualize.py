from jax import Array
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Callable

X_color = '#1f77b4'
Y_color = '#2ca02c'
Z_color = '#ff7f0e'

def plot_sdes(x: Array, y: Array, z: Array, ts: Array | None = None, only_idx: int | None = None, alpha: float = 0.1):
    if ts is None:
        ts = jnp.linspace(0, 1, x.shape[1])
    if only_idx is not None: 
        plt.plot(ts, x[only_idx], color = X_color)
        plt.plot(ts, y[only_idx], color = Y_color)
        plt.plot(ts, z[only_idx], color = Z_color)
    plt.plot(ts, x.T, color = X_color, alpha=alpha)
    plt.plot(ts, y.T, color = Y_color, alpha=alpha)
    plt.plot(ts, z.T, color = Z_color, alpha=alpha)
    return plt.gca()

def plot_causal_dag(adj: Array) -> plt.Axes:
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.draw_networkx(G, labels = {0: 'X', 1: 'Y', 2: 'Z'}, node_color=[X_color, Y_color, Z_color])
    return plt.gca()

def plot_line_p_values(p_values: list[list[float]], metadata: list[dict], x_axis: Callable[[dict], float]) -> plt.Axes:
    x = jnp.array(list(map(x_axis, metadata)))
    y = jnp.array(p_values)
    y_mean = jnp.mean(y, axis=1)
    y_std = jnp.std(y, axis=1)

    plt.plot(x, y_mean)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    plt.ylim(0, 1)

    return plt.gca()
