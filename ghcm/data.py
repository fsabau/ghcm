import jax.numpy as jnp
import jax.random as jrn
from jax import Array
from jax.typing import ArrayLike
from dataclasses import dataclass
from ghcm.distribution import Distribution
from ghcm.typing import Key, BinaryArray
from typing import Sequence, Tuple
import diffrax

def dag_erdos_renyi(key: Key, num_nodes: int, prob: float = 0.5, self_loops: bool = True) -> BinaryArray:
    key, subkey = jrn.split(key)
    full_adj = jrn.bernoulli(subkey, prob, (num_nodes, num_nodes))
    dag_adj = jnp.tril(full_adj, 0 if self_loops else -1)

    key, subkey = jrn.split(key)
    dag_adj = jrn.permutation(subkey, dag_adj, 0)
    dag_adj = jrn.permutation(subkey, dag_adj, 1)
    return dag_adj

def dag_from_edges(num_nodes: int, edge_list: Sequence[Tuple[int, int]]) -> BinaryArray:
    adj = jnp.zeros((num_nodes, num_nodes), dtype=bool)
    rows, cols = zip(*edge_list)
    return adj.at[rows, cols].set(True)

@dataclass
class SDEConfig:
    num: int
    x0: Distribution
    adjacency: Array
    drift: Distribution
    diffusion: Distribution


class SDESimulator:
    def __init__(self, config: SDEConfig):
        self._config = config


    def __call__(self, ts: ArrayLike, key: Key) -> Array:
        cfg = self._config
        x0 = cfg.x0.sample(key)


