from abc import ABC, abstractmethod
from typing import Sequence, Tuple
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
import jax.random as jrn
from ghcm.typing import Key, Shape, BinaryArray
import equinox as eqnx

class Distribution(ABC):
    @abstractmethod
    def sample(self, key: Key) -> Array:
        pass

class Uniform(eqnx.Module, Distribution): 
    minval: Array
    maxval: Array
    shape: Shape
    
    def __init__(self, minval: ArrayLike, maxval: ArrayLike, shape: Shape | None = None):
        minval = jnp.array(minval)
        maxval = jnp.array(maxval)
        if shape is None:
            assert minval.shape == maxval.shape
            shape = minval.shape
        self.shape = shape
        self.minval = minval
        self.maxval = maxval

    def sample(self, key: Key) -> Array:
        key, subkey = jrn.split(key)
        return jrn.uniform(subkey, self.shape, dtype=float, minval=self.minval, maxval=self.maxval)

class DiracDelta(eqnx.Module, Distribution):
    value: Array

    def __init__(self, value: ArrayLike, shape: Shape | None = None):
        if shape is None:
            self.value = jnp.array(value)
        else:
            self.value = jnp.ones(shape) * value

    def sample(self, key: Array) -> Array:
        return self.value

class Bernoulli(eqnx.Module, Distribution):
    prob: Array
    shape: Shape
    
    def __init__(self, prob: ArrayLike, shape: Shape | None = None):
        if shape is None:
            shape = prob.shape
        self.shape = shape
        self.prob = jnp.array(prob)
    
    def sample(self, key: Key) -> Array: 
        key, subkey = jrn.split(key)
        return jrn.bernoulli(subkey, self.prob, self.shape)

class Normal(eqnx.Module, Distribution):
    mean: Array
    std: Array
    shape: Shape

    def __init__(self, mean: ArrayLike, std: ArrayLike, shape: Shape | None = None):
        if shape is None:
            assert mean.shape == std.shape
            shape = mean.shape
        self.mean = jnp.array(mean)
        self.std = jnp.array(std)
        self.shape = shape
    
    def sample(self, key: Key) -> Array:
        key, subkey = jrn.split(key)
        return self.mean + self.std * jrn.normal(subkey, self.shape)

class Mixture(eqnx.Module, Distribution):
    mixture: Sequence[Distribution]
    shape: Shape

    def __init__(self, mixture: Sequence[Distribution], shape: Shape | None = None):
        if shape is None:
            shape = mixture[0].shape
            for dist in mixture:
                assert shape == dist.shape
        self.shape = shape
        self.mixture = mixture

    def sample(self, key: Key) -> Array:
        key, cat_key, distr_key = jrn.split(key, 3)

        idx = jrn.categorical(cat_key, jnp.ones(len(self.mixture)), shape=self.shape)

        sample = jnp.zeros(self.shape)        

        distr_keys = jrn.split(distr_key, len(self.mixture))
        for i, distr in enumerate(self.mixture):
            sample = jnp.where(idx == i, distr.sample(distr_keys[i]), sample)

        return sample

class DAGDistribution(ABC):
    @abstractmethod
    def sample_dag(self, key: Key) -> BinaryArray:
        pass

class ErdosRenyiDAG(eqnx.Module, DAGDistribution):
    num_nodes: int
    prob: float = 0.5
    self_loops: bool = True

    def sample_dag(self, key: Key) -> BinaryArray:
        key, subkey = jrn.split(key)
        full_adj = jrn.bernoulli(subkey, self.prob, (self.num_nodes, self.num_nodes))
        dag_adj = jnp.tril(full_adj, 0 if self.self_loops else -1)

        key, subkey = jrn.split(key)
        dag_adj = jrn.permutation(subkey, dag_adj, 0)
        dag_adj = jrn.permutation(subkey, dag_adj, 1)
        return dag_adj

class DiracDeltaDAG(eqnx.Module, DAGDistribution):
    num_nodes: int
    edge_list: Sequence[Tuple[int, int]]

    def sample_dag(self, key: Key) -> BinaryArray:
        adj = jnp.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        rows, cols = zip(*self.edge_list)
        return adj.at[rows, cols].set(True).T