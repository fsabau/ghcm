from abc import ABC, abstractmethod
from typing import Sequence
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.random as jrn
from ghcm.typing import Key, Shape

class Distribution(ABC):
    @abstractmethod
    def sample(self, key: Key, shape: Shape) -> tuple[Array, Key]:
        pass

class Uniform(Distribution): 
    def __init__(self, minval: ArrayLike = 0, maxval: ArrayLike = 1):
        self._minval = minval
        self._maxval = maxval

    def sample(self, key: Key, shape: Shape) -> Array:
        key, subkey = jrn.split(key)
        return jrn.uniform(subkey, shape, dtype=float, minval = self._minval, maxval=self._maxval)

class DiracDelta(Distribution):
    def __init__(self, value: ArrayLike):
        self._value = value
    
    def sample(self, key: Array) -> Array:
        return jnp.array(self._value)

class Bernoulli(Distribution):
    def __init__(self, prob: ArrayLike = 0.5):
        self._prob = prob
    
    def sample(self, key: Key, shape: Shape) -> Array: 
        key, subkey = jrn.split(key)
        return jrn.bernoulli(subkey, self._prob, shape)

class Normal(Distribution):
    def __init__(self, mean: ArrayLike = 0.0, std: ArrayLike = 1.0):
        self._mean = mean
        self._std = std
    
    def sample(self, key: Key, shape: Shape) -> Array:
        key, subkey = jrn.split(key)
        return self._mean + self._std * jrn.normal(subkey, shape)

class Mixture(Distribution):
    def __init__(self, mixture: Sequence[Distribution], weights: Array | None = None):
        self._mixture = mixture
        self._weights = weights
    
    def sample(self, key: Key, shape: Shape) -> Array:
        key, subkey = jrn.split(key)

        weights = jnp.ones(len(self._mixture)) if self._weights == None else self._weights
        idx = jrn.categorical(subkey, weights)

        distr = self._mixture[idx]
        return distr.sample(key, shape)
