from __future__ import annotations
import equinox as eqx
from typing import Callable
from jax import Array
import jax
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
import abc


class RegressionMethod(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def fit(cls, X: Array, Y: Array, **params) -> 'RegressionMethod':
        pass

    @abc.abstractmethod
    def predict(self, X: Array, **params) -> Array:
        pass

    @abc.abstractmethod
    def predict_params(self) -> dict:
        pass

#def vmapped_sigker_matrix_fn() -> Callable:
#    sigker = SigKernel(refinement_factor=4, static_kernel_kind="rbf", add_time=True)
#    return jax.vmap(sigker.kernel_matrix, in_axes=0)

class SigKernelRidgeRegression(eqx.Module, RegressionMethod):
    alphas: Array
    reg_strength: float
    K_train: Array
    X_train: Array
    signature_kernel: SigKernel
    Y_dim: int
    N: int

    def __init__(self, X_train: Array, Y_train: Array, reg_strength: float = 1):
        self.reg_strength = reg_strength
        self.X_train = X_train
        self.N = X_train.shape[0]
        self.Y_dim = Y_train.shape[-1]
        self.signature_kernel = SigKernel(refinement_factor=4, static_kernel_kind="rbf", add_time=True)

        K = self.compute_gram_matrix(X_train, X_train)
        self.K_train = K

        self.alphas = jnp.linalg.solve(K + reg_strength * jnp.eye(self.N), Y_train.reshape(self.N, -1))

    @classmethod
    def fit(cls, X: Array, Y: Array, **params) -> 'SigKernelRidgeRegression':
        return cls(X, Y, **params)
    
    def compute_gram_matrix(self, X: Array, Y: Array) -> Array:
        K = self.signature_kernel.kernel_matrix(X, Y)
        return K.squeeze()

    def predict(self, X: Array, K: Array | None = None) -> Array:
        if K is None:
            K = self.compute_gram_matrix(X, self.X_train)
        return (K @ self.alphas).reshape(self.N, -1, self.Y_dim)
    
    def predict_params(self) -> dict:
        return {'K': self.K_train}
