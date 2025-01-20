from __future__ import annotations
import equinox as eqx
from jax import Array
from typing import Self
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
import abc


class RegressionMethod(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def fit(cls, X: Array, Y: Array, **params) -> Self:
        pass

    @abc.abstractmethod
    def predict(self, X: Array, **params) -> Array:
        pass

    @abc.abstractmethod
    def predict_params(self) -> dict:
        pass

class SigKernelRidgeRegression(eqx.Module, RegressionMethod):
    alphas: Array
    reg_strength: float
    K_train: Array
    X_train: Array
    signature_kernel: SigKernel

    def __init__(self, X_train: Array, Y_train: Array, reg_strength: float = 1):
        self.reg_strength = reg_strength
        self.X_train = X_train
        self.signature_kernel = SigKernel(refinement_factor=4, static_kernel_kind="rbf", add_time=True)

        K = self.compute_gram_matrix(X_train, X_train)
        self.K_train = K

        self.alphas = jnp.linalg.solve(K + reg_strength * jnp.eye(K.shape[0]), Y_train)

    @classmethod
    def fit(cls, X: Array, Y: Array, **params) -> Self:
        return cls(X, Y, **params)
    
    def compute_gram_matrix(self, X: Array, Y: Array) -> Array:
        K = self.signature_kernel.kernel_matrix(jnp.expand_dims(X, 2), jnp.expand_dims(Y, 2))
        return K.squeeze()

    def predict(self, X: Array, K: Array | None = None) -> Array:
        if K is None:
            K = self.compute_gram_matrix(X, self.X_train)
        return K @ self.alphas
    
    def predict_params(self) -> dict:
        return {'K': self.K_train}
