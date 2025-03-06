import equinox as eqx
import logging
from jax import Array
from ghcm.regression import RegressionMethod
import jax.numpy as jnp
import jax
import abc
from ghcm.typing import Key, BinaryArray

class CITest(abc.ABC):
    @abc.abstractmethod
    def ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> float:
        pass


    @abc.abstractmethod
    def vmapped_ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> list[float]:
        pass

class GHCM(eqx.Module, CITest):
    regression: type[RegressionMethod]
    mc_integration_sample_size: int = 100000

    @eqx.filter_jit
    def ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> float:
        assert X.shape[0] == Y.shape[0] == Z.shape[0]
        n = X.shape[0] 
        # logging.info(f"Number of samples (n): {n}")

        fitted_on_X = self.regression.fit(Z, X)
        # logging.info("Fitted regression Z on X")
        X_pred = fitted_on_X.predict(X, **fitted_on_X.predict_params())
        # logging.info("Predicted X")

        fitted_on_Y = self.regression.fit(Z, Y)
        # logging.info("Fitted regression Z on Y")
        Y_pred = fitted_on_Y.predict(Y, **fitted_on_Y.predict_params())
        # logging.info("Predicted Y")

        res_X = (X - X_pred).reshape(n, -1)
        res_Y = (Y - Y_pred).reshape(n, -1)

        Gamma = (res_X @ res_X.T) * (res_Y @ res_Y.T)
        T = (1/n) * jnp.sum(Gamma)
        # logging.info(f"Test statistic: {T}")

        J = jnp.full((n,n), 1/n)
        A = 1/(n-1) * (Gamma - J @ Gamma - Gamma @ J + J @ Gamma @ J)

        eigvals = jnp.linalg.eigvals(A)
        # logging.info(f"Computed eigvals")
        
        p_value = self.monte_carlo_p_value(eigvals, T, key)
        return p_value
    
    def monte_carlo_p_value(self, eigen_values: Array, test_statistic: float, key: Key) -> float:
        # logging.info("Monte Carlo integration started")
        normal_sample = jax.random.normal(key, shape=(self.mc_integration_sample_size, eigen_values.shape[-1]))
        S = (normal_sample**2) * eigen_values
        return jnp.mean(jnp.sum(S, axis=1) > test_statistic)

    def vmapped_ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> list[float]:
        return list(jax.vmap(self.ci_test, in_axes=0)(X, Y, Z, key))

def conditionally_independent_sym(dag: BinaryArray, permutation: tuple[int, int, int] = (0, 1, 2)) -> bool:
    dag = dag[permutation, :][:, permutation]
    if dag[0, 1] or dag[1, 0]:
        return False
    if dag[0, 2] and dag[1, 2]:
        return False
    return True

def conditionally_independent_h_future_extended(structure: str, permutation: str) -> bool:
    pass