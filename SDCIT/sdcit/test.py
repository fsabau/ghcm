import numpy as np
from typing import Any, Callable
from sdcit.sdcit_mod import c_SDCIT as sdci_test

import abc

Array = np.ndarray
Key = Any

class CITest(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> float:
        pass

    @classmethod
    @abc.abstractmethod
    def vmapped_ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> list[float]:
        pass

class SDCIT(CITest):
    def __init__(self, kernel_fn: Callable):
        self.kernel_fn = kernel_fn

    def ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> float:
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)
        Z = np.expand_dims(Z, axis=0)
        kx = np.asarray(self.kernel_fn(X, X).squeeze(axis=0)).astype('float64')
        ky = np.asarray(self.kernel_fn(Y, Y).squeeze(axis=0)).astype('float64')
        kz = np.asarray(self.kernel_fn(Z, Z).squeeze(axis=0)).astype('float64')
        ts, p_value = sdci_test(kx, ky, kz)
        return p_value
    

    def vmapped_ci_test(self, X: Array, Y: Array, Z: Array, key: Key) -> list[float]:
        Kx = np.array(self.kernel_fn(X, X))
        Ky = np.array(self.kernel_fn(Y, Y))
        Kz = np.array(self.kernel_fn(Z, Z))

        p_values = []
        n = X.shape[0]
        for i in range(n):
            test_statistic, p_value = sdci_test(Kx[i], Ky[i], Kz[i])
            p_values.append(p_value)
        return p_values