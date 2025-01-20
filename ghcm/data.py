import abc
import jax.numpy as jnp
import jax.random as jrn
from jax import Array
import jax
from jax.typing import ArrayLike
from ghcm.typing import Key, BinaryArray
from ghcm.distribution import Distribution, DAGDistribution
from typing import Tuple
from diffrax import VirtualBrownianTree, Euler, ODETerm, MultiTerm, ControlTerm, SaveAt, diffeqsolve
import equinox as eqx
from dataclasses import dataclass, field
from frozendict import frozendict

class SDE(eqx.Module):
    dim: int
    x0: Distribution
    drift: Array
    drift_bias: Array
    diffusion: Array
    diffusion_bias: Array

    def __call__(self, key: Key, ts: ArrayLike) -> Array:
        t0 = ts[0]
        t1 = ts[-1]        
        dt0 = 0.002
        drift = lambda t, x, args: self.drift @ x + self.drift_bias
        diffusion = lambda t, x, args: jnp.diag(self.diffusion @ x + self.diffusion_bias)
        key, bm_key = jrn.split(key)
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(self.dim,), key=bm_key)
        terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
        solver = Euler()
        saveat = SaveAt(ts=ts)

        x0 = self.x0.sample(key)
        sol = diffeqsolve(terms, solver, t0, t1, dt0, x0, saveat=saveat)
        return sol.ys
    
    def metadata(self) -> dict:
        return dict(
            data_name="SDE",
            drift=self.drift,
            drift_bias=self.drift_bias,
            diffusion=self.diffusion,
            diffusion_bias=self.diffusion_bias
        )


class StructuralCausalModel(abc.ABC):
    @abc.abstractmethod
    def generate_batch(self, key: Key, batch_size: int, **params) -> Tuple[Array, Array, Array]:
        pass

    @abc.abstractmethod
    def causal_graph(self, key: Key) -> BinaryArray:
        pass


@dataclass
class SDEParams:
    batch_size: int
    drift_strength: frozendict[tuple[int, int], float] = field(default_factory=frozendict)


class SDEGenerator(eqx.Module, StructuralCausalModel):
    adj: DAGDistribution
    x0: Distribution
    drift: Distribution
    drift_bias: Distribution
    diffusion: Distribution
    diffusion_bias: Distribution

    def generate_batch(self, key: Key, ts: Array, params: SDEParams) -> tuple[Array, Array, Array]:
        key, dag_key = jrn.split(key)
        dag = self.adj.sample_dag(dag_key)

        key, drift_key, drift_bias_key, diff_key, diff_bias_key, path_key = jrn.split(key, 6)
        drift = self.drift.sample(drift_key) * dag
        drift_bias = self.drift_bias.sample(drift_bias_key)
        diffusion = self.diffusion.sample(diff_key) * dag
        diffusion_bias = self.diffusion_bias.sample(diff_bias_key)

        ### Apply fixed params
        for (v, u), w in params.drift_strength.items():
            assert dag[u, v] != 0
            drift = drift.at[u, v].set(w)
        ### Done

        sde = SDE(3, self.x0, drift, drift_bias, diffusion, diffusion_bias)
        sde_batch = jax.vmap(sde, in_axes=(0, None))

        keys = jrn.split(path_key, params.batch_size)

        paths_batch = sde_batch(keys, ts)
        x = paths_batch[:, :, 0]
        y = paths_batch[:, :, 1]
        z = paths_batch[:, :, 2]
        return x, y, z
    
    def metadata(self, params: SDEParams) -> frozendict:
        return frozendict(
            adj=self.adj,
            x0=self.x0,
            drift=self.drift,
            drift_bias=self.drift_bias,
            diffusion=self.diffusion,
            diffusion_bias=self.diffusion_bias,
            drift_strength=params.drift_strength,
            batch_size=params.batch_size
        )

    def causal_graph(self, key: Key) -> BinaryArray:
        key, dag_key = jrn.split(key)
        dag = self.adj.sample_dag(dag_key)
        return dag 

