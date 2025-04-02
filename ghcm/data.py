import abc
import jax.numpy as jnp
import jax.random as jrn
from jax import Array
import jax
from jax.typing import ArrayLike
from ghcm.typing import Key, BinaryArray
from ghcm.distribution import Distribution, DAGDistribution, DiracDelta, DiracDeltaDAG
from typing import Tuple, Callable
from diffrax import VirtualBrownianTree, Euler, ODETerm, MultiTerm, ControlTerm, SaveAt, diffeqsolve
import equinox as eqx
from dataclasses import dataclass, field
from frozendict import frozendict


class StructuralCausalModel(abc.ABC):
    @abc.abstractmethod
    def generate_batch(self, key: Key, batch_size: int, **params) -> Tuple[Array, Array, Array]:
        pass

    @abc.abstractmethod
    def causal_graph(self, key: Key) -> BinaryArray:
        pass

    @abc.abstractmethod
    def metadata(self, params) -> frozendict:
        pass

class SDE(abc.ABC):

    @staticmethod
    def sample_path(
            key: Key, 
            ts: Array,
            dim: int,
            x0: Distribution, 
            drift: Callable, 
            diffusion: Callable) -> Array:
        t0 = ts[0]
        t1 = ts[-1]        
        dt0 = 0.002

        key, bm_key = jrn.split(key)
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(dim,), key=bm_key)
        terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
        solver = Euler()
        saveat = SaveAt(ts=ts)

        x0_arr = x0.sample(key)
        sol = diffeqsolve(terms, solver, t0, t1, dt0, x0_arr, throw=False, saveat=saveat)
        return sol.ys

@dataclass
class SDEParams:
    batch_size: int

class LinearSDE(eqx.Module, SDE):
    dim: int
    x0: Distribution
    drift: Array
    drift_bias: Array
    diffusion: Array
    diffusion_bias: Array

    def __call__(self, key: Key, ts: ArrayLike) -> Array:
        drift = lambda t, x, args: self.drift @ x + self.drift_bias
        diffusion = lambda t, x, args: jnp.diag(self.diffusion @ x + self.diffusion_bias)

        return self.sample_path(key, ts, self.dim, self.x0, drift, diffusion)
    
    def metadata(self) -> dict:
        return dict(
            data_name="LinearSDE",
            drift=self.drift,
            drift_bias=self.drift_bias,
            diffusion=self.diffusion,
            diffusion_bias=self.diffusion_bias
        )

@dataclass
class LinearSDEParams(SDEParams):
    drift_strength: frozendict[tuple[int, int], float] = field(default_factory=frozendict)


class LinearSDEGenerator(eqx.Module, StructuralCausalModel):
    adj: DAGDistribution
    x0: Distribution
    drift: Distribution
    drift_bias: Distribution
    diffusion: Distribution
    diffusion_bias: Distribution

    def generate_batch(self, key: Key, ts: Array, params: LinearSDEParams) -> tuple[Array, Array, Array]:
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

        sde = LinearSDE(3, self.x0, drift, drift_bias, diffusion, diffusion_bias)
        sde_batch = jax.vmap(sde, in_axes=(0, None))

        keys = jrn.split(path_key, params.batch_size)

        paths_batch = sde_batch(keys, ts)
        x = paths_batch[:, :, 0:1]
        y = paths_batch[:, :, 1:2]
        z = paths_batch[:, :, 2:3]
        return x, y, z
    
    def metadata(self, params: LinearSDEParams) -> frozendict:
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
        return dag.T
    
@dataclass
class PathDepSDEParams(SDEParams):
    pass

class PathDepSDE(eqx.Module, SDE):
    dim: int = eqx.field(static=True)
    total_dim: int = eqx.field(static=True)
    x0: Distribution
    drift: Array
    drift_bias: Array
    diffusion: Array
    diffusion_bias: Array

    def __call__(self, key: Key, ts: ArrayLike) -> Array:
        edges = self.drift

        num_edges = self.total_dim - self.dim

        drift = jnp.zeros((self.total_dim, self.total_dim), dtype=float)

        indices = jnp.stack(jnp.nonzero(edges, size=9, fill_value=-1), axis=1)

        for i, edge in enumerate(indices):
            v = edge[0]
            u = edge[1]

            def true_branch(drift):
                edge_idx = i + self.dim
                drift = drift.at[edge_idx, u].set(1.0)
                drift = drift.at[v, edge_idx].set(self.drift[v, u])
                return drift

            drift = jax.lax.cond(jnp.logical_not(jnp.logical_or(v == -1, u == -1)), true_branch, lambda x: x, drift)

        drift_bias = jnp.pad(self.drift_bias, (0, num_edges))
        diffusion_bias = jnp.pad(self.diffusion_bias, (0, num_edges))

        key, x0_key = jrn.split(key)
        x0 = DiracDelta(jnp.pad(self.x0.sample(x0_key), (0, num_edges)))
        drift_fn = lambda t, x, args: drift @ x + drift_bias
        diffusion_fn = lambda t, x, args: jnp.diag(diffusion_bias)

        return self.sample_path(key, ts, int(self.total_dim), x0, drift_fn, diffusion_fn)

class PathDepSDEGenerator(eqx.Module, StructuralCausalModel):
    adj: DAGDistribution
    x0: Distribution
    drift: Distribution
    diffusion_bias: Distribution

    def generate_batch(self, key: Key, ts: Array, params: PathDepSDEParams) -> tuple[Array, Array, Array]:
        key, dag_key = jrn.split(key, 2)
        dag = self.adj.sample_dag(dag_key)

        key, drift_key, diff_bias_key, path_key = jrn.split(key, 4)
        drift = self.drift.sample(drift_key) * dag
        diffusion_bias = self.diffusion_bias.sample(diff_bias_key)

        num_extra = jnp.count_nonzero(dag)
        total = 3 + num_extra
        sde = PathDepSDE(3, total, self.x0, drift, jnp.zeros(3), jnp.zeros((3, 3)), diffusion_bias)
        sde_batch = eqx.filter_vmap(sde, in_axes=(0, None))

        keys = jrn.split(path_key, params.batch_size)

        paths_batch = sde_batch(keys, ts)
        x = paths_batch[:, :, 0:1]
        y = paths_batch[:, :, 1:2]
        z = paths_batch[:, :, 2:3]
        return x, y, z

    def causal_graph(self, key: Key) -> BinaryArray:
        return self.adj.sample_dag(key).T

    def metadata(self, params: PathDepSDEParams) -> frozendict:
        return frozendict(
            adj=self.adj,
            x0=self.x0,
            drift=self.drift,
            diffusion_bias=self.diffusion_bias,
            batch_size=params.batch_size
        )


class NonLinearSDE(eqx.Module, SDE):
    def __call__(self, key: Key, ts: ArrayLike) -> Array:
        pass