from ghcm.data import StructuralCausalModel,  SDEParams
from ghcm.test import CITest
from ghcm.typing import Key, BinaryArray
from pathlib import  Path
import jax.random
from jax import Array
import jax.numpy as jnp
from frozendict import frozendict
from typing import Callable
import pickle
from enum import Enum
from dataclasses import dataclass
from functools import partial

class TestType(Enum):
    SYM = 'sym'
    FUTURE_EXTENDED = 'future'

@dataclass
class TestParams:
    test_type: TestType = TestType.SYM
    future_pct: float = 0.5
    permutation: tuple[int, int, int] = (0, 1, 2)


def conditionally_independent_sym(dag: BinaryArray, permutation: tuple[int, int, int] = (0, 1, 2)) -> bool:
    dag = dag[permutation, :][:, permutation]
    if dag[0, 1] or dag[1, 0]:
        return False
    if dag[0, 2] and dag[1, 2]:
        return False
    return True

def conditionally_independent_h_future_extended(dag: BinaryArray, permutation: tuple[int, int, int] = (0, 1, 2)) -> bool:
    structure = None
    if dag[0, 1] and dag[1, 2]:
        structure = 'chain'
    if dag[1, 0] and dag[1, 2]:
        structure = 'fork'
    if dag[0, 1] and dag[2, 1]:
        structure = 'collider'

    if structure is None:
        raise ValueError("DAG doesnt have one of the three forms: chain, fork, collider")
    
    return (structure, permutation) not in [
        ('chain', (0, 1, 2)),
        ('chain', (1, 2, 0)),
        ('fork', (1, 0, 2)),
        ('fork', (1, 2, 0)),
        ('collider', (0, 1, 2)),
        ('collider', (0, 2, 1)),
        ('collider', (2, 0, 1)),
        ('collider', (2, 1, 0))
    ]

def conditionally_independent(
        test_type: TestType, 
        dag: BinaryArray, 
        permutation: tuple[int, int, int] = (0, 1, 2)
        ) -> bool:
    if test_type == TestType.SYM:
        return conditionally_independent_sym(dag, permutation)
    elif test_type == TestType.FUTURE_EXTENDED:
        return conditionally_independent_h_future_extended(dag, permutation)

class ExperimentSDE:
    name: str

    data_generator: StructuralCausalModel
    data_params: list[SDEParams]
    test_params: TestParams

    ci_test: CITest
    num_runs: int

    cache_dir: Path

    def __init__(
            self, 
            name: str,
            data_generator: StructuralCausalModel, 
            data_params: list[SDEParams], 
            test_params: TestParams,
            ci_test: CITest, 
            num_runs: int, 
            cache_dir: Path = Path('results/'),
            ):
        self.name = name
        self.data_generator = data_generator
        self.data_params = data_params
        self.test_params = test_params
        self.ci_test = ci_test
        self.num_runs = num_runs
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir
    
    def ci_tests_with_params(self, x: Array, y: Array, z: Array, keys: Key) -> list[list[float]]:
        vs = [x, y, z]
        idx = self.test_params.permutation
        x, y, z = vs[idx[0]], vs[idx[1]], vs[idx[2]]
        # print(x.shape, y.shape, z.shape)
        if self.test_params.test_type == TestType.SYM:
            return self.ci_test.vmapped_ci_test(x, y, z, keys)
        elif self.test_params.test_type == TestType.FUTURE_EXTENDED:
            ts = x.shape[2]
            idx = ts - int(self.test_params.future_pct * ts)
            x_past = x[:, :, :idx, :]
            y_future = y[:, :, idx:, :]
            y_past = y[:, :, :idx, :]            
            z_full = z

            
            y_past = jnp.pad(y_past, ((0, 0), (0, 0), (0, ts-idx), (0,0)), mode='edge')

            cond = jnp.concat([y_past, z_full], axis=3)

            return self.ci_test.vmapped_ci_test(x_past, y_future, cond, keys)
    
    @staticmethod
    @partial(jax.jit, static_argnames='size')
    def linearly_interpolate_nan_data_1d(time_series: Array, size: int) -> Array:
        time_series = time_series.squeeze()
        xp = jnp.sort(jnp.argwhere(~jnp.isnan(time_series), size=size).squeeze())
        fp = jnp.take(time_series, xp)
        return jnp.expand_dims(jnp.interp(jnp.arange(size), xp, fp), 1)

    @staticmethod
    def linearly_interpolate_nan_data(time_series: Array, axis: int = 2) -> Array:
        with_size = partial(ExperimentSDE.linearly_interpolate_nan_data_1d, size=time_series.shape[axis])
        map_over_experiments = lambda x: jax.lax.map(with_size, x)
        map_over_traj = lambda x: jax.lax.map(map_over_experiments, x)
        return map_over_traj(time_series)


    def run_experiment(
            self,
            seed: int = 123, 
            reset_cache: bool = False
            ) -> tuple[list[list[float]], list[frozendict]]:
        experiment_file = self.cache_dir / (self.name + '_' + str(seed) + '.pkl')

        if experiment_file.exists() and not reset_cache:
            with experiment_file.open('rb') as f:
                results, metadata = pickle.load(f)
                return results, metadata

        results = []
        metadata = []
        for i, params in enumerate(self.data_params):
            key = jax.random.key(seed + i)
            data_key, test_key = jax.random.split(key, 2)

            data_keys = jax.random.split(data_key, self.num_runs)
            ts = jnp.linspace(0.0, 1.0, 100)
            x, y, z = jax.vmap(
                self.data_generator.generate_batch, 
                in_axes=(0, None, None), 
                )(data_keys, ts, params)


            x = self.linearly_interpolate_nan_data(x)
            y = self.linearly_interpolate_nan_data(y)
            z = self.linearly_interpolate_nan_data(z)



            meta = self.data_generator.metadata(params)
            meta = meta | {'test_params': self.test_params}

            test_keys = jax.random.split(test_key, self.num_runs)

            p_values = self.ci_tests_with_params(x, y, z, test_keys)

            results.append(list(p_values))
            metadata.append(meta)

        with experiment_file.open('wb') as f:
            pickle.dump((results, metadata), f)

        return results, metadata

