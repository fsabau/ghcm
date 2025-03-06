from ghcm.data import StructuralCausalModel,  SDEParams
from ghcm.test import CITest
from ghcm.typing import Key
from pathlib import  Path
import jax.random
from jax import Array
import jax.numpy as jnp
from frozendict import frozendict
from typing import Callable
import pickle
from enum import Enum
from dataclasses import dataclass

class TestType(Enum):
    SYM = 'sym'
    FUTURE_EXTENDED = 'future'

@dataclass
class TestParams:
    test_type: TestType = TestType.SYM
    future_pct: float = 0.5
    permutation: tuple[int, int, int] = (0, 1, 2)

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

            print(y_past.shape, z_full.shape)
            cond = jnp.concat([y_past, z_full], axis=3)

            return self.ci_test.vmapped_ci_test(x_past, y_future, cond, keys)

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
            meta = self.data_generator.metadata(params)
            meta = meta | {'test_params': self.test_params}

            test_keys = jax.random.split(test_key, self.num_runs)

            p_values = self.ci_tests_with_params(x, y, z, test_keys)

            results.append(list(p_values))
            metadata.append(meta)

        with experiment_file.open('wb') as f:
            pickle.dump((results, metadata), f)

        return results, metadata

