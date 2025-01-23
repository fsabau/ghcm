from ghcm.data import SDEGenerator, SDEParams
from ghcm.test import CITest
from ghcm.typing import Key
from pathlib import  Path
import jax.random
from jax import Array
import jax.numpy as jnp
from frozendict import frozendict
from typing import Callable
import pickle


class ExperimentLinearSDE:
    name: str

    data_generator: SDEGenerator
    data_params: list[SDEParams]

    ci_test: CITest
    num_runs: int

    cache_dir: Path

    def __init__(
            self, 
            name: str,
            data_generator: SDEGenerator, 
            data_params: list[SDEParams], 
            ci_test: CITest, 
            num_runs: int, 
            cache_dir: Path = Path('experiments/')
            ):
        self.name = name
        self.data_generator = data_generator
        self.data_params = data_params
        self.ci_test = ci_test
        self.num_runs = num_runs
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir

    def run_experiment(self, seed: int = 123, reset_cache: bool = False) -> tuple[list[list[float]], list[frozendict]]:
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

            test_keys = jax.random.split(test_key, self.num_runs)
            p_values = self.get_batched_ci_test()(x, y, z, test_keys)

            results.append(list(p_values))
            metadata.append(meta)

        with experiment_file.open('wb') as f:
            pickle.dump((results, metadata), f)

        return results, metadata

    def get_batched_ci_test(self) -> Callable:
        return jax.vmap(self.ci_test.ci_test, in_axes=0)

