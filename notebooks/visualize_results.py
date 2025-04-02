import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM
    from ghcm.experiment import ExperimentSDE, conditionally_independent, TestType
    from ghcm.typing import X, Y, Z
    from ghcm.visualize import plot_sdes, plot_causal_dag
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    import jax.random as jrn
    import jax.numpy as jnp
    import jax
    return (
        DiracDelta,
        DiracDeltaDAG,
        ExperimentSDE,
        GHCM,
        LinearSDEGenerator,
        LinearSDEParams,
        Mixture,
        Normal,
        TestType,
        Uniform,
        X,
        Y,
        Z,
        conditionally_independent,
        jax,
        jnp,
        jrn,
        np,
        pickle,
        plot_causal_dag,
        plot_sdes,
        plt,
    )


@app.cell
def _(TestType, jax):
    from pathlib import Path
    cache_dir = Path('results/')

    SEED = 60
    BATCH_SIZE = 64
    NUM_RUNS = 1000
    STRUCTURES = [
        'chain',
        'fork',
        'collider'
    ]
    PERMUTATIONS = [
        'XYZ',
        'XZY',
        'YXZ',
        'YZX',
        'ZXY',
        'ZYX'
    ]
    METHOD = ['ghcm', 'sdcit']
    TYPE = 'future'

    def str_to_test_type(type: str):
        return {
            'sym': TestType.SYM,
            'future': TestType.FUTURE_EXTENDED
        }[type]

    EXPERIMENT = 'drift_dep'

    def permutation_str_to_tuple(perm: str) -> tuple:
        return tuple(map(lambda c: ord(c) - ord('X'), perm))


    key = jax.random.key(SEED)
    return (
        BATCH_SIZE,
        EXPERIMENT,
        METHOD,
        NUM_RUNS,
        PERMUTATIONS,
        Path,
        SEED,
        STRUCTURES,
        TYPE,
        cache_dir,
        key,
        permutation_str_to_tuple,
        str_to_test_type,
    )


@app.cell
def _(
    BATCH_SIZE,
    EXPERIMENT,
    NUM_RUNS,
    SEED,
    TYPE,
    Tuple,
    cache_dir,
    pickle,
):
    def get_filename(method: str, structure: str, permutation: str) -> str:
        return f"{EXPERIMENT}_{method}_{TYPE}_{structure}_{permutation}_bs{BATCH_SIZE}_runs{NUM_RUNS}_{SEED}.pkl"

    def load_file(name: str) -> Tuple[list[list[float]], list[dict]]:
        with (cache_dir / name).open('rb') as f:
            results, metadata = pickle.load(f)
            return results, metadata
    return get_filename, load_file


@app.cell
def _(
    PERMUTATIONS,
    TYPE,
    conditionally_independent,
    get_filename,
    key,
    load_file,
    np,
    permutation_str_to_tuple,
    plot_causal_dag,
    plt,
    str_to_test_type,
):
    def plot_structure(structure):

        figs, axes = plt.subplots(nrows=2, ncols=3)
        plt.subplots_adjust(hspace=1.8)

        for p_idx, perm in enumerate(PERMUTATIONS):
            r, c = p_idx // 3, p_idx % 3
            ghcm_result, ghcm_metadata = load_file(get_filename('ghcm', structure, perm))
            ghcm_pvs = np.array(ghcm_result[0])
            ghcm_mean, ghcm_std = np.mean(ghcm_pvs),np.std(ghcm_pvs)

            sdci_result, sdci_metadata = load_file(get_filename('sdcit', structure, perm))
            sdci_pvs = np.array(sdci_result[0])
            sdci_mean, sdci_std = np.mean(sdci_pvs),np.std(sdci_pvs)


            graph = ghcm_metadata[0]['adj'].sample_dag(key).T
            should_reject = not conditionally_independent(str_to_test_type(TYPE), graph, permutation_str_to_tuple(perm))

            if should_reject:
                ghcm_error = np.mean(ghcm_pvs > 0.05)
                sdci_error = np.mean(sdci_pvs > 0.05)
                error_str = "type 1 error"
            else:
                ghcm_error = np.mean(ghcm_pvs < 0.05)
                sdci_error = np.mean(sdci_pvs < 0.05)
                error_str = f"type 2 error"

            text = f"null: {perm[0]} ⊥⊥ {perm[1]} | {perm[2]}\nshould reject: {should_reject}\n\nGHCM\np value: {ghcm_mean:.2f} +- {ghcm_std:.2f}\n{error_str}: {ghcm_error:.3f}\n\nSDCIT\np value: {sdci_mean:.2f} +- {sdci_std:.2f}\n{error_str}: {sdci_error:.3f}"
            axes[r,c].set_xlabel(text)

            plot_causal_dag(graph, ax=axes[r, c])
        plt.show()
    return (plot_structure,)


@app.cell
def _(plot_structure):
    plot_structure('chain')
    return


@app.cell
def _(plot_structure):
    plot_structure('fork')
    return


@app.cell
def _(plot_structure):
    plot_structure('collider')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
