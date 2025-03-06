import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM, conditionally_independent
    from ghcm.experiment import ExperimentSDE
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
def _(jax):
    from pathlib import Path
    cache_dir = Path('results/')

    SEED = 123
    BATCH_SIZE = 64
    NUM_RUNS = 400
    EDGES = [
        ['XY', 'ZX'],
        ['XY', 'ZY'],
        ['XZ', 'YZ'],
        ['XZ', 'ZY'],
        ['YX', 'YZ'],
        ['ZX', 'ZY'],
    ]
    METHOD = ['ghcm', 'sdcit']
    EXPERIMENT = 'drift_dep'

    key = jax.random.key(SEED)
    return (
        BATCH_SIZE,
        EDGES,
        EXPERIMENT,
        METHOD,
        NUM_RUNS,
        Path,
        SEED,
        cache_dir,
        key,
    )


@app.cell
def _(BATCH_SIZE, EXPERIMENT, NUM_RUNS, SEED, Tuple, cache_dir, pickle):
    def get_filename(method: str, edges: list[str]) -> str:
        edges_str = '_'.join(sorted(edges))
        return f"{EXPERIMENT}_{method}_bs{BATCH_SIZE}_runs{NUM_RUNS}_{edges_str}_{SEED}.pkl"

    def load_file(name: str) -> Tuple[list[list[float]], list[dict]]:
        with (cache_dir / name).open('rb') as f:
            results, metadata = pickle.load(f)
            return results, metadata
    return get_filename, load_file


@app.cell
def _(
    EDGES,
    conditionally_independent,
    get_filename,
    key,
    load_file,
    np,
    plot_causal_dag,
    plt,
):
    figs, axes = plt.subplots(nrows=2, ncols=3)
    plt.subplots_adjust(hspace=1.3)

    for idx in range(6):
        r,c = idx//3, idx % 3
        edges = EDGES[idx]
        ghcm_result, ghcm_metadata = load_file(get_filename('ghcm', edges))
        ghcm_pvs = np.array(ghcm_result[0])
        ghcm_mean, ghcm_std = np.mean(ghcm_pvs),np.std(ghcm_pvs)

        sdci_result, sdci_metadata = load_file(get_filename('sdcit', edges))
        sdci_pvs = np.array(sdci_result[0])
        sdci_mean, sdci_std = np.mean(sdci_pvs),np.std(sdci_pvs)

        graph = ghcm_metadata[0]['adj'].sample_dag(key).T
        should_reject = not conditionally_independent(graph)

        if should_reject:
            ghcm_error = np.mean(ghcm_pvs > 0.05)
            sdci_error = np.mean(sdci_pvs > 0.05)
            error_str = "type 1 error"
        else:
            ghcm_error = np.mean(ghcm_pvs < 0.05)
            sdci_error = np.mean(sdci_pvs < 0.05)
            error_str = f"type 2 error"

        text = f"should reject: {should_reject}\nGHCM\np value: {ghcm_mean:.2f} +- {ghcm_std:.2f}\n{error_str}: {ghcm_error:.3f}\n\nSDCIT\np value: {sdci_mean:.2f} +- {sdci_std:.2f}\n{error_str}: {sdci_error:.3f}"
        axes[r,c].set_xlabel(text)

        plot_causal_dag(graph, ax=axes[r,c])

    plt.show()
    return (
        axes,
        c,
        edges,
        error_str,
        figs,
        ghcm_error,
        ghcm_mean,
        ghcm_metadata,
        ghcm_pvs,
        ghcm_result,
        ghcm_std,
        graph,
        idx,
        r,
        sdci_error,
        sdci_mean,
        sdci_metadata,
        sdci_pvs,
        sdci_result,
        sdci_std,
        should_reject,
        text,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
