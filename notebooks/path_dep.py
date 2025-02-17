import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import SDEGenerator, SDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM, conditionally_independent
    from ghcm.experiment import ExperimentLinearSDE
    from ghcm.typing import X, Y, Z
    from ghcm.visualize import plot_sdes, plot_causal_dag
    import jax.random as jrn
    import jax.numpy as jnp
    import jax
    return (
        DiracDelta,
        DiracDeltaDAG,
        ExperimentLinearSDE,
        GHCM,
        Mixture,
        Normal,
        SDEGenerator,
        SDEParams,
        SigKernelRidgeRegression,
        Uniform,
        X,
        Y,
        Z,
        conditionally_independent,
        jax,
        jnp,
        jrn,
        plot_causal_dag,
        plot_sdes,
    )


@app.cell
def _(X, Y, Z):
    import marimo as mo

    SEED = mo.cli_args().get("seed") or 123
    BATCH_SIZE = mo.cli_args().get("batch_size") or 64
    NUM_RUNS = mo.cli_args().get("num_runs") or 3
    EDGE = mo.cli_args().get("edge") or ['XY', 'YZ']

    edges = list(map(lambda e: {
        'XY': (X, Y),
        'XZ': (X, Z),
        'Yx': (Y, X),
        'YZ': (Y, Z),
        'ZX': (Z, X),
        'ZY': (Z, Y)
    }[e], EDGE))
    edges_str = "_".join(EDGE)
    return BATCH_SIZE, EDGE, NUM_RUNS, SEED, edges, edges_str, mo


@app.cell
def _(
    DiracDeltaDAG,
    Mixture,
    Normal,
    SDEGenerator,
    Uniform,
    X,
    Y,
    Z,
    edges,
):
    generator = SDEGenerator(
        adj = DiracDeltaDAG(3, [(X, X), (Y, Y), (Z, Z)] + edges), 
        x0 = Normal(0, 0.2, shape=(3,)),
        drift = Mixture([
            Uniform([
                [-0.5, -2, -2],
                [-2, -0.5, -2],
                [-2, -2, -0.5],
            ], [
                [0.5, -1, -1],
                [-1, 0.5, -1],
                [-1, -1, 0.5],
            ]),
            Uniform([
                [-0.5, 1, 1],
                [1, -0.5, 1],
                [1, 1, -0.5],
            ], [
                [0.5, -1, 2],
                [2, 0.5, 2],
                [2, 2, 0.5],
            ])
        ]),
        drift_bias = Uniform(-0.1, 0.1, shape=(3,)),
        diffusion = Uniform([
                [-0.5, 0, 0],
                [0, -0.5, 0],
                [0, 0, -0.5],
            ], [
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5],
            ]),
        diffusion_bias = Uniform(-0.2, 0.2, shape=(3,))
    )
    return (generator,)


@app.cell
def _(SEED, jnp, jrn):
    key = jrn.key(SEED)
    ts = jnp.linspace(0, 1, 100)
    return key, ts


@app.cell
def _(conditionally_independent, generator, key, plot_causal_dag):
    dag = generator.causal_graph(key)
    should_reject = not conditionally_independent(dag)
    plot_causal_dag(dag)
    return dag, should_reject


@app.cell
def _(SDEParams, generator, key, ts):
    x, y, z = generator.generate_batch(key, ts, SDEParams(batch_size=100))
    return x, y, z


@app.cell
def _(plot_sdes, x, y, z):
    plot_sdes(x, y, z)
    return


@app.cell
def _(jax, jnp):
    from sdcit.test import SDCIT
    from sigkerax.sigkernel import SigKernel
    sigker = SigKernel(refinement_factor=4, static_kernel_kind="rbf", add_time=True)
    sdci = SDCIT(lambda x, y: jax.vmap(sigker.kernel_matrix, in_axes=0)(jnp.array(x), jnp.array(y)).squeeze(-1))
    return SDCIT, SigKernel, sdci, sigker


@app.cell
def _(
    BATCH_SIZE,
    ExperimentLinearSDE,
    GHCM,
    NUM_RUNS,
    SDEParams,
    SigKernelRidgeRegression,
    edges_str,
    generator,
    sdci,
):
    ghcm = GHCM(SigKernelRidgeRegression)

    experiment = ExperimentLinearSDE(
        name=f"path_dep_{edges_str}",
        data_generator=generator,
        data_params=[
            SDEParams(batch_size=BATCH_SIZE),
        ],
        ci_test=sdci,
        num_runs=NUM_RUNS,
    )
    return experiment, ghcm


@app.cell
def _(SEED, experiment):
    results, metadata = experiment.run_experiment(seed=SEED, reset_cache=True)
    return metadata, results


@app.cell
def _(jnp, results, should_reject):
    p_values = jnp.array(results[0])
    mean = jnp.mean(p_values)
    std = jnp.std(p_values)

    print(f"p value: {mean} +- {std}")
    print(f"should reject: {should_reject}")
    if should_reject:
        error = jnp.mean(p_values > 0.05)
        print(f"type 1 error: {error}")
    else:
        error = jnp.mean(p_values < 0.05)
        print(f"type 2 error: {error}")
    return error, mean, p_values, std


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
