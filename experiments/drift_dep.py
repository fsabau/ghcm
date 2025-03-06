import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM, conditionally_independent_sym
    from ghcm.experiment import ExperimentSDE, TestParams, TestType
    from ghcm.typing import X, Y, Z
    from ghcm.visualize import plot_sdes, plot_causal_dag
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
        TestParams,
        TestType,
        Uniform,
        X,
        Y,
        Z,
        conditionally_independent_sym,
        jax,
        jnp,
        jrn,
        plot_causal_dag,
        plot_sdes,
    )


@app.cell
def _():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax, fig, plt


@app.cell
def _(jax, jnp):
    def get_ci_test(name: str):
        if name == 'ghcm':
            from ghcm.regression import SigKernelRidgeRegression
            from ghcm.test import GHCM
            return GHCM(SigKernelRidgeRegression)
        elif name == 'sdcit':
            from sdcit.test import SDCIT
            from sigkerax.sigkernel import SigKernel
            sigker = SigKernel(refinement_factor=4, static_kernel_kind="rbf", add_time=True)
            return SDCIT(lambda x, y: jax.vmap(sigker.kernel_matrix, in_axes=0)(jnp.array(x), jnp.array(y)).squeeze(-1))
        else:
            raise Exception("Invalid CI test name")
    return (get_ci_test,)


@app.cell
def _(X, Y, Z, get_ci_test):
    import marimo as mo

    SEED = mo.cli_args().get("seed") or 135
    BATCH_SIZE = mo.cli_args().get("batch_size") or 32
    NUM_RUNS = mo.cli_args().get("num_runs") or 3
    STRUCTURE = mo.cli_args().get("structure") or 'chain'
    PERMUTATION = mo.cli_args().get("permutation") or 'XYZ'
    METHOD = mo.cli_args().get("method") or 'sdcit'

    edges = {
        'chain': [(X, Y), (Y, Z)],
        'fork': [(Y, X), (Y, Z)],
        'collider': [(X, Y), (Z, Y)],
    }[STRUCTURE]
    permutation = tuple(map(lambda c: ord(c) - ord('X'), PERMUTATION))
    ci_test = get_ci_test(METHOD)
    return (
        BATCH_SIZE,
        METHOD,
        NUM_RUNS,
        PERMUTATION,
        SEED,
        STRUCTURE,
        ci_test,
        edges,
        mo,
        permutation,
    )


@app.cell
def _(
    DiracDeltaDAG,
    LinearSDEGenerator,
    Mixture,
    Normal,
    Uniform,
    X,
    Y,
    Z,
    edges,
):
    generator = LinearSDEGenerator(
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
def _(SEED, generator, jnp, jrn):
    key = jrn.key(SEED)
    print(generator.drift.sample(key))
    ts = jnp.linspace(0, 1, 100)
    return key, ts


@app.cell
def _(
    conditionally_independent_sym,
    generator,
    key,
    permutation,
    plot_causal_dag,
):
    dag = generator.causal_graph(key)
    should_reject = not conditionally_independent_sym(dag, permutation)
    plot_causal_dag(dag)
    return dag, should_reject


@app.cell
def _(LinearSDEParams, generator, key, ts):
    x, y, z = generator.generate_batch(key, ts, LinearSDEParams(batch_size=32))
    return x, y, z


@app.cell
def _(plot_sdes, x, y, z):
    plot_sdes(x, y, z)
    return


@app.cell
def _(
    BATCH_SIZE,
    ExperimentSDE,
    LinearSDEParams,
    METHOD,
    NUM_RUNS,
    PERMUTATION,
    STRUCTURE,
    TestParams,
    TestType,
    ci_test,
    generator,
    permutation,
):
    experiment = ExperimentSDE(
        name=f"drift_dep_{METHOD}_bs{BATCH_SIZE}_runs{NUM_RUNS}_{STRUCTURE}_{PERMUTATION}",
        data_generator=generator,
        data_params=[
            LinearSDEParams(batch_size=BATCH_SIZE),
        ],
        test_params=TestParams(
            test_type=TestType.FUTURE_EXTENDED,
            permutation=permutation
        ),
        ci_test=ci_test,
        num_runs=NUM_RUNS,
    )
    return (experiment,)


@app.cell
def _(SEED, experiment):
    results, metadata = experiment.run_experiment(seed=SEED, reset_cache=True)
    return metadata, results


@app.cell
def _(METHOD, PERMUTATION, STRUCTURE, jnp, results, should_reject):
    p_values = jnp.array(results[0])
    mean = jnp.mean(p_values)
    std = jnp.std(p_values)

    print(f"CI test: {METHOD}")
    print(f"structure: {STRUCTURE}")
    print(f"permutation: {PERMUTATION}")
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
