import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM
    from ghcm.experiment import ExperimentSDE, TestParams, TestType, conditionally_independent
    from ghcm.typing import X, Y, Z
    from ghcm.visualize import plot_sdes, plot_causal_dag
    import jax.random as jrn
    import jax.numpy as jnp
    import jax.lax
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
        conditionally_independent,
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
            packed_sigker = lambda t: sigker.kernel_matrix(t[0], t[1])
            return SDCIT(lambda x, y: jax.lax.map(packed_sigker, (jnp.array(x), jnp.array(y)), batch_size=100).squeeze(-1))
        else:
            raise Exception("Invalid CI test name")
    return (get_ci_test,)


@app.cell
def _(TestType, X, Y, Z, get_ci_test):
    import marimo as mo

    SEED = mo.cli_args().get("seed") or 148
    BATCH_SIZE = mo.cli_args().get("batch_size") or 32
    NUM_RUNS = mo.cli_args().get("num_runs") or 3
    STRUCTURE = mo.cli_args().get("structure") or 'chain'
    PERMUTATION = mo.cli_args().get("permutation") or 'XYZ'
    TEST_TYPE = mo.cli_args().get("type") or 'sym'
    METHOD = mo.cli_args().get("method") or 'sdcit'

    edges = {
        'chain': [(X, Y), (Y, Z)],
        'fork': [(Y, X), (Y, Z)],
        'collider': [(X, Y), (Z, Y)],
    }[STRUCTURE]
    permutation = tuple(map(lambda c: ord(c) - ord('X'), PERMUTATION))
    ci_test = get_ci_test(METHOD)
    test_type = {
        'sym': TestType.SYM,
        'future': TestType.FUTURE_EXTENDED
    }[TEST_TYPE]
    return (
        BATCH_SIZE,
        METHOD,
        NUM_RUNS,
        PERMUTATION,
        SEED,
        STRUCTURE,
        TEST_TYPE,
        ci_test,
        edges,
        mo,
        permutation,
        test_type,
    )


@app.cell
def _(
    DiracDelta,
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
        x0 = Normal(0, 0.5, shape=(3,)),
        drift = DiracDelta(0, shape=(3, 3)),
        drift_bias = Mixture([
            Uniform(0.6, 1.2, shape=(3,)),
            Uniform(-1.2, -0.6, shape=(3,))
        ]),
        diffusion = Uniform([
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ], [
                [0, 1.5, 1.5],
                [1.5, 0, 1.5],
                [1.5, 1.5, 0],
            ]),
        diffusion_bias =  Uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]),
    )
    return (generator,)


@app.cell
def _(SEED, jnp, jrn):
    key = jrn.key(SEED)
    ts = jnp.linspace(0, 1, 100)
    return key, ts


@app.cell
def _(
    conditionally_independent,
    generator,
    key,
    permutation,
    plot_causal_dag,
    test_type,
):
    dag = generator.causal_graph(key)
    should_reject = not conditionally_independent(test_type, dag, permutation)
    plot_causal_dag(dag)
    return dag, should_reject


@app.cell
def _(LinearSDEParams, generator, key, ts):
    x, y, z = generator.generate_batch(key, ts, LinearSDEParams(batch_size=32))
    return x, y, z


@app.cell
def _(plot_sdes, x, y, z):
    plot_sdes(x, y, z, only_idx=12)
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
    TEST_TYPE,
    TestParams,
    ci_test,
    generator,
    permutation,
    test_type,
):
    experiment = ExperimentSDE(
        name=f"diffusion_dep_{METHOD}_{TEST_TYPE}_{STRUCTURE}_{PERMUTATION}_bs{BATCH_SIZE}_runs{NUM_RUNS}",
        data_generator=generator,
        data_params=[
            LinearSDEParams(batch_size=BATCH_SIZE),
        ],
        test_params=TestParams(
            test_type=test_type,
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
def _(
    METHOD,
    PERMUTATION,
    STRUCTURE,
    TEST_TYPE,
    jnp,
    results,
    should_reject,
):
    p_values = jnp.array(results[0])
    mean = jnp.mean(p_values)
    std = jnp.std(p_values)

    print(f"CI test: {METHOD}")
    print(f"type: {TEST_TYPE}")
    print(f"structure: {STRUCTURE}")
    print(f"null: {PERMUTATION[0]} ⊥⊥ {PERMUTATION[1]} | {PERMUTATION[2]}")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
