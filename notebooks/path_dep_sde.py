import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import PathDepSDEGenerator, PathDepSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM
    from ghcm.experiment import ExperimentSDE, TestType, TestParams
    from ghcm.visualize import plot_line_p_values, plot_sdes
    import jax.random as jrn
    import jax.numpy as jnp
    return (
        DiracDelta,
        DiracDeltaDAG,
        ExperimentSDE,
        GHCM,
        Mixture,
        Normal,
        PathDepSDEGenerator,
        PathDepSDEParams,
        SigKernelRidgeRegression,
        TestParams,
        TestType,
        Uniform,
        jnp,
        jrn,
        plot_line_p_values,
        plot_sdes,
    )


@app.cell
def _(
    DiracDelta,
    DiracDeltaDAG,
    GHCM,
    Mixture,
    PathDepSDEGenerator,
    SigKernelRidgeRegression,
    Uniform,
):
    generator = PathDepSDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (2, 1), (0, 1), (2, 2)]), 
        x0 = DiracDelta([0.5, 0.0, -2.0]),
        drift = Mixture([Uniform(1.0, 1.2, shape=(3, 3)), Uniform(1.0, 1.2, shape=(3,3))]),
        diffusion_bias = DiracDelta([0.1, 0.1, 0.1]),
    )
    ghcm = GHCM(SigKernelRidgeRegression)
    return generator, ghcm


@app.cell
def _(
    ExperimentSDE,
    PathDepSDEParams,
    TestParams,
    TestType,
    generator,
    ghcm,
):
    experiment = ExperimentSDE(
        name="path_dep_sde_batch",
        data_generator=generator,
        data_params=[
            PathDepSDEParams(batch_size=32, drop_prob=0.1),
        ],
        test_params=TestParams(
            test_type=TestType.SYM
        ),
        ci_test=ghcm,
        num_runs=2,
    )
    return (experiment,)


@app.cell
def _(PathDepSDEParams, generator, jnp, jrn):
    key = jrn.key(123)
    ts = jnp.linspace(0, 1, 100)

    x, y, z = generator.generate_batch(key, ts, PathDepSDEParams(batch_size=32, drop_prob=0.5))
    return key, ts, x, y, z


@app.cell
def _(plot_sdes, x, y, z):
    plot_sdes(x, y, z, only_idx=30)
    return


@app.cell
def _(experiment):
    p_values = experiment.run_experiment(seed=123, reset_cache=True)
    p_values
    return (p_values,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
