import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM
    from ghcm.experiment import ExperimentSDE, TestParams
    from ghcm.visualize import plot_line_p_values
    return (
        DiracDelta,
        DiracDeltaDAG,
        ExperimentSDE,
        GHCM,
        LinearSDEGenerator,
        LinearSDEParams,
        Mixture,
        Normal,
        SigKernelRidgeRegression,
        TestParams,
        Uniform,
        plot_line_p_values,
    )


@app.cell
def _(
    DiracDelta,
    DiracDeltaDAG,
    GHCM,
    LinearSDEGenerator,
    Mixture,
    SigKernelRidgeRegression,
    Uniform,
):
    generator = LinearSDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (1, 2), (0, 2), (1, 1)]), 
        x0 = Uniform(-0.2, 0.2, shape=(3,)),
        drift = Mixture([Uniform(1, 2, shape=(3, 3)), Uniform(-2, -1, shape=(3,3))]),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([1, 1, 1])
    )
    ghcm = GHCM(SigKernelRidgeRegression)
    return generator, ghcm


@app.cell
def _(ExperimentSDE, LinearSDEParams, TestParams, generator, ghcm):
    experiment = ExperimentSDE(
        name="linear_sde_batch",
        data_generator=generator,
        data_params=[
            LinearSDEParams(batch_size=16),
            LinearSDEParams(batch_size=32),
        ],
        test_params=TestParams(permutation=(0, 1, 2)),
        ci_test=ghcm,
        num_runs=2,
    )
    return (experiment,)


@app.cell
def _(experiment):
    results, metadata = experiment.run_experiment(seed=125, reset_cache=True)
    return metadata, results


@app.cell
def _(results):
    results
    return


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(metadata, plot_line_p_values, results):
    plot_line_p_values(results, metadata, x_axis=lambda meta: meta['batch_size'])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
