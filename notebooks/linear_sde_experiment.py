import marimo

__generated_with = "0.9.23"
app = marimo.App(width="medium")


@app.cell
def __():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import SDEGenerator, SDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM
    from ghcm.experiment import ExperimentLinearSDE
    from ghcm.visualize import plot_line_p_values
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
        plot_line_p_values,
    )


@app.cell
def __():
    return


@app.cell
def __(
    DiracDelta,
    DiracDeltaDAG,
    GHCM,
    Mixture,
    SDEGenerator,
    SigKernelRidgeRegression,
    Uniform,
):
    generator = SDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (2, 1), (2, 0), (1, 1)]), 
        x0 = Uniform(-0.2, 0.2, shape=(3,)),
        drift = Mixture([Uniform(1, 2, shape=(3, 3)), Uniform(-2, -1, shape=(3,3))]),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([1, 1, 1])
    )
    ghcm = GHCM(SigKernelRidgeRegression)
    return generator, ghcm


@app.cell
def __(ExperimentLinearSDE, SDEParams, generator, ghcm):
    experiment = ExperimentLinearSDE(
        name="linear_sde_batch",
        data_generator=generator,
        data_params=[
            SDEParams(batch_size=50),
        ],
        ci_test=ghcm,
        num_runs=2,
    )
    return (experiment,)


@app.cell
def __(experiment):
    results, metadata = experiment.run_experiment(seed=124, reset_cache=True)
    return metadata, results


@app.cell
def __(results):
    results
    return


@app.cell
def __(metadata):
    metadata
    return


@app.cell
def __(metadata, plot_p_values, results):
    plot_p_values(results, metadata, x_axis=lambda meta: meta['batch_size'])
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
