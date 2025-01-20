import marimo

__generated_with = "0.9.23"
app = marimo.App(width="medium")


@app.cell
def __():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import SDEGenerator, SDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal
    from ghcm.test import GHCM
    from ghcm.experiment import ExperimentLinearSDE
    return (
        DiracDelta,
        DiracDeltaDAG,
        ExperimentLinearSDE,
        GHCM,
        Normal,
        SDEGenerator,
        SDEParams,
        SigKernelRidgeRegression,
        Uniform,
    )


@app.cell
def __(
    DiracDelta,
    DiracDeltaDAG,
    GHCM,
    SDEGenerator,
    SigKernelRidgeRegression,
    Uniform,
):
    generator = SDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (1, 2), (0, 2), (1, 1)]), 
        x0 = Uniform(-1, 1, shape=(3,)),
        drift = Uniform(1, 2, shape=(3, 3)),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([1, 1, 1])
    )
    ghcm = GHCM(SigKernelRidgeRegression)
    return generator, ghcm


@app.cell
def __(ExperimentLinearSDE, SDEParams, generator, ghcm):
    experiment = ExperimentLinearSDE(
        name="linear_sde",
        data_generator=generator,
        data_params=[
            SDEParams(drift_strength={(1, 2): -2}, batch_size=50), 
            SDEParams(drift_strength={(1, 1): -2}, batch_size=50)
        ],
        ci_test=ghcm,
        num_runs=5,
    )
    return (experiment,)


@app.cell
def __(experiment):
    results, metadata = experiment.run_experiment()
    return metadata, results


@app.cell
def __(results):
    results
    return


@app.cell
def __(metadata):
    metadata
    return


if __name__ == "__main__":
    app.run()
