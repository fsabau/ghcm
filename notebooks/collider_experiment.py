import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal, Mixture
    from ghcm.test import GHCM
    from ghcm.visualize import plot_causal_dag, plot_sdes
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt
    import networkx as nx
    import logging
    import jax.profiler
    import marimo as mo

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return (
        DiracDelta,
        DiracDeltaDAG,
        GHCM,
        LinearSDEGenerator,
        LinearSDEParams,
        Mixture,
        Normal,
        SigKernelRidgeRegression,
        Uniform,
        jax,
        jnp,
        logging,
        mo,
        nx,
        plot_causal_dag,
        plot_sdes,
        plt,
    )


@app.cell
def _(
    DiracDelta,
    DiracDeltaDAG,
    LinearSDEGenerator,
    Mixture,
    Normal,
    Uniform,
):
    generator = LinearSDEGenerator(
        adj = DiracDeltaDAG(3, [(1, 2), (0, 2), (0, 0), (1, 1)]), 
        x0 = Normal([1, -1, 0], [0.3, 0.3, 0.3], shape=(3,)),
        drift = Mixture([Uniform(0.5, 1.0, shape=(3, 3)), Uniform(-1.0, -0.5, shape=(3, 3))]),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([0.3, 0.3, 0.3])
    )
    return (generator,)


@app.cell
def _(LinearSDEParams, generator, jax, jnp):
    key = jax.random.key(131)
    ts = jnp.array(jnp.linspace(0.0, 1.0, 100))

    x, y, z = generator.generate_batch(key, ts, LinearSDEParams(batch_size=100))
    return key, ts, x, y, z


@app.cell
def _(generator, key, plot_causal_dag):
    plot_causal_dag(generator.causal_graph(key))
    return


@app.cell
def _(plot_sdes, x, y, z):
    plot_sdes(x, y, z)
    return


@app.cell
def _(GHCM, SigKernelRidgeRegression):
    ghcm = GHCM(SigKernelRidgeRegression)
    return (ghcm,)


@app.cell
def _(ghcm, key, x, y, z):
    p_value = ghcm.ci_test(x, y, z, key)
    return (p_value,)


@app.cell
def _(p_value):
    p_value
    return


@app.cell
def _(mo):
    mo.md(r"""p-value shoule be low""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
