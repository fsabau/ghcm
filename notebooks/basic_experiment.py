import marimo

__generated_with = "0.9.23"
app = marimo.App(width="medium")


@app.cell
def __():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import SDEGenerator, SDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform, Normal
    from ghcm.test import GHCM
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt
    import networkx as nx
    import logging
    import jax.profiler

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return (
        DiracDelta,
        DiracDeltaDAG,
        GHCM,
        Normal,
        SDEGenerator,
        SDEParams,
        SigKernelRidgeRegression,
        Uniform,
        jax,
        jnp,
        logging,
        nx,
        plt,
    )


@app.cell
def __(DiracDelta, DiracDeltaDAG, Normal, SDEGenerator, Uniform):
    generator = SDEGenerator(
        adj = DiracDeltaDAG(3, [(2, 1), (2, 0), (2, 2)]), 
        x0 = Normal(0, 0.3, shape=(3,)),
        drift = Uniform(1, 2, shape=(3, 3)),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([0.3, 0.3, 0.3])
    )
    return (generator,)


@app.cell
def __(SDEParams, generator, jax, jnp):
    key = jax.random.key(127)
    ts = jnp.array(jnp.linspace(0.0, 1.0, 100))

    x, y, z = generator.generate_batch(key, ts, SDEParams(batch_size=100))
    return key, ts, x, y, z


@app.cell
def __(generator, key, nx, plt):
    X_color = '#1f77b4'
    Y_color = '#2ca02c'
    Z_color = '#ff7f0e'
    G = nx.from_numpy_array(generator.causal_graph(key).T, create_using=nx.DiGraph)
    nx.draw_networkx(G, labels = {0: 'X', 1: 'Y', 2: 'Z'}, node_color=[X_color, Y_color, Z_color])
    plt.gca()
    return G, X_color, Y_color, Z_color


@app.cell
def __(X_color, Y_color, Z_color, plt, ts, x, y, z):
    idx = 67
    plt.plot(ts, x[idx], color = X_color)
    plt.plot(ts, y[idx], color = Y_color)
    plt.plot(ts, z[idx], color = Z_color)
    return (idx,)


@app.cell
def __(GHCM, SigKernelRidgeRegression):
    ghcm = GHCM(SigKernelRidgeRegression)
    return (ghcm,)


@app.cell
def __(ghcm, key, x, y, z):
    p_value = ghcm.ci_test(x, y, z, key)
    return (p_value,)


@app.cell
def __(p_value):
    p_value
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
