import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from ghcm.data import LinearSDE, LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, ErdosRenyiDAG, Normal, Uniform
    from ghcm.typing import X,Y,Z
    import networkx as nx
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt
    return (
        DiracDelta,
        DiracDeltaDAG,
        ErdosRenyiDAG,
        LinearSDE,
        LinearSDEGenerator,
        LinearSDEParams,
        Normal,
        Uniform,
        X,
        Y,
        Z,
        jax,
        jnp,
        mo,
        nx,
        plt,
    )


@app.cell
def _(jax):
    key = jax.random.key(1728)
    return (key,)


@app.cell
def _(ErdosRenyiDAG, key, nx):
    adj = ErdosRenyiDAG(3, 0.5, True).sample_dag(key)
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    return G, adj


@app.cell
def _(G, nx, plt):
    nx.draw_networkx(G)
    plt.gca()
    return


@app.cell
def _(DiracDelta, LinearSDE, jnp):
    sde = LinearSDE(
        3,
        DiracDelta([-0.5, 0, 0.5]),
        jnp.array([[0.5, 0, 0], [1, -0.5, -1], [0, 0, 1]]),
        jnp.array([0, 0, 0]),
        jnp.diag(jnp.array([0.5, 0.5, 0.5])),
        jnp.array([1, 1, 1])
    )
    return (sde,)


@app.cell
def _(jnp, key, sde):
    ts = jnp.array(jnp.linspace(0.0, 1.0, 100))
    paths = sde(key, ts)
    return paths, ts


@app.cell
def _(paths, plt, ts):
    plt.plot(ts, paths)
    return


@app.cell
def _(jax, key, sde):
    sde_batched = jax.vmap(sde, in_axes=(0, None))
    keys = jax.random.split(key, 100)
    return keys, sde_batched


@app.cell
def _(keys, sde_batched, ts):
    paths_batch = sde_batched(keys, ts)
    return (paths_batch,)


@app.cell
def _(paths_batch, plt, ts):
    plt.plot(ts, paths_batch[14])
    return


@app.cell
def _(DiracDelta, DiracDeltaDAG, LinearSDEGenerator, Uniform):
    generator = LinearSDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (0, 1), (0, 2), (1, 2)]), 
        x0 = DiracDelta([-0.5, 0, 0.5]),
        drift = Uniform(minval=1, maxval=2, shape=(3, 3)),
        drift_bias = Uniform(minval=[1, -2, 2], maxval=[2, -1, 3]),
        diffusion = DiracDelta(0, shape=(3, 3)),
        diffusion_bias = DiracDelta([1, 2, 3])
    )
    return (generator,)


@app.cell
def _(LinearSDEParams, generator, key, ts):
    x, y, z = generator.generate_batch(key, ts, params=LinearSDEParams(batch_size=100))
    dag = generator.causal_graph(key)
    dag
    return dag, x, y, z


@app.cell
def _(plt, ts, x, y, z):
    plt.plot(ts, x[7])
    plt.plot(ts, y[7])
    plt.plot(ts, z[7])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
