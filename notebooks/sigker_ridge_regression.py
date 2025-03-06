import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import LinearSDEGenerator, LinearSDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform
    from sigkerax.sigkernel import SigKernel
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt
    return (
        DiracDelta,
        DiracDeltaDAG,
        LinearSDEGenerator,
        LinearSDEParams,
        SigKernel,
        SigKernelRidgeRegression,
        Uniform,
        jax,
        jnp,
        plt,
    )


@app.cell
def _(DiracDelta, DiracDeltaDAG, LinearSDEGenerator, Uniform):
    generator = LinearSDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (0, 1)]), 
        x0 = DiracDelta([0.0, -0.1, 0.0]),
        drift = Uniform(2, 3, shape=(3, 3)),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([0.3, 0.3, 0.3])
    )
    return (generator,)


@app.cell
def _(LinearSDEParams, generator, jax, jnp):
    key = jax.random.key(123)
    ts = jnp.array(jnp.linspace(0.0, 1.0, 200))

    x, y, z = generator.generate_batch(key, ts, LinearSDEParams(batch_size=100))
    return key, ts, x, y, z


@app.cell
def _(generator, key):
    print(generator.causal_graph(key))
    return


@app.cell
def _(plt, ts, x, y, z):
    plt.plot(ts, x[9])
    plt.plot(ts, y[9])
    plt.plot(ts, z[9])
    print(x.shape)
    return


@app.cell
def _():
    #sigker = SigKernel(refinement_factor=2, static_kernel_kind="rbf", add_time=True)
    #k = sigker.kernel_matrix(jnp.expand_dims(x, 2), jnp.expand_dims(x, 2))
    return


@app.cell
def _(SigKernelRidgeRegression, x, y):
    sigker_rr = SigKernelRidgeRegression(X_train=x, Y_train=y, reg_strength=1)
    return (sigker_rr,)


@app.cell
def _(sigker_rr, x):
    y_pred = sigker_rr.predict(x)
    return (y_pred,)


@app.cell
def _(plt, ts, x, y, y_pred):
    idx = 57
    plt.plot(ts, x[idx])
    plt.plot(ts, y_pred[idx])
    plt.plot(ts, y[idx])
    return (idx,)


@app.cell
def _(y, y_pred):
    res = y - y_pred
    res.shape
    return (res,)


@app.cell
def _(jnp, res, z):
    mse = jnp.mean(res * res, axis = 0)
    target_mse = jnp.mean(z * z, axis = 0)
    return mse, target_mse


@app.cell
def _(mse, plt, target_mse, ts):
    plt.plot(ts, mse)
    plt.plot(ts, target_mse)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
