import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    from ghcm.regression import SigKernelRidgeRegression
    from ghcm.data import SDEGenerator
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Uniform
    from sigkerax.sigkernel import SigKernel
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt
    return (
        DiracDelta,
        DiracDeltaDAG,
        SDEGenerator,
        SigKernel,
        SigKernelRidgeRegression,
        Uniform,
        jax,
        jnp,
        plt,
    )


@app.cell
def __(DiracDelta, DiracDeltaDAG, SDEGenerator, Uniform):
    generator = SDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (0, 1)]), 
        x0 = DiracDelta([0.0, -0.1, 0.0]),
        drift = Uniform(2, 3, shape=(3, 3)),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([0.3, 0.3, 0.3])
    )
    return (generator,)


@app.cell
def __(generator, jax, jnp):
    key = jax.random.key(123)
    ts = jnp.array(jnp.linspace(0.0, 1.0, 200))

    x, y, z = generator.generate_batch(key, ts, 250)
    return key, ts, x, y, z


@app.cell
def __(generator, key):
    print(generator.causal_graph(key))
    return


@app.cell
def __(plt, ts, x, y, z):
    plt.plot(ts, x[9])
    plt.plot(ts, y[9])
    plt.plot(ts, z[9])
    return


@app.cell
def __():
    #sigker = SigKernel(refinement_factor=2, static_kernel_kind="rbf", add_time=True)
    #k = sigker.kernel_matrix(jnp.expand_dims(x, 2), jnp.expand_dims(x, 2))
    return


@app.cell
def __(SigKernelRidgeRegression, x, y):
    sigker_rr = SigKernelRidgeRegression(X_train=x, Y_train=y, reg_strength=1)
    return (sigker_rr,)


@app.cell
def __(sigker_rr, x):
    y_pred = sigker_rr.predict(x)
    return (y_pred,)


@app.cell
def __(plt, ts, x, y, y_pred):
    idx = 57
    plt.plot(ts, x[idx])
    plt.plot(ts, y_pred[idx])
    plt.plot(ts, y[idx])
    return (idx,)


@app.cell
def __(y, y_pred):
    res = y - y_pred
    res.shape
    return (res,)


@app.cell
def __(jnp, res, z):
    mse = jnp.mean(res * res, axis = 0)
    target_mse = jnp.mean(z * z, axis = 0)
    return mse, target_mse


@app.cell
def __(mse, plt, target_mse, ts):
    plt.plot(ts, mse)
    plt.plot(ts, target_mse)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
