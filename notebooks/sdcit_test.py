import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    from ghcm.data import SDEGenerator, SDEParams
    from ghcm.distribution import DiracDeltaDAG, DiracDelta, Mixture, Uniform
    from ghcm.test import conditionally_independent
    from ghcm.visualize import plot_causal_dag, plot_sdes
    import jax.random as jrn
    import jax.numpy as jnp
    import jax
    return (
        DiracDelta,
        DiracDeltaDAG,
        Mixture,
        SDEGenerator,
        SDEParams,
        Uniform,
        conditionally_independent,
        jax,
        jnp,
        jrn,
        plot_causal_dag,
        plot_sdes,
    )


@app.cell
def _(jax, jnp):
    from sdcit.test import SDCIT
    from sigkerax.sigkernel import SigKernel
    sigker = SigKernel(refinement_factor=4, static_kernel_kind="rbf", add_time=True)
    sdci = SDCIT(lambda x, y: jax.vmap(sigker.kernel_matrix, in_axes=0)(jnp.array(x), jnp.array(y)).squeeze(-1))
    return SDCIT, SigKernel, sdci, sigker


@app.cell
def _(DiracDelta, DiracDeltaDAG, Mixture, SDEGenerator, Uniform, jnp, jrn):
    generator = SDEGenerator(
        adj = DiracDeltaDAG(3, [(0, 0), (2, 1), (2, 0), (1, 1)]), 
        x0 = Uniform(-0.2, 0.2, shape=(3,)),
        drift = Mixture([Uniform(1, 2, shape=(3, 3)), Uniform(-2, -1, shape=(3,3))]),
        drift_bias = DiracDelta([0.0, 0.0, 0.0]),
        diffusion = DiracDelta(0, shape=(3,3)),
        diffusion_bias = DiracDelta([1, 1, 1])
    )
    key = jrn.key(123)
    ts = jnp.linspace(0, 1, 100)
    return generator, key, ts


@app.cell
def _(conditionally_independent, generator, key, plot_causal_dag):
    dag = generator.causal_graph(key)
    should_reject = not conditionally_independent(dag)
    plot_causal_dag(dag)
    return dag, should_reject


@app.cell
def _(SDEParams, generator, key, ts):
    x, y, z = generator.generate_batch(key, ts, SDEParams(batch_size=64))
    return x, y, z


@app.cell
def _(plot_sdes, x, y, z):
    plot_sdes(x, y, z)
    return


@app.cell
def _(key, sdci, x, y, z):
    p_value = sdci.ci_test(x, y, z, key)
    p_value
    return (p_value,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
