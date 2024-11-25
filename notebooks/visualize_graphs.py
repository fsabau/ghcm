import marimo

__generated_with = "0.9.23"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from ghcm.data import dag_from_edges, dag_erdos_renyi
    import networkx as nx
    import jax
    return dag_erdos_renyi, dag_from_edges, jax, mo, nx


@app.cell
def __(jax):
    key = jax.random.key(1721)
    return (key,)


@app.cell
def __(dag_erdos_renyi, key, nx):
    # adj = dag_from_edges(8, [(0, 1), (1, 2)])

    adj = dag_erdos_renyi(key, 3, self_loops=True)
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    return G, adj


@app.cell
def __(G, nx):
    import matplotlib.pyplot as plt
    nx.draw_networkx(G)
    plt.gca()
    return (plt,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
