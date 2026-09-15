import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import sys

    sys.path.insert(0, "src/Utils")
    sys.path.insert(0, "src/Utils/FEATHER/src")

    from types import SimpleNamespace

    import networkx as nx
    import numpy as np

    from feather import FEATHER

    return FEATHER, SimpleNamespace, np, nx


@app.cell
def _(FEATHER, SimpleNamespace, np, nx):
    G = nx.path_graph(4)

    features = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    args = SimpleNamespace(
        theta_max=1.0,
        eval_points=2,
        order=1,
    )

    model = FEATHER()
    model.fit(G, features, args)

    embedding = model.get_embedding()
    return (embedding,)


@app.cell
def _(embedding):
    embedding
    return


if __name__ == "__main__":
    app.run()
