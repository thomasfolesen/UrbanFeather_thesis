# pixi run marimo run notebooks/marimo/urbanfeather_sandbox.py
# pixi run marimo edit notebooks/marimo/urbanfeather_sandbox.py

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]
    feather_src = project_root / "src" / "Utils" / "FEATHER" / "src"

    if str(feather_src) not in sys.path:
        sys.path.insert(0, str(feather_src))

    try:
        from feather import FEATHER
    except Exception as exc:
        raise RuntimeError(
            "Could not import src/Utils/FEATHER/src/feather.py. "
            "Place this file in notebooks/marimo inside UrbanFeather_thesis."
        ) from exc
    return FEATHER, SimpleNamespace, mo, np, nx, pd, plt, project_root


@app.cell
def _(mo, project_root):
    mo.md(f"""
    # UrbanFEATHER visual explainer

    This app explains the weighted UrbanFEATHER implementation used in this repository.

    It imports the real class from:

    ```text
    {project_root}/src/Utils/FEATHER/src/feather.py
    ```

    The app intentionally uses a tiny four-node street network instead of downloading a city.
    That keeps every number visible and makes the notebook work without network access.

    The pipeline we will inspect is:

    ```text
    street lengths
         |
         v
    inverse-distance edge strengths
         |
         v
    normalized weighted adjacency matrix A_tilde
         |
         +--------------------------+
         |                          |
         v                          v
    node feature matrix X      theta evaluation points
         |                          |
         +------------+-------------+
                      |
                      v
               cosine + sine
                      |
                      v
                     X0
                      |
                A_tilde @ X
                      |
                 repeated order times
                      |
                      v
           concatenate every order
                      |
                      v
             FEATHER node embedding
    ```

    **Key idea:** `X` says **WHAT** information is being propagated.
    `A_tilde` says **HOW** information moves through the street graph.

    **Tip:** the FEATHER and street-length sliders are frozen at the top of the app,
    so you can change them while looking at any visualization below.
    """)
    return


@app.cell
def _(mo):
    theta_max = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=2.0,
        label="theta_max",
        show_value=True,
    )

    eval_points = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=3,
        label="eval_points",
        show_value=True,
    )

    order = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=2,
        label="order",
        show_value=True,
    )

    length_01 = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=100,
        label="Street 0-1 length (m)",
        show_value=True,
    )

    length_12 = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=50,
        label="Street 1-2 length (m)",
        show_value=True,
    )

    length_23 = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=200,
        label="Street 2-3 length (m)",
        show_value=True,
    )

    selected_node = mo.ui.dropdown(
        options={"Node 0": 0, "Node 1": 1, "Node 2": 2, "Node 3": 3},
        value="Node 1",
        label="Node to inspect",
    )

    selected_feature = mo.ui.dropdown(
        options={"Food": 0, "School": 1, "Transit": 2},
        value="Food",
        label="Amenity feature to inspect",
    )

    show_numbers = mo.ui.checkbox(
        value=False,
        label="Show exact numeric tables",
    )

    show_original = mo.ui.checkbox(
        value=False,
        label="Compare with original unweighted FEATHER mixing",
    )

    compare_baseline = mo.ui.checkbox(
        value=True,
        label="Compare current settings with the default baseline",
    )

    # Keep the main experiment sliders fixed to the top of the browser.
    # This behaves like Excel's frozen rows: you can scroll through the
    # explanations and plots while still changing the FEATHER parameters.
    _frozen_parameter_bar = mo.vstack(
        [
            mo.md("### 🪶 UrbanFEATHER parameters — frozen while you scroll"),
            mo.hstack([theta_max, eval_points, order], widths="equal"),
            mo.hstack([length_01, length_12, length_23], widths="equal"),
        ],
        gap=0.5,
    ).style(
        {
            "position": "fixed",
            "top": "0",
            "left": "50%",
            "transform": "translateX(-50%)",
            "width": "min(1500px, calc(100% - 24px))",
            "z-index": "10000",
            "box-sizing": "border-box",
            "padding": "10px 18px 12px 18px",
            "background": "#171b1f",
            "color": "#f5f7fa",
            "border": "1px solid #3d454d",
            "border-top": "0",
            "border-radius": "0 0 12px 12px",
            "box-shadow": "0 5px 18px rgba(0, 0, 0, 0.32)",
        }
    )

    # A fixed element is removed from the normal document flow, so reserve
    # roughly the same amount of space at the top of the notebook.
    _frozen_bar_spacer = mo.Html(
        "<div style='height: 205px' aria-hidden='true'></div>"
    )

    mo.vstack(
        [
            _frozen_parameter_bar,
            _frozen_bar_spacer,
            mo.md(
                """
                ### View options

                The six experiment sliders stay at the top. These controls only
                change what the explanations below focus on.
                """
            ),
            mo.hstack([selected_node, selected_feature], widths="equal"),
            mo.hstack([show_numbers, show_original, compare_baseline], widths="equal"),
        ]
    )
    return (
        compare_baseline,
        eval_points,
        length_01,
        length_12,
        length_23,
        order,
        selected_feature,
        selected_node,
        show_numbers,
        show_original,
        theta_max,
    )


@app.cell
def _(
    SimpleNamespace,
    eval_points,
    length_01,
    length_12,
    length_23,
    np,
    nx,
    order,
    theta_max,
):
    feature_names = ["food", "school", "transit"]

    # Small illustrative binary amenity-presence matrix.
    # Rows are nodes and columns are amenity categories.
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    G = nx.path_graph(4)
    G[0][1]["weight"] = float(length_01.value)
    G[1][2]["weight"] = float(length_12.value)
    G[2][3]["weight"] = float(length_23.value)

    args = SimpleNamespace(
        theta_max=float(theta_max.value),
        eval_points=int(eval_points.value),
        order=int(order.value),
    )

    theta = np.linspace(0.01, args.theta_max, args.eval_points)

    n_nodes = G.number_of_nodes()
    n_features = features.shape[1]
    values_per_order = n_features * args.eval_points * 2
    values_per_node = values_per_order * args.order
    return (
        G,
        args,
        feature_names,
        features,
        n_features,
        theta,
        values_per_node,
        values_per_order,
    )


@app.cell
def _(
    args,
    feature_names,
    features,
    mo,
    pd,
    values_per_node,
    values_per_order,
):
    _feature_table = pd.DataFrame(features, columns=feature_names)
    _feature_table.index = [f"node {_i}" for _i in range(len(_feature_table))]

    mo.vstack(
        [
            mo.md(
                f"""
                ## 0. What the current parameter settings mean

                The feature matrix in this teaching example has **{len(feature_names)} binary amenity features**.
                A `1` means that category is present at that node in the toy example; a `0` means it is absent.

                With the current FEATHER settings:

                ```text
                {len(feature_names)} features
                x {args.eval_points} evaluation points
                x 2 parts (cosine + sine)
                = {values_per_order} values per propagation order

                {values_per_order} values per order
                x {args.order} orders
                = {values_per_node} final values per node
                ```

                `theta_max` changes the **range being sampled**, but does not change the number of columns.
                `eval_points` changes how many characteristic-function samples are taken.
                `order` changes both propagation depth and the number of blocks concatenated into the final embedding.
                """
            ),
            _feature_table,
        ]
    )
    return


@app.cell
def _(FEATHER, G, np, pd):
    _model_for_matrix = FEATHER()
    A_tilde = _model_for_matrix._create_A_tilde(G)
    A_tilde_dense = A_tilde.toarray()

    raw_edge_rows = []
    for _u, _v, _data in G.edges(data=True):
        _length = float(_data["weight"])
        _strength = 1.0 / _length
        raw_edge_rows.append(
            {
                "edge": f"{_u}-{_v}",
                "length_m": _length,
                "inverse_distance_strength": _strength,
            }
        )

    raw_edge_table = pd.DataFrame(raw_edge_rows)

    # Educational comparison with original unweighted FEATHER.
    _A_unweighted = np.zeros_like(A_tilde_dense, dtype=float)
    for _u, _v in G.edges():
        _A_unweighted[_u, _v] = 1.0
        _A_unweighted[_v, _u] = 1.0
    _degree = _A_unweighted.sum(axis=1)
    _degree[_degree == 0] = 1.0
    A_original_dense = _A_unweighted / _degree[:, None]
    return A_original_dense, A_tilde, A_tilde_dense, raw_edge_table


@app.cell
def _(G, feature_names, features, plt, selected_node):
    _selected = int(selected_node.value)
    _pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}

    _strength_by_edge = {
        tuple(sorted((_u, _v))): 1.0 / float(_data["weight"])
        for _u, _v, _data in G.edges(data=True)
    }
    _max_strength = max(_strength_by_edge.values())
    _widths = [
        5.0 + 10.0 * (_strength_by_edge[tuple(sorted((_u, _v)))] / _max_strength)
        for _u, _v in G.edges()
    ]

    _node_colors = []
    _node_sizes = []
    for _node in G.nodes():
        if _node == _selected:
            _node_colors.append("#f4b942")
            _node_sizes.append(2300)
        else:
            _node_colors.append("#65a9d8")
            _node_sizes.append(1700)

    _fig, _ax = plt.subplots(figsize=(12, 4.2))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")

    import networkx as _nx

    _nx.draw_networkx_edges(
        G,
        _pos,
        ax=_ax,
        width=_widths,
        edge_color="#111827",
        alpha=1.0,
    )
    _nx.draw_networkx_nodes(
        G,
        _pos,
        ax=_ax,
        node_color=_node_colors,
        node_size=_node_sizes,
        edgecolors="#111827",
        linewidths=2.5,
    )

    for _node in G.nodes():
        _present = [feature_names[_j] for _j, _v in enumerate(features[_node]) if _v > 0]
        _feature_text = ", ".join(_present) if _present else "none"
        _ax.text(
            _pos[_node][0],
            _pos[_node][1],
            f"node {_node}\n{_feature_text}",
            ha="center",
            va="center",
            fontsize=10,
            color="#0f172a",
            fontweight="bold" if _node == _selected else "normal",
        )

    for _u, _v, _data in G.edges(data=True):
        _x = (_pos[_u][0] + _pos[_v][0]) / 2.0
        _length = float(_data["weight"])
        _strength = 1.0 / _length
        _ax.text(
            _x,
            0.17,
            f"{_length:.0f} m\n1/L = {_strength:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#374151", "alpha": 1.0},
        )

    _ax.set_title(
        "Urban street graph: thicker dark edge = larger raw inverse-distance strength",
        color="#111827",
        fontsize=14,
        pad=16,
    )
    _ax.set_xlim(-0.45, 3.45)
    _ax.set_ylim(-0.55, 0.65)
    _ax.axis("off")
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(mo, raw_edge_table, show_numbers):
    _extra = raw_edge_table if bool(show_numbers.value) else mo.md(
        "Turn on **Show exact numeric tables** if you want the edge-strength table."
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 1. UrbanFEATHER edge weighting

                The weighted implementation interprets the edge attribute called `weight` as a street length and transforms it with:

                ```python
                strength = 1 / length
                ```

                Therefore a shorter street has a larger raw connection strength.

                This is only the first step. These raw strengths are normalized per source node next.
                """
            ),
            _extra,
        ]
    )
    return


@app.cell
def _(A_tilde_dense, plt):
    _fig, _ax = plt.subplots(figsize=(7.5, 6.2))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")
    _im = _ax.imshow(A_tilde_dense, cmap="Blues", vmin=0.0, vmax=1.0)

    _ax.set_xticks(range(A_tilde_dense.shape[1]))
    _ax.set_yticks(range(A_tilde_dense.shape[0]))
    _ax.set_xticklabels([f"take from {i}" for i in range(A_tilde_dense.shape[1])], rotation=30, ha="right", color="#111827")
    _ax.set_yticklabels([f"node {i}" for i in range(A_tilde_dense.shape[0])], color="#111827")
    _ax.set_xlabel("Source neighbour whose previous X row is used", color="#111827")
    _ax.set_ylabel("Node being updated", color="#111827")
    _ax.set_title("A_tilde: normalized UrbanFEATHER mixing shares", color="#111827", pad=14)

    for _i in range(A_tilde_dense.shape[0]):
        for _j in range(A_tilde_dense.shape[1]):
            _value = A_tilde_dense[_i, _j]
            _text_color = "white" if _value > 0.55 else "#111827"
            _ax.text(_j, _i, f"{_value:.2f}", ha="center", va="center", color=_text_color, fontsize=11)

    _cbar = _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.set_label("share", color="#111827")
    _cbar.ax.tick_params(colors="#111827")
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(A_tilde_dense, mo, np, pd, selected_node):
    _node = int(selected_node.value)
    _row = A_tilde_dense[_node]
    _row_sum = float(_row.sum())

    _rows = []
    for _source, _share in enumerate(_row):
        if _share > 0:
            _rows.append({"source neighbour": _source, "normalized share": _share})
    _table = pd.DataFrame(_rows)

    mo.vstack(
        [
            mo.md(
                f"""
                ### Read one row of `A_tilde`

                You selected **node {_node}**.

                Row {_node} means:

                > When node {_node} is updated, what fraction should it take from each neighbour's previous FEATHER values?

                Current row sum: `{_row_sum:.6f}`

                ```text
                {np.round(_row, 4)}
                ```

                For connected nodes, the row sums to approximately 1. This gives FEATHER a weighted average of neighbouring information.
                """
            ),
            _table,
        ]
    )
    return


@app.cell
def _(
    A_original_dense,
    A_tilde_dense,
    mo,
    np,
    plt,
    selected_node,
    show_original,
):
    if not bool(show_original.value):
        mo.md(
            "Enable **Compare with original unweighted FEATHER mixing** to see why the UrbanFEATHER weighting matters."
        )
    else:
        _node = int(selected_node.value)
        _x = np.arange(A_tilde_dense.shape[1])
        _width = 0.36

        _fig, _ax = plt.subplots(figsize=(9, 4.6))
        _fig.patch.set_facecolor("white")
        _ax.set_facecolor("#f8fafc")
        _ax.bar(_x - _width / 2, A_original_dense[_node], width=_width, label="Original FEATHER: equal neighbours")
        _ax.bar(_x + _width / 2, A_tilde_dense[_node], width=_width, label="UrbanFEATHER: inverse-distance")
        _ax.set_xticks(_x)
        _ax.set_xticklabels([f"source {i}" for i in _x], color="#111827")
        _ax.set_ylabel("mixing share", color="#111827")
        _ax.set_ylim(0, 1.05)
        _ax.set_title(f"How node {_node} mixes neighbours: original vs UrbanFEATHER", color="#111827")
        _ax.tick_params(colors="#111827")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.legend()
        _fig.tight_layout()

        mo.vstack(
            [
                mo.md(
                    """
                    ### What changed from original FEATHER?

                    Original FEATHER gives connected neighbours equal probability when the graph is unweighted.
                    UrbanFEATHER instead derives the transition shares from the street lengths.

                    The graph structure is still important: only connected nodes can contribute.
                    The new part is **how much** each connected neighbour contributes.
                    """
                ),
                _fig,
            ]
        )
    return


@app.cell
def _(args, np, plt, theta):
    _fig, _ax = plt.subplots(figsize=(10.5, 2.4))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")

    _ax.hlines(0, 0.0, max(0.1, args.theta_max), color="#374151", linewidth=2)
    _ax.scatter(theta, np.zeros_like(theta), s=100, color="#2563eb", edgecolors="#111827", linewidths=1.2, zorder=3)

    for _i, _value in enumerate(theta):
        _ax.text(_value, 0.10 if _i % 2 == 0 else -0.14, f"{_value:.3f}", ha="center", va="center", color="#111827", fontsize=9)

    _ax.set_xlim(-0.05 * max(1.0, args.theta_max), args.theta_max * 1.05)
    _ax.set_ylim(-0.35, 0.35)
    _ax.set_yticks([])
    _ax.set_xlabel("theta", color="#111827")
    _ax.tick_params(axis="x", colors="#111827")
    _ax.set_title(
        f"theta_max = {args.theta_max:g}, eval_points = {args.eval_points}: FEATHER samples these positions",
        color="#111827",
    )
    _ax.spines[["left", "right", "top"]].set_visible(False)
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(
    args,
    feature_names,
    features,
    mo,
    np,
    plt,
    selected_feature,
    selected_node,
    theta,
):
    _node = int(selected_node.value)
    _feature_index = int(selected_feature.value)
    _feature_name = feature_names[_feature_index]
    _feature_value = float(features[_node, _feature_index])

    _dense_theta = np.linspace(0.0, max(args.theta_max, 0.1), 400)
    _cos_curve = np.cos(_feature_value * _dense_theta)
    _sin_curve = np.sin(_feature_value * _dense_theta)
    _sample_cos = np.cos(_feature_value * theta)
    _sample_sin = np.sin(_feature_value * theta)

    _fig, _ax = plt.subplots(figsize=(10.5, 5.2))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")
    _ax.plot(_dense_theta, _cos_curve, linewidth=2.8, label="cos(feature x theta)")
    _ax.plot(_dense_theta, _sin_curve, linewidth=2.8, label="sin(feature x theta)")
    _ax.scatter(theta, _sample_cos, s=75, edgecolors="#111827", linewidths=1.1, zorder=4)
    _ax.scatter(theta, _sample_sin, s=75, edgecolors="#111827", linewidths=1.1, zorder=4)
    _ax.axhline(0.0, color="#6b7280", linewidth=1)
    _ax.set_xlabel("theta", color="#111827")
    _ax.set_ylabel("characteristic-function component", color="#111827")
    _ax.tick_params(colors="#111827")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend()
    _ax.set_title(
        f"Node {_node}, feature '{_feature_name}' = {_feature_value:g}: dots are the values FEATHER keeps",
        color="#111827",
    )
    _fig.tight_layout()

    _note = (
        "Because the selected binary feature is 0, cos(0 x theta) stays 1 and sin(0 x theta) stays 0. "
        "Choose a node where this feature is 1 to see the oscillation."
        if _feature_value == 0
        else
        "Because the selected binary feature is 1, the sampled values follow cos(theta) and sin(theta)."
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. Theta and the characteristic function

                FEATHER creates:

                ```python
                theta = np.linspace(0.01, theta_max, eval_points)
                ```

                Then each node feature is multiplied by every theta value and transformed with cosine and sine.

                {_note}
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(feature_names, features, np, pd, theta):
    # Match the implementation in feather.py exactly.
    X_theta_outer = np.outer(features, theta)
    X_theta = X_theta_outer.reshape(features.shape[0], -1)
    X0 = np.concatenate([np.cos(X_theta), np.sin(X_theta)], axis=1)

    x0_columns = []
    for _feature_name in feature_names:
        for _theta_index in range(len(theta)):
            x0_columns.append(f"cos { _feature_name } t{_theta_index + 1}")
    for _feature_name in feature_names:
        for _theta_index in range(len(theta)):
            x0_columns.append(f"sin { _feature_name } t{_theta_index + 1}")

    x_theta_columns = []
    for _feature_name in feature_names:
        for _theta_index in range(len(theta)):
            x_theta_columns.append(f"{_feature_name} x t{_theta_index + 1}")

    x_theta_table = pd.DataFrame(
        X_theta,
        index=[f"node {_i}" for _i in range(X_theta.shape[0])],
        columns=x_theta_columns,
    )
    return X0, x0_columns, x_theta_table


@app.cell
def _(X0, mo, plt, x0_columns):
    _fig, _ax = plt.subplots(figsize=(min(18, max(10, X0.shape[1] * 0.55)), 4.8))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")
    _im = _ax.imshow(X0, cmap="coolwarm", aspect="auto", vmin=-1.0, vmax=1.0)

    _ax.set_yticks(range(X0.shape[0]))
    _ax.set_yticklabels([f"node {i}" for i in range(X0.shape[0])], color="#111827")
    _ax.set_xticks(range(X0.shape[1]))
    _ax.set_xticklabels(x0_columns, rotation=70, ha="right", fontsize=8, color="#111827")
    _ax.set_title("X0: cosine and sine values before graph propagation", color="#111827", pad=12)

    if X0.shape[1] <= 24:
        for _i in range(X0.shape[0]):
            for _j in range(X0.shape[1]):
                _ax.text(_j, _i, f"{X0[_i, _j]:.2f}", ha="center", va="center", fontsize=7, color="#111827")

    _cbar = _fig.colorbar(_im, ax=_ax, fraction=0.025, pad=0.02)
    _cbar.set_label("value", color="#111827")
    _cbar.ax.tick_params(colors="#111827")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. X0: what FEATHER will propagate

                `X0` has shape `{X0.shape}`.

                At this point, **no neighbouring node information has been mixed yet**.
                Each row only comes from that node's own feature vector and the theta transformation.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, show_numbers, x_theta_table):
    if bool(show_numbers.value):
        mo.vstack(
            [
                mo.md("### Exact feature x theta values before cosine/sine"),
                x_theta_table,
            ]
        )
    else:
        mo.md("Exact `feature x theta` values are hidden. Turn on **Show exact numeric tables** to display them.")
    return


@app.cell
def _(A_tilde, X0, args, np):
    _current = X0.copy()
    order_embeddings = []

    for _order_number in range(1, args.order + 1):
        _current = A_tilde.dot(_current)
        order_embeddings.append(np.asarray(_current).copy())

    stage_matrices = [X0] + order_embeddings
    manual_embedding = np.concatenate(order_embeddings, axis=1)
    return manual_embedding, order_embeddings, stage_matrices


@app.cell
def _(args, mo):
    stage = mo.ui.slider(
        start=0,
        stop=int(args.order),
        step=1,
        value=int(args.order),
        label="Propagation stage to inspect (0 = X0)",
        show_value=True,
    )

    theta_sample = mo.ui.slider(
        start=1,
        stop=int(args.eval_points),
        step=1,
        value=1,
        label="theta sample to color on the graph",
        show_value=True,
    )

    component_part = mo.ui.dropdown(
        options={"Cosine": "cos", "Sine": "sin"},
        value="Cosine",
        label="Component part",
    )

    mo.vstack(
        [
            mo.md("## 4. Watch information propagate through the graph"),
            mo.hstack([stage, theta_sample, component_part], widths="equal"),
        ]
    )
    return component_part, stage, theta_sample


@app.cell
def _(
    G,
    args,
    component_part,
    feature_names,
    n_features,
    np,
    plt,
    selected_feature,
    selected_node,
    stage,
    stage_matrices,
    theta_sample,
):
    _stage = int(stage.value)
    _node = int(selected_node.value)
    _feature_index = int(selected_feature.value)
    _theta_index = int(theta_sample.value) - 1
    _part = component_part.value

    _base = _feature_index * args.eval_points + _theta_index
    _component_index = _base if _part == "cos" else n_features * args.eval_points + _base
    _component_label = f"{_part} {feature_names[_feature_index]} t{_theta_index + 1}"

    _values = np.asarray(stage_matrices[_stage])[:, _component_index]
    _pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}

    _strength_by_edge = {
        tuple(sorted((_u, _v))): 1.0 / float(_data["weight"])
        for _u, _v, _data in G.edges(data=True)
    }
    _max_strength = max(_strength_by_edge.values())
    _widths = [
        5.0 + 10.0 * (_strength_by_edge[tuple(sorted((_u, _v)))] / _max_strength)
        for _u, _v in G.edges()
    ]

    _fig, _ax = plt.subplots(figsize=(12, 4.5))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")

    import networkx as _nx

    _nx.draw_networkx_edges(G, _pos, ax=_ax, width=_widths, edge_color="#111827", alpha=1.0)
    _nodes = _nx.draw_networkx_nodes(
        G,
        _pos,
        ax=_ax,
        node_color=_values,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        node_size=[2300 if _i == _node else 1700 for _i in G.nodes()],
        edgecolors="#111827",
        linewidths=[3.5 if _i == _node else 2.0 for _i in G.nodes()],
    )

    for _i in G.nodes():
        _ax.text(
            _pos[_i][0],
            _pos[_i][1],
            f"node {_i}\n{_values[_i]:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
            fontweight="bold" if _i == _node else "normal",
        )

    _cbar = _fig.colorbar(_nodes, ax=_ax, fraction=0.035, pad=0.02)
    _cbar.set_label(_component_label, color="#111827")
    _cbar.ax.tick_params(colors="#111827")
    _ax.set_xlim(-0.45, 3.45)
    _ax.set_ylim(-0.5, 0.5)
    _ax.axis("off")
    _ax.set_title(
        f"Stage {_stage}: node values for '{_component_label}'",
        color="#111827",
        pad=16,
    )
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(A_tilde_dense, mo, np, plt, selected_node, stage):
    _node = int(selected_node.value)
    _stage = int(stage.value)

    if _stage == 0:
        _mixing_power = np.eye(A_tilde_dense.shape[0])
    else:
        _mixing_power = np.linalg.matrix_power(A_tilde_dense, _stage)

    source_influence = _mixing_power[_node]

    _fig, _ax = plt.subplots(figsize=(8.5, 4.3))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")
    _bars = _ax.bar([f"source {i}" for i in range(len(source_influence))], source_influence)
    _ax.set_ylim(0, 1.05)
    _ax.set_ylabel("share of X0 used", color="#111827")
    _ax.set_title(
        f"After {_stage} propagation step(s), where can node {_node}'s value come from?",
        color="#111827",
    )
    _ax.tick_params(colors="#111827")
    _ax.spines[["top", "right"]].set_visible(False)

    for _bar, _value in zip(_bars, source_influence):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _value + 0.025, f"{_value:.3f}", ha="center", va="bottom", color="#111827")

    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ### What `order` means visually

                At stage `{_stage}`, the selected node can be written as a weighted combination of the original `X0` rows using:

                ```text
                A_tilde^{_stage}
                ```

                The bars below show that row of `A_tilde^{_stage}` for node `{_node}`.
                Increasing `order` does **not** mean metres of travel. It means more repeated graph-propagation steps.
                """
            ),
            _fig,
        ]
    )
    return (source_influence,)


@app.cell
def _(
    X0,
    args,
    component_part,
    feature_names,
    mo,
    n_features,
    pd,
    selected_feature,
    selected_node,
    source_influence,
    stage,
    stage_matrices,
    theta_sample,
):
    _node = int(selected_node.value)
    _stage = int(stage.value)
    _feature_index = int(selected_feature.value)
    _theta_index = int(theta_sample.value) - 1
    _part = component_part.value
    _base = _feature_index * args.eval_points + _theta_index
    _component_index = _base if _part == "cos" else n_features * args.eval_points + _base
    _component_label = f"{_part} {feature_names[_feature_index]} t{_theta_index + 1}"

    _rows = []
    for _source in range(X0.shape[0]):
        _share = float(source_influence[_source])
        if _share > 1e-12:
            _x0_value = float(X0[_source, _component_index])
            _rows.append(
                {
                    "source node": _source,
                    "A_tilde^stage share": _share,
                    "source X0 value": _x0_value,
                    "contribution": _share * _x0_value,
                }
            )

    _table = pd.DataFrame(_rows)
    _reconstructed = float(_table["contribution"].sum()) if not _table.empty else 0.0
    _actual = float(stage_matrices[_stage][_node, _component_index])

    mo.vstack(
        [
            mo.md(
                f"""
                ### Reconstruct one propagated number

                Selected value:

                ```text
                node {_node}
                stage {_stage}
                component {_component_label}
                ```

                FEATHER's value: `{_actual:.10f}`  
                Reconstructed from source nodes: `{_reconstructed:.10f}`  
                Match: `{abs(_actual - _reconstructed) < 1e-10}`

                This is the easiest way to read repeated propagation:

                ```text
                final component
                = sum over source nodes(
                    A_tilde^stage share
                    x source node X0 component
                  )
                ```
                """
            ),
            _table,
        ]
    )
    return


@app.cell
def _(manual_embedding, np, plt, values_per_order):
    _fig, _ax = plt.subplots(figsize=(min(18, max(11, manual_embedding.shape[1] * 0.16)), 4.8))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")
    _limit = max(1.0, float(np.max(np.abs(manual_embedding))))
    _im = _ax.imshow(manual_embedding, cmap="coolwarm", aspect="auto", vmin=-_limit, vmax=_limit)
    _ax.set_yticks(range(manual_embedding.shape[0]))
    _ax.set_yticklabels([f"node {i}" for i in range(manual_embedding.shape[0])], color="#111827")
    _ax.set_xlabel("concatenated FEATHER columns", color="#111827")
    _ax.set_title(f"Final node embedding: shape {manual_embedding.shape}", color="#111827", pad=12)
    _ax.tick_params(axis="x", colors="#111827")

    for _boundary in range(values_per_order, manual_embedding.shape[1], values_per_order):
        _ax.axvline(_boundary - 0.5, color="#111827", linewidth=2.0)

    _order_count = manual_embedding.shape[1] // values_per_order
    for _order_index in range(_order_count):
        _center = _order_index * values_per_order + (values_per_order - 1) / 2.0
        _ax.text(_center, -0.8, f"order {_order_index + 1}", ha="center", va="bottom", color="#111827", fontsize=10, fontweight="bold")

    if manual_embedding.shape[1] > 60:
        _ax.set_xticks([])

    _cbar = _fig.colorbar(_im, ax=_ax, fraction=0.025, pad=0.02)
    _cbar.set_label("embedding value", color="#111827")
    _cbar.ax.tick_params(colors="#111827")
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(FEATHER, G, args, features, manual_embedding, mo, np):
    _actual_model = FEATHER()
    _actual_model.fit(G, features, args)
    feather_embedding = np.asarray(_actual_model.get_embedding())

    embeddings_match = bool(np.allclose(manual_embedding, feather_embedding))

    mo.md(
        f"""
        ## 5. Check the explanation against the real repository implementation

        The visual notebook manually reproduced the algorithm step by step.
        It also ran the real `FEATHER.fit()` imported from the repository.

        ```text
        manual shape: {manual_embedding.shape}
        real FEATHER shape: {feather_embedding.shape}
        values match: {embeddings_match}
        ```

        If `values match` is `True`, the visual calculations and the actual weighted FEATHER implementation agree.
        """
    )
    return (feather_embedding,)


@app.cell
def _(
    FEATHER,
    SimpleNamespace,
    args,
    compare_baseline,
    feather_embedding,
    features,
    mo,
    np,
    nx,
):
    if not bool(compare_baseline.value):
        mo.md("Baseline comparison is disabled.")
    else:
        _baseline_graph = nx.path_graph(4)
        _baseline_graph[0][1]["weight"] = 100.0
        _baseline_graph[1][2]["weight"] = 50.0
        _baseline_graph[2][3]["weight"] = 200.0
        _baseline_args = SimpleNamespace(theta_max=2.0, eval_points=3, order=2)

        _baseline_model = FEATHER()
        _baseline_model.fit(_baseline_graph, features, _baseline_args)
        _baseline_embedding = np.asarray(_baseline_model.get_embedding())

        if _baseline_embedding.shape == feather_embedding.shape:
            _mae = float(np.mean(np.abs(feather_embedding - _baseline_embedding)))
            _message = f"Mean absolute difference from baseline = `{_mae:.6f}`"
        else:
            _message = (
                f"The baseline shape is `{_baseline_embedding.shape}` but the current shape is `{feather_embedding.shape}`. "
                "Changing `eval_points` or `order` changes dimensionality, so an element-by-element difference is not meaningful."
            )

        mo.md(
            f"""
            ## 6. What changed compared with the default example?

            Baseline:

            ```text
            lengths = [100, 50, 200] metres
            theta_max = 2.0
            eval_points = 3
            order = 2
            ```

            Current:

            ```text
            theta_max = {args.theta_max}
            eval_points = {args.eval_points}
            order = {args.order}
            ```

            {_message}

            Use this section while moving **one slider at a time**. It helps separate:

            - changes to weighting (`street lengths`)
            - changes to characteristic-function sampling (`theta_max`, `eval_points`)
            - changes to graph depth and embedding size (`order`)
            """
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. How this maps to the full UrbanFEATHER project

    This app isolates the weighted FEATHER core so that the mathematics stays inspectable.
    In the full project, a city graph can contain thousands of OSM street nodes and edges.

    The conceptual mapping is:

    ```text
    full city / OSM data
            |
            v
    street-network graph
            |
            +---- edge lengths --------------------------+
            |                                             |
            v                                             v
    node feature matrix X                         inverse-distance weighting
            |                                             |
            |                                             v
            |                                         A_tilde
            |                                             |
            +------------------+--------------------------+
                               |
                               v
                         weighted FEATHER
                               |
                               v
                       node embedding matrix
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
            visualization             thesis ML work
                                      graph-level input
    ```

    The important separation for future classification work is:

    **Do not sort FEATHER rows to create the ML representation.**
    Sorting used for a human-readable plot is a visualization operation, not part of FEATHER itself.

    For graph classification, a later step should convert the variable number of node rows into a fixed-size graph representation using a permutation-invariant operation such as pooling.
    """)
    return


@app.cell
def _(
    args,
    feather_embedding,
    feature_names,
    mo,
    pd,
    project_root,
    show_numbers,
    x0_columns,
):
    embedding_columns = []
    for _order_number in range(1, args.order + 1):
        for _column in x0_columns:
            embedding_columns.append(f"order {_order_number} | {_column}")

    embedding_table = pd.DataFrame(
        feather_embedding,
        index=[f"node {_i}" for _i in range(feather_embedding.shape[0])],
        columns=embedding_columns,
    )

    _preview_columns = min(18, embedding_table.shape[1])
    _preview = embedding_table.iloc[:, :_preview_columns].round(4)
    _full_table = (
        embedding_table.round(6)
        if bool(show_numbers.value)
        else mo.md(
            "Turn on **Show exact numeric tables** to display every FEATHER output column."
        )
    )

    _real_files = sorted(project_root.glob("output/**/FeatherResult.csv"))
    if _real_files:
        _latest = max(_real_files, key=lambda _p: _p.stat().st_mtime)
        try:
            _real_preview = pd.read_csv(_latest, nrows=5)
            _real_message = mo.vstack(
                [
                    mo.md(
                        f"""
                        ### A real UrbanFEATHER output file was found

                        The app found:

                        ```text
                        {_latest.relative_to(project_root)}
                        ```

                        Only the first five rows are loaded here, so even a very large city output
                        does not need to be read fully just to inspect its structure.
                        """
                    ),
                    _real_preview,
                ]
            )
        except Exception as _exc:
            _real_message = mo.md(
                f"A `FeatherResult.csv` was found at `{_latest}`, but its preview could not be read: `{_exc}`"
            )
    else:
        _real_message = mo.md(
            """
            ### No existing `FeatherResult.csv` was found under `output/`

            That is fine. The matrix above is still the direct output of the repository's real
            `FEATHER.fit()` for the interactive toy UrbanFEATHER graph. If you later run the full
            OSM pipeline and it writes `output/.../FeatherResult.csv`, this app will automatically
            preview the newest file the next time it starts.
            """
        )

    mo.vstack(
        [
            mo.md(
                f"""
                # 8. What UrbanFEATHER actually outputs

                The core FEATHER result is **not a heatmap, wavegraph, or bar chart**.
                The direct output is a matrix:

                ```text
                rows    = nodes
                columns = feature x theta x real/imaginary x order
                ```

                For the current settings:

                ```text
                output shape = {feather_embedding.shape}
                node features = {len(feature_names)}
                eval_points = {args.eval_points}
                orders = {args.order}
                ```

                The table below previews the first {_preview_columns} columns. Every later plot is a
                **derived view of this matrix** (or, for Pandana/bar graphs, a view of separate
                accessibility-distance data).
                """
            ),
            _preview,
            _full_table,
            _real_message,
        ]
    )
    return


@app.cell
def _(args, mo):
    output_order = mo.ui.slider(
        start=1,
        stop=int(args.order),
        step=1,
        value=int(args.order),
        label="Order used by the output visualisations",
        show_value=True,
    )

    bar_limit = mo.ui.slider(
        start=100,
        stop=2500,
        step=100,
        value=500,
        label="Toy bar-graph distance range (m)",
        show_value=True,
    )

    sort_wavegraph = mo.ui.checkbox(
        value=True,
        label="Sort WaveGraph by the accessibility-distance proxy",
    )

    mo.vstack(
        [
            mo.md(
                """
                # 9. How the project visualisations are created

                The predecessor project used several different plots for different purposes.
                This section rebuilds small, self-contained versions so the transformation from
                **data -> plot** is visible.

                - **Pandana-style heatmap:** distance/accessibility data on street nodes.
                - **Bar graph:** the same distance data grouped into five distance bands.
                - **WaveGraph:** FEATHER real values averaged across theta, then sorted for display.
                - **FEATHER magnitude heatmap:** real + imaginary FEATHER information condensed to a node score.

                The app does **not** download OSM/Pandana data. Instead it derives a tiny nearest-amenity
                distance example from the four-node street graph. That keeps the explanation runnable offline.
                """
            ),
            mo.hstack([output_order, bar_limit, sort_wavegraph], widths="equal"),
        ]
    )
    return bar_limit, output_order, sort_wavegraph


@app.cell
def _(
    G,
    args,
    feature_names,
    features,
    mo,
    n_features,
    np,
    nx,
    order_embeddings,
    output_order,
    pd,
    selected_feature,
):
    _order_index = int(output_order.value) - 1
    _feature_index = int(selected_feature.value)
    _feature_name = feature_names[_feature_index]

    _block = np.asarray(order_embeddings[_order_index])
    _part_width = n_features * args.eval_points
    _real = _block[:, :_part_width].reshape(
        G.number_of_nodes(), n_features, args.eval_points
    )
    _imag = _block[:, _part_width:].reshape(
        G.number_of_nodes(), n_features, args.eval_points
    )

    # FEATHER magnitude view: pair real and imaginary values for the same theta,
    # then aggregate across evaluation points for one feature and order.
    _point_magnitude = np.sqrt(_real**2 + _imag**2)
    magnitude_scores = _point_magnitude[:, _feature_index, :].sum(axis=1)

    # WaveGraph reduction described in the bachelor project: mean over the real
    # evaluation points for one feature and order.
    wave_values = _real[:, _feature_index, :].mean(axis=1)

    # Self-contained stand-in for the Pandana distance used as a separate
    # accessibility measure / WaveGraph sort key. The full project uses Pandana;
    # here we use weighted shortest-path distance to the nearest node where the
    # selected amenity feature is present.
    _amenity_nodes = [
        _node
        for _node in G.nodes()
        if float(features[_node, _feature_index]) > 0.0
    ]

    toy_accessibility_distances = []
    for _node in G.nodes():
        if not _amenity_nodes:
            _distance = float("nan")
        else:
            _distance = min(
                float(nx.shortest_path_length(G, _node, _target, weight="weight"))
                for _target in _amenity_nodes
            )
        toy_accessibility_distances.append(_distance)

    toy_accessibility_distances = np.asarray(toy_accessibility_distances, dtype=float)

    output_score_table = pd.DataFrame(
        {
            "node": list(G.nodes()),
            "selected feature": [_feature_name] * G.number_of_nodes(),
            "toy distance to nearest amenity (m)": toy_accessibility_distances,
            "WaveGraph mean real value": wave_values,
            "FEATHER magnitude sum": magnitude_scores,
        }
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ## Derived node-level outputs for `{_feature_name}`, order {int(output_order.value)}

                This table makes the three different quantities explicit:

                - **distance** comes from the separate accessibility calculation,
                - **WaveGraph value** comes from the real FEATHER output,
                - **magnitude** combines real and imaginary FEATHER output.
                """
            ),
            output_score_table.round(5),
        ]
    )
    return magnitude_scores, toy_accessibility_distances, wave_values


@app.cell
def _(G, feature_names, magnitude_scores, output_order, plt, selected_feature):
    _feature_index = int(selected_feature.value)
    _feature_name = feature_names[_feature_index]
    _pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}

    _strengths = [1.0 / float(_data["weight"]) for _, _, _data in G.edges(data=True)]
    _max_strength = max(_strengths)
    _widths = [5.0 + 9.0 * (_s / _max_strength) for _s in _strengths]

    _fig, _ax = plt.subplots(figsize=(11.5, 4.2))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")

    import networkx as _nx

    _nx.draw_networkx_edges(
        G,
        _pos,
        ax=_ax,
        width=_widths,
        edge_color="#111827",
        alpha=1.0,
    )
    _nodes = _nx.draw_networkx_nodes(
        G,
        _pos,
        ax=_ax,
        node_color=magnitude_scores,
        cmap="viridis",
        node_size=2200,
        edgecolors="#111827",
        linewidths=2.5,
    )

    for _node, _score in enumerate(magnitude_scores):
        _ax.text(
            _pos[_node][0],
            _pos[_node][1],
            f"node {_node}\n{_score:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
        )

    _cbar = _fig.colorbar(_nodes, ax=_ax, fraction=0.035, pad=0.02)
    _cbar.set_label("summed FEATHER magnitude")
    _ax.set_xlim(-0.45, 3.45)
    _ax.set_ylim(-0.5, 0.5)
    _ax.axis("off")
    _ax.set_title(
        f"FEATHER magnitude heatmap analogue — {_feature_name}, order {int(output_order.value)}",
        color="#111827",
        pad=16,
    )
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How the FEATHER magnitude heatmap is made

    For each node, feature, order, and theta sample, pair the real and imaginary parts:

    \[
    m(\theta) = \sqrt{\operatorname{Re}(\theta)^2 + \operatorname{Im}(\theta)^2}
    \]

    Then aggregate those magnitudes across the theta samples for that feature/order.
    The resulting **one value per node** can be used as the colour of the node on a map.

    In a real city, the nodes are drawn at their geographic street-network positions.
    Here the four nodes stay in a straight line so the calculation is easy to follow.
    """)
    return


@app.cell
def _(G, feature_names, plt, selected_feature, toy_accessibility_distances):
    _feature_index = int(selected_feature.value)
    _feature_name = feature_names[_feature_index]
    _pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}

    _fig, _ax = plt.subplots(figsize=(11.5, 4.2))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")

    import networkx as _nx

    _nx.draw_networkx_edges(
        G,
        _pos,
        ax=_ax,
        width=7.0,
        edge_color="#1f2937",
    )
    _nodes = _nx.draw_networkx_nodes(
        G,
        _pos,
        ax=_ax,
        node_color=toy_accessibility_distances,
        cmap="viridis_r",
        node_size=2200,
        edgecolors="#111827",
        linewidths=2.5,
    )

    for _node, _distance in enumerate(toy_accessibility_distances):
        _ax.text(
            _pos[_node][0],
            _pos[_node][1],
            f"node {_node}\n{_distance:.0f} m",
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
        )

    _cbar = _fig.colorbar(_nodes, ax=_ax, fraction=0.035, pad=0.02)
    _cbar.set_label("distance to nearest selected amenity (m)")
    _ax.set_xlim(-0.45, 3.45)
    _ax.set_ylim(-0.5, 0.5)
    _ax.axis("off")
    _ax.set_title(
        f"Pandana-style accessibility heatmap analogue — {_feature_name}",
        color="#111827",
        pad=16,
    )
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(bar_limit, mo, np, plt, toy_accessibility_distances):
    _limit = float(bar_limit.value)
    _width = _limit / 5.0

    _counts = np.zeros(5, dtype=int)
    for _distance in toy_accessibility_distances:
        if _distance <= _width:
            _counts[0] += 1
        elif _distance <= 2 * _width:
            _counts[1] += 1
        elif _distance <= 3 * _width:
            _counts[2] += 1
        elif _distance <= 4 * _width:
            _counts[3] += 1
        else:
            _counts[4] += 1

    _percentages = 100.0 * _counts / max(1, len(toy_accessibility_distances))
    _labels = [
        f"0-{_width:.0f}",
        f"{_width:.0f}-{2*_width:.0f}",
        f"{2*_width:.0f}-{3*_width:.0f}",
        f"{3*_width:.0f}-{4*_width:.0f}",
        f"> {4*_width:.0f}",
    ]

    _fig, _ax = plt.subplots(figsize=(9.5, 4.6))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")
    _bars = _ax.bar(_labels, _percentages)
    _ax.set_ylim(0, 100)
    _ax.set_ylabel("nodes in distance band (%)", color="#111827")
    _ax.set_xlabel("distance band (metres)", color="#111827")
    _ax.set_title("Bar-graph idea: reduce all node distances to five citywide bands", color="#111827")
    _ax.tick_params(colors="#111827")
    _ax.spines[["top", "right"]].set_visible(False)

    for _bar, _value in zip(_bars, _percentages):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _value + 2,
            f"{_value:.0f}%",
            ha="center",
            va="bottom",
            color="#111827",
        )

    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ### How the bar graph is made

                The project bar graph is a **citywide summary of accessibility-distance data**, not a FEATHER embedding plot.

                With the current teaching limit `{_limit:.0f} m`, the app divides the range into four equal bands of
                `{_width:.0f} m` plus one final open-ended band. It then asks:

                > What percentage of street nodes falls into each distance band?

                The full project used the same idea with Pandana accessibility scores and city-scale distances.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(
    feature_names,
    mo,
    np,
    output_order,
    plt,
    selected_feature,
    sort_wavegraph,
    toy_accessibility_distances,
    wave_values,
):
    _feature_index = int(selected_feature.value)
    _feature_name = feature_names[_feature_index]

    if bool(sort_wavegraph.value):
        _indices = np.argsort(toy_accessibility_distances)
        _sorting_text = "sorted by the separate accessibility-distance proxy"
    else:
        _indices = np.arange(len(wave_values))
        _sorting_text = "kept in FEATHER/node order"

    _x = np.arange(len(_indices))
    _values = wave_values[_indices]
    _distances = toy_accessibility_distances[_indices]

    _fig, _ax = plt.subplots(figsize=(10.5, 5.0))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("#f8fafc")
    _ax.plot(_x, _values, marker="o", linewidth=2.5)
    _ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(
        [f"node {_node}\n{_distance:.0f}m" for _node, _distance in zip(_indices, _distances)],
        color="#111827",
    )
    _ax.set_ylabel("mean real FEATHER value across theta", color="#111827")
    _ax.set_xlabel("display order", color="#111827")
    _ax.tick_params(colors="#111827")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_title(
        f"WaveGraph analogue — {_feature_name}, order {int(output_order.value)} ({_sorting_text})",
        color="#111827",
    )
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                """
                ### How the WaveGraph is made

                The predecessor project reduced each node's **real FEATHER evaluation points** to a single mean value.
                It then sorted nodes using a separate accessibility score so the x-axis became human-readable.

                ```text
                FEATHER real values for one node/order/feature
                              |
                       mean across theta
                              |
                         one y-value
                              |
                sort nodes by accessibility score
                              |
                          WaveGraph
                ```

                The sorting is a **display transformation**. It is not part of FEATHER and should not be required by the ML pipeline.
                Turn the checkbox off above to see the same FEATHER values in their original node order.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## What each visualisation is really showing

    | Visualisation | Input data | Reduction | Main purpose |
    |---|---|---|---|
    | Pandana heatmap | accessibility distances | one distance value per node | spatial accessibility |
    | Bar graph | accessibility distances | percentages in distance bands | citywide distance distribution |
    | WaveGraph | FEATHER real values | mean theta values + display sorting | inspect FEATHER trends across nodes |
    | FEATHER magnitude heatmap | FEATHER real + imaginary values | magnitude across theta | spatial view of FEATHER response |

    This distinction matters for the thesis: the **raw FEATHER embedding matrix contains more information than any one of these plots**.
    A classifier should start from a deliberately constructed representation of the raw embedding, rather than reverse-engineering pixels or sorted WaveGraph curves.
    """)
    return


@app.cell
def _(mo):
    pooling_method = mo.ui.dropdown(
        options={
            "Mean pooling": "mean",
            "Max pooling": "max",
            "Min pooling": "min",
            "Mean + max + min": "mean_max_min",
        },
        value="Mean pooling",
        label="Graph-level pooling",
    )

    run_toy_ml = mo.ui.checkbox(
        value=True,
        label="Run the small synthetic classification demonstration",
    )

    mo.vstack(
        [
            mo.md(
                """
                # 10. From UrbanFEATHER output to whole-city classification

                FEATHER returns **one row per node**, but your thesis classifier needs **one example per city**.
                Cities also have different numbers of street nodes, so a normal classifier cannot directly consume the raw matrices.

                The simplest bridge is permutation-invariant graph pooling:

                ```text
                city street graph
                       |
                       v
                FEATHER node matrix Z
                [n_nodes x embedding_dimension]
                       |
                       v
                pool down the node rows
                (mean / max / min)
                       |
                       v
                one fixed-size city fingerprint g
                       |
                       v
                classifier
                       |
                       v
                city class
                ```

                Pooling removes dependence on arbitrary node row order while keeping each FEATHER embedding dimension aligned.
                """
            ),
            mo.hstack([pooling_method, run_toy_ml], widths="equal"),
        ]
    )
    return pooling_method, run_toy_ml


@app.cell
def _(feather_embedding, mo, np, plt, pooling_method):
    _method = pooling_method.value

    def _pool(_matrix):
        if _method == "max":
            return np.max(_matrix, axis=0)
        if _method == "min":
            return np.min(_matrix, axis=0)
        if _method == "mean_max_min":
            return np.concatenate(
                [
                    np.mean(_matrix, axis=0),
                    np.max(_matrix, axis=0),
                    np.min(_matrix, axis=0),
                ]
            )
        return np.mean(_matrix, axis=0)

    graph_embedding = _pool(np.asarray(feather_embedding))

    _shuffled = np.asarray(feather_embedding)[::-1]
    _shuffled_embedding = _pool(_shuffled)
    pooling_is_permutation_invariant = bool(
        np.allclose(graph_embedding, _shuffled_embedding)
    )

    _fig, _ax = plt.subplots(figsize=(12.0, 2.8))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")
    _limit = max(1.0, float(np.max(np.abs(graph_embedding))))
    _im = _ax.imshow(
        graph_embedding.reshape(1, -1),
        cmap="coolwarm",
        aspect="auto",
        vmin=-_limit,
        vmax=_limit,
    )
    _ax.set_yticks([0])
    _ax.set_yticklabels(["city fingerprint"])
    _ax.set_xlabel("pooled FEATHER dimensions")
    _ax.set_title(
        f"Fixed-size graph embedding after {pooling_method.value.replace('_', ' ')} pooling"
    )
    if graph_embedding.size > 80:
        _ax.set_xticks([])
    _fig.colorbar(_im, ax=_ax, fraction=0.025, pad=0.02)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ## Pool the node rows into one city vector

                Current FEATHER matrix:

                ```text
                {feather_embedding.shape[0]} nodes x {feather_embedding.shape[1]} FEATHER values
                ```

                Current graph-level vector:

                ```text
                {graph_embedding.shape[0]} values
                ```

                To demonstrate why this is useful, the app reverses the node rows and pools again.

                ```text
                same city vector after row reordering = {pooling_is_permutation_invariant}
                ```

                That is exactly the property you want when OSM node IDs have no meaningful cross-city order.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(FEATHER, args, mo, np, nx, plt, pooling_method, run_toy_ml):
    mo.stop(
        not bool(run_toy_ml.value),
        mo.md("Synthetic ML demonstration is disabled."),
    )

    try:
        from sklearn.decomposition import PCA as _PCA
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.metrics import accuracy_score as _accuracy_score
        from sklearn.metrics import confusion_matrix as _confusion_matrix
        from sklearn.model_selection import train_test_split as _train_test_split
        from sklearn.pipeline import make_pipeline as _make_pipeline
        from sklearn.preprocessing import StandardScaler as _StandardScaler
    except Exception as _exc:
        mo.stop(
            True,
            mo.md(
                f"The ML demonstration needs scikit-learn. Import failed with: `{_exc}`"
            ),
        )

    _method = pooling_method.value

    def _pool_matrix(_matrix):
        if _method == "max":
            return np.max(_matrix, axis=0)
        if _method == "min":
            return np.min(_matrix, axis=0)
        if _method == "mean_max_min":
            return np.concatenate(
                [
                    np.mean(_matrix, axis=0),
                    np.max(_matrix, axis=0),
                    np.min(_matrix, axis=0),
                ]
            )
        return np.mean(_matrix, axis=0)

    _rng = np.random.default_rng(42)
    _vectors = []
    _labels = []

    # Synthetic examples only: Class A has denser amenity presence and more
    # extra street connections; Class B is sparser. These labels are invented
    # solely to demonstrate the pipeline from graph -> FEATHER -> pooling -> ML.
    for _label in [0, 1]:
        for _example in range(24):
            _n = int(_rng.integers(5, 9))
            _graph = nx.path_graph(_n)

            if _label == 0:
                _length_low, _length_high = 50.0, 190.0
                _amenity_probability = 0.62
                _extra_edge_probability = 0.45
            else:
                _length_low, _length_high = 120.0, 430.0
                _amenity_probability = 0.28
                _extra_edge_probability = 0.12

            for _u, _v in _graph.edges():
                _graph[_u][_v]["weight"] = float(
                    _rng.uniform(_length_low, _length_high)
                )

            for _u in range(_n - 2):
                if _rng.random() < _extra_edge_probability:
                    _v = _u + 2
                    _graph.add_edge(
                        _u,
                        _v,
                        weight=float(_rng.uniform(_length_low, _length_high)),
                    )

            _features = (
                _rng.random((_n, 3)) < _amenity_probability
            ).astype(float)

            # Ensure every amenity category exists somewhere in every toy city.
            for _feature_index in range(_features.shape[1]):
                if np.sum(_features[:, _feature_index]) == 0:
                    _features[
                        int(_rng.integers(0, _n)),
                        _feature_index,
                    ] = 1.0

            _model = FEATHER()
            _model.fit(_graph, _features, args)
            _node_embedding = np.asarray(_model.get_embedding())
            _vectors.append(_pool_matrix(_node_embedding))
            _labels.append(_label)

    _X = np.vstack(_vectors)
    _y = np.asarray(_labels)

    _X_train, _X_test, _y_train, _y_test = _train_test_split(
        _X,
        _y,
        test_size=0.30,
        random_state=42,
        stratify=_y,
    )

    _classifier = _make_pipeline(
        _StandardScaler(),
        _LogisticRegression(max_iter=2000, random_state=42),
    )
    _classifier.fit(_X_train, _y_train)
    _predictions = _classifier.predict(_X_test)
    _accuracy = float(_accuracy_score(_y_test, _predictions))
    _cm = _confusion_matrix(_y_test, _predictions, labels=[0, 1])

    _scaled = _StandardScaler().fit_transform(_X)
    _pca_points = _PCA(n_components=2, random_state=42).fit_transform(_scaled)

    _scatter_fig, _scatter_ax = plt.subplots(figsize=(8.5, 5.6))
    _scatter_fig.patch.set_facecolor("white")
    _scatter_ax.set_facecolor("#f8fafc")
    for _label, _name in [(0, "Toy Class A"), (1, "Toy Class B")]:
        _mask = _y == _label
        _scatter_ax.scatter(
            _pca_points[_mask, 0],
            _pca_points[_mask, 1],
            s=65,
            alpha=0.8,
            label=_name,
            edgecolors="#111827",
            linewidths=0.6,
        )
    _scatter_ax.set_xlabel("PCA view 1")
    _scatter_ax.set_ylabel("PCA view 2")
    _scatter_ax.set_title("Toy city fingerprints after FEATHER + pooling")
    _scatter_ax.legend()
    _scatter_ax.spines[["top", "right"]].set_visible(False)
    _scatter_fig.tight_layout()

    _cm_fig, _cm_ax = plt.subplots(figsize=(5.4, 4.7))
    _cm_fig.patch.set_facecolor("white")
    _cm_ax.set_facecolor("white")
    _im = _cm_ax.imshow(_cm, cmap="Blues")
    _cm_ax.set_xticks([0, 1])
    _cm_ax.set_yticks([0, 1])
    _cm_ax.set_xticklabels(["A", "B"])
    _cm_ax.set_yticklabels(["A", "B"])
    _cm_ax.set_xlabel("predicted class")
    _cm_ax.set_ylabel("true class")
    _cm_ax.set_title("Toy held-out predictions")
    for _i in range(2):
        for _j in range(2):
            _cm_ax.text(
                _j,
                _i,
                str(int(_cm[_i, _j])),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if _cm[_i, _j] > _cm.max() / 2 else "#111827",
            )
    _cm_fig.colorbar(_im, ax=_cm_ax, fraction=0.046, pad=0.04)
    _cm_fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ## A complete classification workflow, using synthetic toy cities

                This block creates **48 generated toy street graphs** only to make the workflow executable.
                The two labels are intentionally artificial:

                - **Toy Class A:** denser amenity presence and more extra street connections.
                - **Toy Class B:** sparser amenity presence and fewer extra street connections.

                Every toy graph is processed exactly in this order:

                ```text
                graph + binary node features
                          |
                          v
                    UrbanFEATHER
                          |
                          v
                   node matrix Z
                          |
                          v
                    {pooling_method.value}
                          |
                          v
                   graph fingerprint
                          |
                          v
                  logistic regression
                          |
                          v
                       A or B
                ```

                Held-out toy accuracy: **{_accuracy:.3f}**

                **This accuracy is not a thesis result.** The labels were generated by the code itself,
                so the purpose is only to demonstrate that the UrbanFEATHER output can be turned into
                fixed-size samples and passed to a conventional classifier.
                """
            ),
            _scatter_fig,
            _cm_fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## How this becomes the real thesis classification experiment

    The real dataset should replace the synthetic toy graphs with **one graph per city**.

    ```text
    City 1 graph -> UrbanFEATHER -> pooling -> g1 -> label y1
    City 2 graph -> UrbanFEATHER -> pooling -> g2 -> label y2
    City 3 graph -> UrbanFEATHER -> pooling -> g3 -> label y3
    ...

    dataset for ML:

    X_graphs = [g1, g2, g3, ...]
    y        = [y1, y2, y3, ...]
    ```

    A sensible first classifier is **regularized logistic regression** because it gives you a simple baseline.
    Later you can compare it with other models, but the representation/evaluation setup should stay fixed.

    ### Important evaluation rules

    1. **One city is one ML sample.** Do not split nodes from the same city between training and test sets.
    2. Use the **same amenity-feature definitions and FEATHER parameters** for every city in an experiment.
    3. Fit scalers, feature selection, PCA, and classifier parameters using **training cities only**.
    4. Keep a validation strategy for choosing `theta_max`, `eval_points`, `order`, weighting rules, and pooling.
    5. Keep the final test cities untouched until the representation and classifier choices are fixed.
    6. Do **not** sort node rows for ML. Mean/max/min pooling is independent of arbitrary OSM row order.
    7. Keep both real and imaginary FEATHER information unless an experiment explicitly tests removing one part.

    The useful research question is therefore not merely *"can a classifier run?"* but:

    > **Which UrbanFEATHER representation choices produce graph fingerprints that generalize to unseen cities?**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## A compact mental model

    ```text
    Street length
        -> decides neighbour strength
        -> normalized into A_tilde

    Node features + theta
        -> cosine and sine values
        -> creates X0

    order = 1
        -> A_tilde @ X0 = X1

    order = 2
        -> A_tilde @ X1 = X2

    order = 3
        -> A_tilde @ X2 = X3

    final node embedding
        -> [X1 | X2 | X3 | ...]
    ```

    When experimenting, change **one parameter at a time** and watch three places:

    1. Does `A_tilde` change?
    2. Does `X0` change?
    3. Does the number or depth of propagation blocks change?

    That tells you *where* in UrbanFEATHER a parameter has its effect.
    """)
    return


if __name__ == "__main__":
    app.run()
