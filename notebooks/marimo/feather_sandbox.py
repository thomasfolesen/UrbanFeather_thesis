import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    # Put this file at:
    #   notebooks/marimo/feather_sandbox_visual.py
    #
    # Open it with:
    #   pixi run marimo edit notebooks/marimo/feather_sandbox_visual.py
    #
    # It intentionally reuses the project's real FEATHER implementation.
    # No global FEATHER parameters are required: marimo controls are passed
    # directly into the calculation through a SimpleNamespace.

    import sys as _sys
    from pathlib import Path as _Path
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd

    _project_root = _Path(__file__).resolve().parents[2]
    _feather_src = _project_root / "src" / "Utils" / "FEATHER" / "src"

    if str(_feather_src) not in _sys.path:
        _sys.path.insert(0, str(_feather_src))

    from feather import FEATHER

    return FEATHER, SimpleNamespace, mo, np, nx, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🪶 UrbanFEATHER visual playground

        This notebook explains **the same FEATHER calculation that your project runs**, but makes
        each intermediate step visible.

        The main idea is:

        ```text
        change a parameter
                ↓
        see what changed
                ↓
        inspect the exact math that caused it
        ```

        The default values reproduce the current weighted four-node smoke-test example:

        ```text
        node 0 ---100 m--- node 1 ---50 m--- node 2 ---200 m--- node 3

        features = [1, 2, 3, 4]
        theta_max = 2.0
        eval_points = 3
        order = 2
        ```
        """
    )
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

    show_tables = mo.ui.checkbox(
        value=False,
        label="Show the large exact-number tables",
    )

    compare_baseline = mo.ui.checkbox(
        value=False,
        label="Compare current result with smoke-test defaults",
    )

    show_sorting_demo = mo.ui.checkbox(
        value=False,
        label="Show why WaveGraph sorting must stay separate from FEATHER",
    )

    mo.vstack(
        [
            mo.md("## 0. Controls"),
            mo.md("### FEATHER parameters"),
            mo.hstack([theta_max, eval_points, order], widths="equal"),
            mo.hstack(
                [show_tables, compare_baseline, show_sorting_demo],
                widths="equal",
            ),
        ]
    )

    return (
        compare_baseline,
        eval_points,
        order,
        show_sorting_demo,
        show_tables,
        theta_max,
    )


@app.cell
def _(mo):
    length_01 = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=100,
        label="Edge 0 ↔ 1 length (m)",
        show_value=True,
    )

    length_12 = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=50,
        label="Edge 1 ↔ 2 length (m)",
        show_value=True,
    )

    length_23 = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=200,
        label="Edge 2 ↔ 3 length (m)",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("### Street lengths"),
            mo.hstack([length_01, length_12, length_23], widths="equal"),
        ]
    )

    return length_01, length_12, length_23


@app.cell
def _(mo):
    feature_0 = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.5,
        value=1.0,
        label="Node 0 feature",
        show_value=True,
    )

    feature_1 = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.5,
        value=2.0,
        label="Node 1 feature",
        show_value=True,
    )

    feature_2 = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.5,
        value=3.0,
        label="Node 2 feature",
        show_value=True,
    )

    feature_3 = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.5,
        value=4.0,
        label="Node 3 feature",
        show_value=True,
    )

    selected_node = mo.ui.dropdown(
        options={
            "Node 0": 0,
            "Node 1": 1,
            "Node 2": 2,
            "Node 3": 3,
        },
        value="Node 1",
        label="Node to inspect closely",
    )

    mo.vstack(
        [
            mo.md("### Node feature values"),
            mo.hstack(
                [feature_0, feature_1, feature_2, feature_3],
                widths="equal",
            ),
            selected_node,
        ]
    )

    return feature_0, feature_1, feature_2, feature_3, selected_node


@app.cell
def _(
    SimpleNamespace,
    eval_points,
    feature_0,
    feature_1,
    feature_2,
    feature_3,
    length_01,
    length_12,
    length_23,
    np,
    nx,
    order,
    theta_max,
):
    G = nx.path_graph(4)

    G[0][1]["weight"] = float(length_01.value)
    G[1][2]["weight"] = float(length_12.value)
    G[2][3]["weight"] = float(length_23.value)

    features = np.array(
        [
            [float(feature_0.value)],
            [float(feature_1.value)],
            [float(feature_2.value)],
            [float(feature_3.value)],
        ],
        dtype=float,
    )

    args = SimpleNamespace(
        theta_max=float(theta_max.value),
        eval_points=int(eval_points.value),
        order=int(order.value),
    )

    return G, args, features


@app.cell
def _(args, features, mo):
    _feature_count = features.shape[1]
    _values_per_order = _feature_count * args.eval_points * 2
    _values_per_node = _values_per_order * args.order

    mo.md(
        f"""
        ## Current configuration at a glance

        ```text
        {_feature_count} feature
            × {args.eval_points} evaluation point(s)
            × 2 (cosine + sine)
            × {args.order} propagation order(s)
            = {_values_per_node} final values per node
        ```

        With **{features.shape[0]} nodes**, the final FEATHER embedding will have shape:

        ```text
        ({features.shape[0]}, {_values_per_node})
        ```

        `eval_points` changes how many characteristic-function samples are taken.  
        `theta_max` changes how far along the characteristic function they are sampled.  
        `order` changes how many times graph information is propagated and how many blocks are saved.
        """
    )
    return


@app.cell
def _(FEATHER, G, pd):
    _weight_rows = []

    for _u, _v, _data in G.edges(data=True):
        _length = float(_data["weight"])
        _strength = 1.0 / _length
        _weight_rows.append(
            {
                "edge": f"{_u} ↔ {_v}",
                "length_m": _length,
                "formula": f"1 / {_length:g}",
                "raw_strength": _strength,
            }
        )

    weight_table = pd.DataFrame(_weight_rows)

    _temporary_model = FEATHER()
    A_tilde = _temporary_model._create_A_tilde(G)
    A_tilde_dense = A_tilde.toarray()

    _transition_rows = []
    for _source in G.nodes:
        for _destination in G.neighbors(_source):
            _length = float(G[_source][_destination]["weight"])
            _raw_strength = 1.0 / _length
            _transition_rows.append(
                {
                    "from": _source,
                    "to": _destination,
                    "length_m": _length,
                    "raw_strength": _raw_strength,
                    "normalized_share": A_tilde_dense[_source, _destination],
                }
            )

    transition_table = pd.DataFrame(_transition_rows)

    return A_tilde, A_tilde_dense, transition_table, weight_table


@app.cell
def _(G, features, mo, np, plt, selected_node, weight_table):
    _selected = int(selected_node.value)
    _positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}

    _raw_strength_by_edge = {
        tuple(map(int, _row["edge"].split(" ↔ "))): float(_row["raw_strength"])
        for _, _row in weight_table.iterrows()
    }

    _max_strength = max(_raw_strength_by_edge.values())
    _edge_widths = []
    _edge_labels = {}

    for _u, _v, _data in G.edges(data=True):
        _length = float(_data["weight"])
        _strength = 1.0 / _length
        _edge_widths.append(1.0 + 6.0 * (_strength / _max_strength))
        _edge_labels[(_u, _v)] = f"{_length:g} m\n1/L={_strength:.4f}"

    _node_sizes = [2500 if _node == _selected else 1800 for _node in G.nodes]
    _node_labels = {
        _node: f"node {_node}\nfeature={features[_node, 0]:g}"
        for _node in G.nodes
    }

    _fig, _ax = plt.subplots(figsize=(10, 3.0))
    _G_for_drawing = G
    import networkx as _nx

    _nx.draw_networkx_nodes(
        _G_for_drawing,
        _positions,
        node_size=_node_sizes,
        ax=_ax,
    )
    _nx.draw_networkx_edges(
        _G_for_drawing,
        _positions,
        width=_edge_widths,
        ax=_ax,
    )
    _nx.draw_networkx_labels(
        _G_for_drawing,
        _positions,
        labels=_node_labels,
        font_size=9,
        ax=_ax,
    )
    _nx.draw_networkx_edge_labels(
        _G_for_drawing,
        _positions,
        edge_labels=_edge_labels,
        font_size=9,
        label_pos=0.5,
        ax=_ax,
    )

    _ax.set_title(
        "Input graph — thicker edge means larger raw inverse-distance strength"
    )
    _ax.set_axis_off()
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                """
                ## 1. Input graph and raw edge weighting

                UrbanFEATHER currently converts street length into a raw connection strength:

                ```text
                raw strength = 1 / street length
                ```

                Therefore a shorter street becomes a stronger connection **before normalization**.
                The larger node is the node currently selected for the detailed explanations below.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, show_tables, transition_table, weight_table):
    if bool(show_tables.value):
        _content = mo.vstack(
            [
                mo.md("### Exact edge calculations"),
                weight_table,
                mo.md("### Directed normalized transition shares"),
                transition_table,
            ]
        )
    else:
        _content = mo.md(
            "*Enable **Show the large exact-number tables** if you want the raw edge and transition tables.*"
        )

    _content
    return


@app.cell
def _(A_tilde_dense, mo, np, plt, selected_node):
    _selected = int(selected_node.value)

    _fig, _ax = plt.subplots(figsize=(7.0, 5.5))
    _image = _ax.imshow(A_tilde_dense, vmin=0.0, vmax=1.0)
    _fig.colorbar(_image, ax=_ax, label="normalized transition share")

    _ax.set_xticks(range(4), labels=[f"to {_i}" for _i in range(4)])
    _ax.set_yticks(range(4), labels=[f"from {_i}" for _i in range(4)])
    _ax.set_title("A_tilde — FEATHER's normalized mixing rules")

    for _row in range(A_tilde_dense.shape[0]):
        for _col in range(A_tilde_dense.shape[1]):
            _ax.text(
                _col,
                _row,
                f"{A_tilde_dense[_row, _col]:.2f}",
                ha="center",
                va="center",
            )

    _fig.tight_layout()

    _selected_row = A_tilde_dense[_selected]
    _active = [
        f"node {_j}: {_selected_row[_j]:.3f}"
        for _j in range(len(_selected_row))
        if _selected_row[_j] > 0
    ]

    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. Normalize the edge strengths → `A_tilde`

                Every row is normalized so the outgoing shares from a connected node add to `1`.

                For **node {_selected}**, the current non-zero shares are:

                ```text
                {', '.join(_active)}
                ```

                Row sums:

                ```text
                {np.round(A_tilde_dense.sum(axis=1), 6)}
                ```

                **Interpretation:** `A_tilde` tells FEATHER **HOW much of each neighbour's previous row to take**.

                Try changing edge `1 ↔ 2` from `50 m` to `100 m` and watch node 1 move from roughly
                `1/3 + 2/3` to `1/2 + 1/2`.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(args, features, np, pd):
    theta = np.linspace(0.01, args.theta_max, args.eval_points)
    X_theta = np.outer(features, theta)
    X_theta = X_theta.reshape(features.shape[0], -1)

    X0 = np.concatenate([np.cos(X_theta), np.sin(X_theta)], axis=1)

    _eval_count = args.eval_points
    X0_columns = (
        [f"cos θ{_i + 1}" for _i in range(_eval_count)]
        + [f"sin θ{_i + 1}" for _i in range(_eval_count)]
    )

    _theta_columns = [f"θ{_i + 1}" for _i in range(_eval_count)]

    X_theta_table = pd.DataFrame(
        X_theta,
        index=[f"node {_i}" for _i in range(features.shape[0])],
        columns=_theta_columns,
    )

    X0_table = pd.DataFrame(
        X0,
        index=[f"node {_i}" for _i in range(features.shape[0])],
        columns=X0_columns,
    )

    return X0, X0_columns, X0_table, X_theta, X_theta_table, theta


@app.cell
def _(args, mo, np, plt, theta):
    _fig, _ax = plt.subplots(figsize=(10, 2.5))
    _ax.axhline(0.0, linewidth=1.0)
    _ax.scatter(theta, np.zeros_like(theta), s=90, zorder=3)

    for _index, _theta in enumerate(theta):
        _ax.annotate(
            f"θ{_index + 1}\n{_theta:.3f}",
            (_theta, 0.0),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
        )

    _padding = max(args.theta_max * 0.08, 0.05)
    _ax.set_xlim(-_padding, args.theta_max + _padding)
    _ax.set_ylim(-0.3, 0.45)
    _ax.set_yticks([])
    _ax.set_xlabel("theta")
    _ax.set_title("Where FEATHER evaluates the characteristic function")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. Choose the characteristic-function evaluation points

                Current values:

                ```text
                theta = {np.array2string(theta, precision=4)}
                ```

                - **`eval_points = {args.eval_points}`** controls the number of dots.
                - **`theta_max = {args.theta_max:g}`** controls how far to the right the dots extend.

                Change either slider at the top and this number line updates immediately.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(features, mo, np, plt, selected_node, theta):
    _selected = int(selected_node.value)
    _feature_value = float(features[_selected, 0])

    _x_max = max(float(theta[-1]), 0.1)
    _curve_theta = np.linspace(0.0, _x_max, 600)
    _cos_curve = np.cos(_feature_value * _curve_theta)
    _sin_curve = np.sin(_feature_value * _curve_theta)

    _sample_cos = np.cos(_feature_value * theta)
    _sample_sin = np.sin(_feature_value * theta)

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _ax.plot(_curve_theta, _cos_curve, label="cos(feature × theta)")
    _ax.plot(_curve_theta, _sin_curve, label="sin(feature × theta)")
    _ax.scatter(theta, _sample_cos, s=65, marker="o", label="sampled cos values")
    _ax.scatter(theta, _sample_sin, s=65, marker="x", label="sampled sin values")

    for _theta in theta:
        _ax.axvline(_theta, linewidth=0.6, alpha=0.25)

    _ax.axhline(0.0, linewidth=0.8, alpha=0.5)
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_xlabel("theta")
    _ax.set_ylabel("characteristic-function component")
    _ax.set_title(
        f"Node {_selected}: feature={_feature_value:g} transformed into cosine and sine samples"
    )
    _ax.legend(loc="best")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ### What `theta_max` and `eval_points` actually do

                For node **{_selected}**, the raw feature is **{_feature_value:g}**.

                FEATHER first forms:

                ```text
                feature × theta
                ```

                and then records **cosine and sine** at each selected theta point.

                This plot is especially useful for experimenting:

                - increase `eval_points` → more samples of the same curves;
                - increase `theta_max` → sample farther along the oscillation;
                - change this node's feature → the curves oscillate at a different rate.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(X_theta_table, X0_table, mo, show_tables):
    if bool(show_tables.value):
        _content = mo.vstack(
            [
                mo.md("### Exact `feature × theta` values"),
                X_theta_table,
                mo.md("### Exact `X0` cosine/sine values"),
                X0_table,
            ]
        )
    else:
        _content = mo.md(
            "*The exact `feature × theta` and `X0` tables are hidden. Enable the table checkbox above to inspect them.*"
        )

    _content
    return


@app.cell
def _(X0, X0_columns, mo, plt):
    _width = max(8.0, 0.75 * len(X0_columns))
    _fig, _ax = plt.subplots(figsize=(_width, 4.2))
    _image = _ax.imshow(X0, aspect="auto")
    _fig.colorbar(_image, ax=_ax, label="value")

    _ax.set_xticks(range(len(X0_columns)), labels=X0_columns, rotation=45, ha="right")
    _ax.set_yticks(range(X0.shape[0]), labels=[f"node {_i}" for _i in range(X0.shape[0])])
    _ax.set_title("X0 — characteristic-function values before graph propagation")

    for _row in range(X0.shape[0]):
        for _col in range(X0.shape[1]):
            if X0.shape[1] <= 12:
                _ax.text(
                    _col,
                    _row,
                    f"{X0[_row, _col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. `X0`: what exists before any graph mixing

                `X0` has shape **{X0.shape}**.

                Each row is one node. Each column is one cosine/sine sample.

                At this point the graph has **not** mixed information between nodes yet.

                > `X0` contains **WHAT** is available to propagate.  
                > `A_tilde` determines **HOW** it will be mixed.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(A_tilde, X0, args):
    _current_x = X0.copy()
    order_embeddings = []

    for _order_number in range(1, args.order + 1):
        _current_x = A_tilde.dot(_current_x)
        order_embeddings.append(_current_x.copy())

    return (order_embeddings,)


@app.cell
def _(args, mo):
    propagation_stage = mo.ui.slider(
        start=0,
        stop=args.order,
        step=1,
        value=min(1, args.order),
        label="Propagation stage to visualize (0 = X0)",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("## 5. Watch graph propagation happen"),
            propagation_stage,
        ]
    )

    return (propagation_stage,)


@app.cell
def _(
    A_tilde_dense,
    G,
    X0,
    X0_columns,
    mo,
    np,
    order_embeddings,
    pd,
    plt,
    propagation_stage,
    selected_node,
):
    _stage = int(propagation_stage.value)
    _selected = int(selected_node.value)

    if _stage == 0:
        _stage_matrix = X0
        _stage_name = "X0 (before propagation)"
        _walk_matrix = np.eye(A_tilde_dense.shape[0])
    else:
        _stage_matrix = order_embeddings[_stage - 1]
        _stage_name = f"X{_stage} (after {_stage} propagation step(s))"
        _walk_matrix = np.linalg.matrix_power(A_tilde_dense, _stage)

    _source_distribution = _walk_matrix[_selected]

    _distribution_table = pd.DataFrame(
        {
            "possible source node": list(range(len(_source_distribution))),
            f"weight after exactly {_stage} step(s)": _source_distribution,
        }
    )
    _distribution_table = _distribution_table[
        _distribution_table.iloc[:, 1] > 1e-12
    ].reset_index(drop=True)

    # Graph view: contributor node size represents A_tilde^stage[selected, source].
    _positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}
    _node_sizes = [
        1300 + 4200 * float(_source_distribution[_node])
        for _node in G.nodes
    ]
    _labels = {
        _node: (
            f"node {_node}\n"
            f"source weight={_source_distribution[_node]:.2f}"
            + ("\nSELECTED" if _node == _selected else "")
        )
        for _node in G.nodes
    }

    _fig_graph, _ax_graph = plt.subplots(figsize=(10, 3.0))
    import networkx as _nx

    _nx.draw_networkx_nodes(
        G,
        _positions,
        node_size=_node_sizes,
        ax=_ax_graph,
    )
    _nx.draw_networkx_edges(G, _positions, width=2.0, ax=_ax_graph)
    _nx.draw_networkx_labels(G, _positions, labels=_labels, font_size=8, ax=_ax_graph)
    _ax_graph.set_title(
        f"Which X0 nodes can contribute to node {_selected} after exactly {_stage} step(s)?"
    )
    _ax_graph.set_axis_off()
    _fig_graph.tight_layout()

    # Bar view of the exact-step source distribution.
    _fig_bar, _ax_bar = plt.subplots(figsize=(8.0, 3.5))
    _ax_bar.bar(range(len(_source_distribution)), _source_distribution)
    _ax_bar.set_xticks(range(len(_source_distribution)), labels=[f"node {_i}" for _i in range(len(_source_distribution))])
    _ax_bar.set_ylim(0.0, 1.05)
    _ax_bar.set_ylabel("source weight")
    _ax_bar.set_title(
        f"Row { _selected } of A_tilde^{_stage} — exact {_stage}-step source weights"
    )
    _fig_bar.tight_layout()

    # Heatmap of the actual X matrix at this stage.
    _width = max(8.0, 0.75 * len(X0_columns))
    _fig_heat, _ax_heat = plt.subplots(figsize=(_width, 4.2))
    _image = _ax_heat.imshow(_stage_matrix, aspect="auto")
    _fig_heat.colorbar(_image, ax=_ax_heat, label="value")
    _ax_heat.set_xticks(range(len(X0_columns)), labels=X0_columns, rotation=45, ha="right")
    _ax_heat.set_yticks(range(_stage_matrix.shape[0]), labels=[f"node {_i}" for _i in range(_stage_matrix.shape[0])])
    _ax_heat.set_title(_stage_name)
    _fig_heat.tight_layout()

    _explanation = (
        "At stage 0, no walking has occurred: the selected node only contains its own X0 row."
        if _stage == 0
        else (
            f"At order {_stage}, FEATHER has multiplied by A_tilde {_stage} time(s). "
            "The bars are the exact-step random-walk weights showing which original X0 rows can contribute. "
            "Walks may return to an earlier node, so order is a number of transitions, not a geographic radius."
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ### Selected view: node {_selected}, stage {_stage}

                {_explanation}

                The graph uses node size to show the same source-weight information as the bar chart.
                """
            ),
            _fig_graph,
            _fig_bar,
            _distribution_table,
            _fig_heat,
        ]
    )
    return


@app.cell
def _(X0_columns, mo, order_embeddings, pd, show_tables):
    if bool(show_tables.value):
        _order_views = []
        for _order_index, _embedding_at_order in enumerate(order_embeddings):
            _table = pd.DataFrame(
                _embedding_at_order,
                index=[f"node {_i}" for _i in range(_embedding_at_order.shape[0])],
                columns=X0_columns,
            )
            _order_views.extend(
                [
                    mo.md(f"### Exact values after order {_order_index + 1}"),
                    _table,
                ]
            )
        _content = mo.vstack(_order_views)
    else:
        _content = mo.md(
            "*Exact matrices for every propagation order are hidden. Turn on the table checkbox to display them.*"
        )

    _content
    return


@app.cell
def _(X0_columns, args, mo):
    trace_order = mo.ui.dropdown(
        options={
            f"Order {_order_number}": _order_number
            for _order_number in range(1, args.order + 1)
        },
        value=f"Order {args.order}",
        label="Order to trace",
    )

    trace_component = mo.ui.dropdown(
        options={
            _component_name: _component_index
            for _component_index, _component_name in enumerate(X0_columns)
        },
        value=X0_columns[0],
        label="Cosine/sine component to trace",
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 6. Trace one FEATHER number all the way back to `X0`

                Use the same selected node from the controls at the top, then choose an order and component.
                """
            ),
            mo.hstack([trace_order, trace_component], widths="equal"),
        ]
    )

    return trace_component, trace_order


@app.cell
def _(
    A_tilde_dense,
    X0,
    X0_columns,
    mo,
    order_embeddings,
    pd,
    plt,
    selected_node,
    show_tables,
    trace_component,
    trace_order,
):
    _selected_node = int(selected_node.value)
    _selected_order = int(trace_order.value)
    _component_index = int(trace_component.value)
    _component_name = X0_columns[_component_index]

    _selected_value = order_embeddings[_selected_order - 1][
        _selected_node,
        _component_index,
    ]

    if _selected_order == 1:
        _previous_matrix = X0
        _previous_label = "X0"
    else:
        _previous_matrix = order_embeddings[_selected_order - 2]
        _previous_label = f"X{_selected_order - 1}"

    _immediate_rows = []
    for _neighbour in range(A_tilde_dense.shape[1]):
        _weight = A_tilde_dense[_selected_node, _neighbour]
        if _weight > 0:
            _previous_value = _previous_matrix[_neighbour, _component_index]
            _immediate_rows.append(
                {
                    "neighbour/source row": f"node {_neighbour}",
                    "transition weight": _weight,
                    f"{_previous_label} {_component_name}": _previous_value,
                    "contribution": _weight * _previous_value,
                }
            )

    _immediate_table = pd.DataFrame(_immediate_rows)
    _immediate_sum = float(_immediate_table["contribution"].sum())

    _walk_rows = []

    def _expand_walks(_current_node, _steps_left, _path, _weight_product):
        if _steps_left == 0:
            _source_x0 = X0[_current_node, _component_index]
            _walk_rows.append(
                {
                    "walk": " → ".join(str(_node) for _node in _path),
                    "source node": _current_node,
                    "path weight": _weight_product,
                    f"X0 {_component_name}": _source_x0,
                    "contribution": _weight_product * _source_x0,
                }
            )
            return

        for _next_node in range(A_tilde_dense.shape[1]):
            _edge_weight = A_tilde_dense[_current_node, _next_node]
            if _edge_weight > 0:
                _expand_walks(
                    _next_node,
                    _steps_left - 1,
                    _path + [_next_node],
                    _weight_product * _edge_weight,
                )

    _expand_walks(
        _selected_node,
        _selected_order,
        [_selected_node],
        1.0,
    )

    _walk_table = pd.DataFrame(_walk_rows)
    _walk_sum = float(_walk_table["contribution"].sum())
    _matches = abs(_walk_sum - float(_selected_value)) < 1e-10

    _fig, _ax = plt.subplots(figsize=(8.5, 3.8))
    _labels = list(_immediate_table["neighbour/source row"])
    _values = list(_immediate_table["contribution"])
    _ax.bar(_labels, _values)
    _ax.axhline(0.0, linewidth=0.8)
    _ax.set_ylabel("contribution")
    _ax.set_title(
        f"Immediate contributions to node {_selected_node}, order {_selected_order}, {_component_name}"
    )
    _fig.tight_layout()

    _details = [
        mo.md(
            f"""
            ### Immediate matrix-multiplication view

            FEATHER calculates:

            ```text
            new value = Σ (transition weight × neighbour's previous value)
            ```

            For the selected value:

            ```text
            node       = {_selected_node}
            order      = {_selected_order}
            component  = {_component_name}
            result     = {_selected_value:.10f}
            ```

            Sum of the immediate neighbour contributions:

            ```text
            {_immediate_sum:.10f}
            ```
            """
        ),
        _immediate_table,
        _fig,
        mo.md(
            f"""
            ### Expand the same value into complete random-walk paths

            Every exact `{_selected_order}`-step path contributes:

            ```text
            product of transition weights along the path
            ×
            the source node's original X0 value
            ```

            Sum of all path contributions = `{_walk_sum:.10f}`  
            FEATHER value = `{_selected_value:.10f}`  
            Reconstruction matches = **{_matches}**
            """
        ),
    ]

    if bool(show_tables.value):
        _details.append(_walk_table)
    else:
        _details.append(
            mo.md(
                f"*There are {len(_walk_table)} exact {_selected_order}-step walk paths. Enable the table checkbox to list every one.*"
            )
        )

    mo.vstack(_details)
    return


@app.cell
def _(FEATHER, G, X0_columns, args, features, np, order_embeddings):
    manual_embedding = np.concatenate(order_embeddings, axis=1)

    _model = FEATHER()
    _model.fit(G, features, args)
    feather_embedding = _model.get_embedding()

    embeddings_match = np.allclose(manual_embedding, feather_embedding)

    final_columns = []
    for _order_index in range(1, args.order + 1):
        for _component in X0_columns:
            final_columns.append(f"order {_order_index} | {_component}")

    return embeddings_match, feather_embedding, final_columns, manual_embedding


@app.cell
def _(embeddings_match, feather_embedding, manual_embedding, mo):
    mo.md(
        f"""
        ## 7. Check the explanation against the real `FEATHER.fit()`

        The notebook manually calculated all propagation blocks and concatenated them.
        It also ran your real FEATHER implementation.

        ```text
        manual shape       = {manual_embedding.shape}
        FEATHER.fit shape  = {feather_embedding.shape}
        values match       = {embeddings_match}
        ```

        **This should stay `True`.** If you later modify `feather.py` and this becomes `False`,
        the notebook has caught a disagreement between its explanation and the real algorithm.
        """
    )
    return


@app.cell
def _(feather_embedding, final_columns, mo, pd, plt, show_tables):
    _width = max(10.0, 0.55 * feather_embedding.shape[1])
    _fig, _ax = plt.subplots(figsize=(_width, 4.5))
    _image = _ax.imshow(feather_embedding, aspect="auto")
    _fig.colorbar(_image, ax=_ax, label="embedding value")

    _ax.set_xticks(
        range(len(final_columns)),
        labels=final_columns,
        rotation=60,
        ha="right",
        fontsize=8,
    )
    _ax.set_yticks(
        range(feather_embedding.shape[0]),
        labels=[f"node {_i}" for _i in range(feather_embedding.shape[0])],
    )
    _ax.set_title("Final FEATHER node embedding — order blocks concatenated side by side")
    _fig.tight_layout()

    _items = [
        mo.md(
            f"""
            ## 8. Final node embedding

            FEATHER stores every propagation block side by side:

            ```text
            X1 | X2 | ... | X_order
            ```

            Final shape: **{feather_embedding.shape}**

            Notice that **X0 itself is not part of the final embedding in this implementation**;
            the saved blocks start after the first propagation.
            """
        ),
        _fig,
    ]

    if bool(show_tables.value):
        _items.extend(
            [
                mo.md("### Exact final embedding"),
                pd.DataFrame(
                    feather_embedding,
                    index=[f"node {_i}" for _i in range(feather_embedding.shape[0])],
                    columns=final_columns,
                ),
            ]
        )

    mo.vstack(_items)
    return


@app.cell
def _(
    FEATHER,
    SimpleNamespace,
    compare_baseline,
    feather_embedding,
    mo,
    np,
    nx,
    plt,
):
    if not bool(compare_baseline.value):
        _content = mo.md(
            "## 9. Compare with the baseline\n\nEnable **Compare current result with smoke-test defaults** at the top to see exactly what your experiment changed."
        )
    else:
        _baseline_graph = nx.path_graph(4)
        _baseline_graph[0][1]["weight"] = 100.0
        _baseline_graph[1][2]["weight"] = 50.0
        _baseline_graph[2][3]["weight"] = 200.0

        _baseline_features = np.array([[1.0], [2.0], [3.0], [4.0]])
        _baseline_args = SimpleNamespace(theta_max=2.0, eval_points=3, order=2)

        _baseline_model = FEATHER()
        _baseline_model.fit(_baseline_graph, _baseline_features, _baseline_args)
        _baseline_embedding = _baseline_model.get_embedding()

        _summary = [
            mo.md(
                f"""
                ## 9. Compare current settings with the smoke-test baseline

                Baseline:

                ```text
                theta_max   = 2.0
                eval_points = 3
                order       = 2
                lengths     = [100, 50, 200] m
                features    = [1, 2, 3, 4]
                shape       = {_baseline_embedding.shape}
                ```

                Current shape = **{feather_embedding.shape}**
                """
            )
        ]

        if feather_embedding.shape == _baseline_embedding.shape:
            _difference = feather_embedding - _baseline_embedding
            _mae = float(np.mean(np.abs(_difference)))
            _max_abs = float(np.max(np.abs(_difference)))

            _fig, _ax = plt.subplots(figsize=(10.0, 4.0))
            _image = _ax.imshow(_difference, aspect="auto")
            _fig.colorbar(_image, ax=_ax, label="current − baseline")
            _ax.set_yticks(range(4), labels=[f"node {_i}" for _i in range(4)])
            _ax.set_xlabel("embedding column")
            _ax.set_title("Where the current embedding differs from the baseline")
            _fig.tight_layout()

            _summary.extend(
                [
                    mo.md(
                        f"""
                        The shapes match, so an element-by-element difference is possible.

                        ```text
                        mean absolute difference = {_mae:.8f}
                        max absolute difference  = {_max_abs:.8f}
                        ```

                        If you changed `theta_max` while keeping the same number of columns,
                        remember that corresponding columns are now evaluated at different theta values.
                        The difference is still useful experimentally, but the semantic sampling locations changed.
                        """
                    ),
                    _fig,
                ]
            )
        else:
            _summary.append(
                mo.md(
                    """
                    The shapes are different because `eval_points` and/or `order` changed.
                    An element-by-element subtraction would therefore be misleading, so this notebook deliberately does not do it.

                    **This shape change is itself one of the effects of those parameters.**
                    """
                )
            )

        _content = mo.vstack(_summary)

    _content
    return


@app.cell
def _(
    feather_embedding,
    features,
    mo,
    np,
    pd,
    plt,
    show_sorting_demo,
):
    if not bool(show_sorting_demo.value):
        _content = mo.md(
            "## 10. FEATHER vs. WaveGraph sorting\n\nEnable the sorting demonstration at the top if you want to see why sorting should remain a visualization-only operation."
        )
    else:
        _node_ids = np.arange(feather_embedding.shape[0])
        _toy_sort_key = features[:, 0]
        _sorted_indices = np.argsort(_toy_sort_key)

        # Use one existing FEATHER column purely so we can see that sorting only
        # reorders rows; it does not create new FEATHER values.
        _column_values = feather_embedding[:, 0]

        _raw_table = pd.DataFrame(
            {
                "plot position": np.arange(len(_node_ids)),
                "node": _node_ids,
                "toy accessibility/feature": _toy_sort_key,
                "first FEATHER column": _column_values,
            }
        )

        _sorted_table = pd.DataFrame(
            {
                "plot position": np.arange(len(_node_ids)),
                "node": _node_ids[_sorted_indices],
                "toy accessibility/feature": _toy_sort_key[_sorted_indices],
                "first FEATHER column": _column_values[_sorted_indices],
            }
        )

        _fig, _ax = plt.subplots(figsize=(9.0, 4.0))
        _ax.plot(
            np.arange(len(_node_ids)),
            _column_values,
            marker="o",
            label="raw FEATHER node order",
        )
        _ax.plot(
            np.arange(len(_node_ids)),
            _column_values[_sorted_indices],
            marker="o",
            label="same values after visualization sorting",
        )
        _ax.set_xticks(np.arange(len(_node_ids)))
        _ax.set_xlabel("display position")
        _ax.set_ylabel("first FEATHER embedding value")
        _ax.set_title("Sorting changes display order, not FEATHER values")
        _ax.legend(loc="best")
        _fig.tight_layout()

        _content = mo.vstack(
            [
                mo.md(
                    """
                    ## 10. FEATHER vs. WaveGraph sorting

                    The predecessor project's WaveGraph sorted nodes by **Pandana accessibility** so a human-readable curve could be drawn.
                    That sorting is **not part of FEATHER**.

                    This tiny example has no Pandana dataframe, so the current node feature is used only as a toy sorting key.
                    The purpose is to demonstrate the separation:

                    ```text
                    FEATHER output                     visualization copy
                    node rows preserved        →       rows may be sorted for plotting
                    ```

                    For city classification, keep the raw FEATHER representation separate from this plotting transformation.
                    """
                ),
                _fig,
                mo.md("### Raw FEATHER row order"),
                _raw_table,
                mo.md("### Same rows sorted only for visualization"),
                _sorted_table,
            ]
        )

    _content
    return


@app.cell
def _(args, feather_embedding, mo):
    _values_per_order = args.eval_points * 2

    mo.md(
        f"""
        # Final mental model

        ```text
        STREET GRAPH
            │
            │  length → 1/length → normalize rows
            ▼
        A_tilde                       controls HOW information moves


        NODE FEATURES
            │
            │  × theta
            ▼
        feature × theta
            │
            │  cos + sin
            ▼
        X0                            controls WHAT information exists
            │
            │  A_tilde · X0
            ▼
        X1
            │
            │  A_tilde · X1
            ▼
        X2
            │
           ...
            │
            ▼
        concatenate X1 | X2 | ...
            │
            ▼
        FINAL NODE EMBEDDING {feather_embedding.shape}
        ```

        With the current settings, each node receives:

        ```text
        {args.eval_points} eval points × 2 (cos + sin) = {_values_per_order} values per order
        {_values_per_order} × {args.order} orders = {feather_embedding.shape[1]} final values per node
        ```

        ### What each control teaches you

        - **Street lengths:** change `A_tilde`, therefore change how strongly neighbours contribute.
        - **Node feature values:** change the characteristic-function curves and therefore `X0`.
        - **`eval_points`:** changes how many cosine/sine samples are stored per order.
        - **`theta_max`:** changes where those samples are taken along the curves.
        - **`order`:** changes how many repeated graph-propagation steps are performed and stored.

        The notebook deliberately keeps **FEATHER calculations**, **human visualization**, and the later **city-classification representation** conceptually separate.
        """
    )
    return


if __name__ == "__main__":
    app.run()
