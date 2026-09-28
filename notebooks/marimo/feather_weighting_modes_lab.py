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

    from feather import FEATHER

    return FEATHER, SimpleNamespace, mo, np, nx, pd, plt


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
        :root {
          --text: #0f172a;
          --muted: #475569;
          --border: #cbd5e1;
          --soft: #f8fafc;
          --card: #ffffff;
          --feather: #0369a1;
          --urban: #7c3aed;
          --proposed: #b45309;
        }

        .wf-card {
          background: var(--card);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 16px;
          box-shadow: 0 2px 8px rgba(15, 23, 42, 0.07);
        }

        .wf-card h1, .wf-card h2, .wf-card h3,
        .wf-card p, .wf-card li, .wf-card strong, .wf-card code {
          color: var(--text) !important;
        }

        .wf-note {
          background: #eff6ff;
          border-left: 5px solid #2563eb;
          border-radius: 10px;
          color: var(--text);
          padding: 12px 14px;
          margin: 10px 0;
        }

        .wf-warn {
          background: #fff7ed;
          border-left: 5px solid #ea580c;
          border-radius: 10px;
          color: var(--text);
          padding: 12px 14px;
          margin: 10px 0;
        }

        .wf-good {
          background: #f0fdf4;
          border-left: 5px solid #16a34a;
          border-radius: 10px;
          color: var(--text);
          padding: 12px 14px;
          margin: 10px 0;
        }

        /* Compact tables: fixed 3-decimal display and small cell padding. */
        .wf-table-wrap {
          max-width: 100%;
          overflow-x: auto;
        }

        table.wf-table {
          border-collapse: collapse;
          width: auto;
          min-width: 0;
          margin: 0 auto;
          color: #0f172a !important;
          background: white;
          font-size: 0.80rem;
          line-height: 1.15;
        }

        /*
        Marimo can apply its dark-theme table styles on top of plain HTML
        tables.  Set the cell backgrounds explicitly so every matrix stays
        readable in both light and dark Marimo themes.
        */
        table.wf-table thead,
        table.wf-table tbody,
        table.wf-table tr,
        table.wf-table th,
        table.wf-table td {
          color: #0f172a !important;
        }

        table.wf-table th,
        table.wf-table td {
          border: 1px solid #d1d5db;
          padding: 3px 6px;
          text-align: right;
          white-space: nowrap;
          background-color: #ffffff !important;
        }

        /* Column headings and row labels get a very light background. */
        table.wf-table thead th,
        table.wf-table tbody th {
          background-color: #f8fafc !important;
          font-weight: 600;
        }

        /* Prevent Marimo hover/selection styling from turning cells dark. */
        table.wf-table tbody tr:hover,
        table.wf-table tbody tr:hover th,
        table.wf-table tbody tr:hover td {
          background-color: #ffffff !important;
          color: #0f172a !important;
        }

        table.wf-table th:first-child,
        table.wf-table td:first-child {
          text-align: left;
        }
        </style>
        """
    )
    return


@app.cell
def _(mo, pd):
    def compact_table(frame):
        """Render a dataframe compactly with floating-point values limited to 3 decimals."""
        _html = frame.to_html(
            classes="wf-table",
            border=0,
            float_format=lambda value: f"{value:.3f}",
        )
        return mo.Html(f"<div class='wf-table-wrap'>{_html}</div>")

    return (compact_table,)


@app.cell
def _(mo):
    weighting_mode = mo.ui.dropdown(
        options={
            "FEATHER - topology only": "FEATHER",
            "UrbanFEATHER - inverse edge length": "UrbanFEATHER",
            "UrbanFEATHER_edgeCost_accessScore": "UrbanFEATHER_edgeCost_accessScore",
        },
        value="UrbanFEATHER_edgeCost_accessScore",
        label="Weighting mode used for propagation",
    )

    # ONE balance slider controls both weights.
    # Moving LEFT makes alpha smaller and beta larger.
    # Moving RIGHT makes alpha larger and beta smaller.
    # Therefore alpha + beta is always exactly 1.0.
    blend_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.5,
        label="beta accessibility  <-  balance  ->  alpha edge cost",
        show_value=False,
    )

    theta_max = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.5,
        value=2.0,
        label="theta_max",
        show_value=True,
    )

    eval_points = mo.ui.slider(
        start=2,
        stop=8,
        step=1,
        value=3,
        label="evaluation points",
        show_value=True,
    )

    order = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=2,
        label="random-walk order",
        show_value=True,
    )

    selected_node = mo.ui.dropdown(
        options={"A": 0, "B": 1, "C": 2, "D": 3},
        value="A",
        label="Node to inspect",
    )

    mo.vstack(
        [
            mo.Html(
                """
                <div class='wf-card'>
                  <h1>FEATHER weighting modes lab</h1>
                  <p>This app compares the same teaching graph under three random-walk rules.</p>
                  <p><strong>The edge/access weighting now uses one balance slider.</strong>
                  Moving it left gives more weight to destination accessibility;
                  moving it right gives more weight to inverse edge cost.</p>
                </div>
                """
            ),
            weighting_mode,
            blend_slider,
            mo.hstack([theta_max, eval_points, order], widths="equal"),
            selected_node,
        ],
        gap=0.7,
    )

    return blend_slider, eval_points, order, selected_node, theta_max, weighting_mode


@app.cell
def _(blend_slider, mo):
    # blend_slider.value is alpha.
    # beta is calculated automatically as the remaining share.
    edge_alpha = float(blend_slider.value)
    access_beta = 1.0 - edge_alpha

    mo.Html(
        f"""
        <div class='wf-good'>
          <strong>Current balance:</strong>
          beta = {access_beta:.2f} accessibility
          &nbsp;&nbsp;|&nbsp;&nbsp;
          alpha = {edge_alpha:.2f} edge cost
          &nbsp;&nbsp;|&nbsp;&nbsp;
          alpha + beta = {edge_alpha + access_beta:.2f}
        </div>
        """
    )

    return access_beta, edge_alpha


@app.cell
def _(np, nx):
    node_labels = ["A", "B", "C", "D"]
    feature_names = ["School", "Shops"]

    edge_lengths = {
        (0, 1): 1.0,
        (0, 2): 3.0,
        (1, 2): 5.0,
        (1, 3): 2.0,
        (2, 3): 1.0,
    }

    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    for (_u, _v), _length in edge_lengths.items():
        graph.add_edge(_u, _v, weight=_length)

    amenities = [
        {
            "name": "School 1",
            "category": "School",
            "edge": (0, 2),
            "fraction": 0.35 / edge_lengths[(0, 2)],
        },
        {
            "name": "School 2",
            "category": "School",
            "edge": (2, 3),
            "fraction": 0.62,
        },
        {
            "name": "Shop 1",
            "category": "Shops",
            "edge": (0, 1),
            "fraction": 0.35 / edge_lengths[(0, 1)],
        },
        {
            "name": "Shop 2",
            "category": "Shops",
            "edge": (1, 3),
            "fraction": 0.58,
        },
    ]

    # This teaching example calculates node-to-amenity network distance using
    # NetworkX so the complete tiny example can be inspected directly.
    _all_pairs = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    _category_index = {"School": 0, "Shops": 1}
    _distances = [[[], []] for _ in range(4)]

    for _amenity in amenities:
        _u, _v = _amenity["edge"]
        _key = tuple(sorted((_u, _v)))
        _length = edge_lengths[_key]
        _from_u = _amenity["fraction"] * _length
        _from_v = (1.0 - _amenity["fraction"]) * _length
        _feature_index = _category_index[_amenity["category"]]

        for _node in range(4):
            _distance = min(
                _all_pairs[_node][_u] + _from_u,
                _all_pairs[_node][_v] + _from_v,
            )
            _distances[_node][_feature_index].append(_distance)

    raw_X = np.zeros((4, 2), dtype=float)
    for _node in range(4):
        for _feature_index in range(2):
            raw_X[_node, _feature_index] = sum(
                1.0 / _distance
                for _distance in _distances[_node][_feature_index]
            )

    # IMPORTANT MODEL CHOICE:
    # The proposed method needs ONE destination-accessibility value per node.
    # We currently use the MEAN of the ORIGINAL raw feature vector X.
    # Example: access(A) = mean([School_A, Shops_A]).
    # This mean is a deliberate modelling choice, not a FEATHER requirement.
    access_scores = raw_X.mean(axis=1)

    # Important feature-direction difference:
    # our teaching notebook's inverse-distance features are "higher = better,"
    # while the original Pandana-generated UrbanFEATHER features appear to
    # encode distance, where "lower = better." Pandana distance features should
    # therefore be transformed before they are used directly as access scores.

    return access_scores, amenities, edge_lengths, feature_names, graph, node_labels, raw_X


@app.cell
def _(
    access_scores,
    amenities,
    compact_table,
    edge_lengths,
    feature_names,
    mo,
    node_labels,
    pd,
    plt,
    raw_X,
):
    _positions = {
        0: (-1.0, 1.0),
        1: (-1.0, -1.0),
        2: (1.0, 1.0),
        3: (1.0, -1.0),
    }

    _fig, _ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")

    for (_u, _v), _length in edge_lengths.items():
        _x1, _y1 = _positions[_u]
        _x2, _y2 = _positions[_v]
        _ax.plot([_x1, _x2], [_y1, _y2], linewidth=3.0, zorder=1)

        _mx, _my = (_x1 + _x2) / 2.0, (_y1 + _y2) / 2.0
        _dx, _dy = 0.0, 0.0
        if (_u, _v) == (0, 2):
            _dy = 0.16
        elif (_u, _v) == (1, 3):
            _dy = -0.16
        elif (_u, _v) == (1, 2):
            _dx, _dy = 0.17, 0.05
        elif (_u, _v) == (0, 1):
            _dx = -0.18
        elif (_u, _v) == (2, 3):
            _dx = 0.18

        # Explicit dark text fixes labels disappearing when marimo uses a dark theme.
        _ax.text(
            _mx + _dx,
            _my + _dy,
            f"length={_length:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "0.7",
            },
            zorder=5,
        )

    for _node, (_x, _y) in _positions.items():
        _ax.scatter([_x], [_y], s=1000, zorder=7)
        _ax.text(
            _x,
            _y,
            node_labels[_node],
            ha="center",
            va="center",
            fontsize=16,
            color="white",
            zorder=8,
        )

        # Explicit color is important: otherwise white text can be drawn on a
        # white box under some notebook themes, making the values look missing.
        _ax.text(
            _x,
            _y - 0.36,
            (
                f"School={raw_X[_node, 0]:.3f}\n"
                f"Shops={raw_X[_node, 1]:.3f}\n"
                f"mean={access_scores[_node]:.3f}"
            ),
            ha="center",
            va="top",
            fontsize=8,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.8",
            },
            zorder=9,
        )

    _label_offsets = {
        "School 1": (-0.15, 0.35),
        "School 2": (0.42, 0.02),
        "Shop 1": (-0.40, -0.04),
        "Shop 2": (0.0, -0.34),
    }

    for _amenity in amenities:
        _u, _v = _amenity["edge"]
        _ux, _uy = _positions[_u]
        _vx, _vy = _positions[_v]
        _fraction = _amenity["fraction"]
        _px = _ux + _fraction * (_vx - _ux)
        _py = _uy + _fraction * (_vy - _uy)
        _marker = "s" if _amenity["category"] == "School" else "^"

        _ax.scatter(
            [_px],
            [_py],
            marker=_marker,
            s=180,
            edgecolors="black",
            zorder=10,
        )

        _ox, _oy = _label_offsets[_amenity["name"]]
        _ax.annotate(
            _amenity["name"],
            xy=(_px, _py),
            xytext=(_px + _ox, _py + _oy),
            ha="center",
            fontsize=9,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "0.6",
            },
            arrowprops={"arrowstyle": "-", "linewidth": 1.0, "color": "#111827"},
            zorder=11,
        )

    _ax.set_xlim(-1.9, 1.9)
    _ax.set_ylim(-1.9, 1.9)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(
        "Canonical teaching graph used by the smoke test",
        fontsize=13,
        color="#111827",
    )

    _x_table = pd.DataFrame(raw_X, index=node_labels, columns=feature_names)
    _x_table["mean accessibility"] = access_scores

    mo.vstack(
        [
            mo.Html(
                """
                <div class='wf-card'>
                  <h2>1. Input graph and raw feature vector X</h2>
                  <p>The graph contains the five road edges A-B, A-C, B-C, B-D and C-D.
                  The diagonal B-C edge has length 5.</p>
                  <p>The School and Shops values use <strong>sum of inverse network distances</strong>,
                  so a closer amenity produces a larger raw X value.</p>
                  <p>For the proposed weighting method, one destination score is needed per node.
                  <strong>We currently use the mean of the raw feature vector X.</strong></p>
                </div>
                """
            ),
            mo.hstack([_fig, compact_table(_x_table)], widths=[2, 1]),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    FEATHER,
    access_beta,
    compact_table,
    edge_alpha,
    graph,
    mo,
    node_labels,
    pd,
    raw_X,
):
    _model = FEATHER()

    _feather_components = _model._create_transition_components(
        graph=graph,
        X=raw_X,
        weighting_mode="FEATHER",
    )

    _urban_components = _model._create_transition_components(
        graph=graph,
        X=raw_X,
        weighting_mode="UrbanFEATHER",
    )

    # alpha + beta is always exactly 1 because beta is computed as 1 - alpha.
    proposed_components = _model._create_transition_components(
        graph=graph,
        X=raw_X,
        weighting_mode="UrbanFEATHER_edgeCost_accessScore",
        edge_alpha=edge_alpha,
        access_beta=access_beta,
    )

    feather_A = _feather_components["A_tilde"].toarray()
    urban_A = _urban_components["A_tilde"].toarray()
    proposed_A = proposed_components["A_tilde"].toarray()

    _feather_df = pd.DataFrame(feather_A, index=node_labels, columns=node_labels)
    _urban_df = pd.DataFrame(urban_A, index=node_labels, columns=node_labels)
    _proposed_df = pd.DataFrame(proposed_A, index=node_labels, columns=node_labels)

    mo.vstack(
        [
            mo.Html(
                f"""
                <div class='wf-card'>
                  <h2>2. Side-by-side transition matrices</h2>
                  <p>All displayed matrix values are limited to <strong>3 decimal places</strong>
                  to keep the three tables compact.</p>
                  <p><strong>FEATHER:</strong> topology only.</p>
                  <p><strong>UrbanFEATHER:</strong> inverse edge length, normalized per source node.</p>
                  <p><strong>Edge + access:</strong>
                  <code>{edge_alpha:.2f} * edge_share + {access_beta:.2f} * access_share</code>.</p>
                  <p>Destination accessibility comes from the <strong>original raw X</strong>,
                  not X0, X1 or X2.</p>
                </div>
                """
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md("### FEATHER"), compact_table(_feather_df)]),
                    mo.vstack([mo.md("### UrbanFEATHER"), compact_table(_urban_df)]),
                    mo.vstack([mo.md("### Edge + access"), compact_table(_proposed_df)]),
                ],
                widths="equal",
                gap=0.5,
            ),
        ],
        gap=0.7,
    )

    return feather_A, proposed_A, proposed_components, urban_A


@app.cell
def _(
    access_scores,
    compact_table,
    mo,
    node_labels,
    pd,
    proposed_components,
):
    _edge_share = proposed_components["edge_share"].toarray()
    _access_share = proposed_components["access_share"].toarray()

    _score_table = pd.DataFrame(
        {"mean raw-X accessibility": access_scores},
        index=node_labels,
    )
    _edge_table = pd.DataFrame(_edge_share, index=node_labels, columns=node_labels)
    _access_table = pd.DataFrame(_access_share, index=node_labels, columns=node_labels)

    mo.vstack(
        [
            mo.Html(
                """
                <div class='wf-card'>
                  <h2>3. What the proposed method combines</h2>
                  <p><strong>edge_share:</strong> How cheap/short is this neighbouring edge
                  relative to my other choices?</p>
                  <p><strong>access_share:</strong> How attractive is that destination node
                  relative to my other neighbouring choices?</p>
                  <p>The destination score is currently <strong>mean(raw X features)</strong>.
                  This is deliberately simple and is a modelling choice that should be tested.</p>
                </div>
                """
            ),
            compact_table(_score_table),
            mo.hstack(
                [
                    mo.vstack([mo.md("### Edge-cost share"), compact_table(_edge_table)]),
                    mo.vstack(
                        [
                            mo.md("### Destination-accessibility share"),
                            compact_table(_access_table),
                        ]
                    ),
                ],
                widths="equal",
                gap=0.5,
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(
    FEATHER,
    SimpleNamespace,
    access_beta,
    compact_table,
    edge_alpha,
    eval_points,
    graph,
    mo,
    node_labels,
    np,
    order,
    pd,
    raw_X,
    selected_node,
    theta_max,
    weighting_mode,
):
    _mode = str(weighting_mode.value)
    _selected = int(selected_node.value)

    _args = SimpleNamespace(
        theta_max=float(theta_max.value),
        eval_points=int(eval_points.value),
        order=int(order.value),
        weighting_mode=_mode,
        edge_alpha=edge_alpha,
        access_beta=access_beta,
    )

    _model = FEATHER()
    _model.fit(graph, raw_X.copy(), _args)

    _A = _model.get_transition_matrix().toarray()
    _theta = np.linspace(0.01, _args.theta_max, _args.eval_points)
    _x_theta = np.outer(raw_X, _theta).reshape(graph.number_of_nodes(), -1)
    _X0 = np.concatenate([np.cos(_x_theta), np.sin(_x_theta)], axis=1)

    _orders = []
    _current = _X0.copy()
    for _step in range(1, _args.order + 1):
        _current = _A @ _current
        _orders.append(_current.copy())

    _first_component_rows = []
    for _destination in range(4):
        _probability = _A[_selected, _destination]
        if _probability > 0:
            _first_component_rows.append(
                {
                    "destination": node_labels[_destination],
                    "transition probability": _probability,
                    "X0 first component": _X0[_destination, 0],
                    "contribution to X1": _probability * _X0[_destination, 0],
                }
            )

    _contrib_table = pd.DataFrame(_first_component_rows)
    _x0_selected = pd.DataFrame(
        [_X0[_selected]],
        index=[f"X0_{node_labels[_selected]}"],
    )
    _x1_selected = pd.DataFrame(
        [_orders[0][_selected]],
        index=[f"X1_{node_labels[_selected]}"],
    )

    _embedding = _model.get_embedding()

    mo.vstack(
        [
            mo.Html(
                f"""
                <div class='wf-card'>
                  <h2>4. Selected mode: X -> X0 -> random walk -> X1</h2>
                  <p><strong>Mode:</strong> {_mode}</p>
                  <p><strong>X</strong> is the original raw node feature vector (School, Shops).</p>
                  <p><strong>X0</strong> is X after FEATHER evaluates theta and applies cosine/sine.
                  X0 still belongs to each node before graph propagation.</p>
                  <p><strong>X1 = A_tilde @ X0</strong>. X1 is therefore the first feature vector
                  after one random-walk propagation step.</p>
                </div>
                """
            ),
            mo.md(f"### Node {node_labels[_selected]}: X0 before propagation"),
            compact_table(_x0_selected),
            mo.md(
                f"### How the first component of X1_{node_labels[_selected]} is calculated"
            ),
            compact_table(_contrib_table),
            mo.md(f"### Node {node_labels[_selected]}: X1 after one propagation step"),
            compact_table(_x1_selected),
            mo.Html(
                f"""
                <div class='wf-note'>
                Final embedding shape: <strong>{_embedding.shape}</strong>.<br>
                With {raw_X.shape[1]} raw features, {_args.eval_points} theta points,
                2 parts (cosine + sine), and {_args.order} orders, each node receives
                {raw_X.shape[1] * _args.eval_points * 2 * _args.order} final values.
                </div>
                """
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class='wf-card'>
          <h2>5. Interpretation and cautions</h2>
          <ul>
            <li><strong>FEATHER</strong> changes only by graph topology.</li>
            <li><strong>UrbanFEATHER</strong> makes shorter edges more likely in the random walk.</li>
            <li><strong>UrbanFEATHER_edgeCost_accessScore</strong> asks both:
            “Is this edge cheap?” and “Is the destination accessible?”</li>
            <li>Accessibility is used twice in the proposed method: first to influence the
            transition probability, and again as information inside X/X0 that gets propagated.
            This is intentional for the experiment, but it may amplify highly accessible nodes
            and should be evaluated empirically.</li>
            <li>The proposed method assumes raw accessibility features are non-negative and
            that larger means better/more accessible.</li>
            <li><strong>Important data-direction warning:</strong> this teaching notebook uses
            inverse-distance features where higher means better. The original Pandana-generated
            UrbanFEATHER features appear to encode distance, where lower means better. Those
            Pandana values should therefore be transformed before being used directly as an
            access score.</li>
            <li>If all neighbouring access scores are zero, the accessibility part falls back
            to the current UrbanFEATHER edge probabilities for that row.</li>
          </ul>
        </div>
        """
    )
    return


if __name__ == "__main__":
    app.run()