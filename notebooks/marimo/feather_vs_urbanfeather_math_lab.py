import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import html
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from scipy import sparse

    project_root = Path(__file__).resolve().parents[2]
    urban_feather_src = project_root / "src" / "Utils" / "FEATHER" / "src"

    if str(urban_feather_src) not in sys.path:
        sys.path.insert(0, str(urban_feather_src))

    try:
        from feather import FEATHER as UrbanFEATHER
        urban_import_status = (
            "Loaded the real UrbanFEATHER FEATHER class from "
            f"{urban_feather_src / 'feather.py'}"
        )
    except Exception as exc:
        UrbanFEATHER = None
        urban_import_status = (
            "Could not import the repository UrbanFEATHER class. "
            "The manual math still runs, but the final repository check is skipped. "
            f"Import error: {exc}"
        )

    return (
        SimpleNamespace,
        UrbanFEATHER,
        html,
        mo,
        np,
        nx,
        pd,
        plt,
        project_root,
        sparse,
        urban_import_status,
    )


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
        :root {
          --uf-text: #0f172a;
          --uf-muted: #475569;
          --uf-border: #94a3b8;
          --uf-card: #ffffff;
          --uf-soft: #f8fafc;
          --uf-code: #111827;
          --uf-code-text: #f8fafc;
          --uf-accent: #174c2a;
          --uf-urban: #7c3aed;
          --uf-plain: #0369a1;
          --uf-warn: #fff7ed;
        }

        .uf-title, .uf-title * { color: var(--uf-text) !important; }
        .uf-card {
          background: var(--uf-card);
          color: var(--uf-text);
          border: 1px solid var(--uf-border);
          border-radius: 14px;
          padding: 16px;
          box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }
        .uf-card h1, .uf-card h2, .uf-card h3, .uf-card h4,
        .uf-card p, .uf-card li, .uf-card strong, .uf-card code {
          color: var(--uf-text) !important;
        }
        .uf-callout {
          background: #eff6ff;
          color: #0f172a;
          border-left: 5px solid #2563eb;
          border-radius: 10px;
          padding: 12px 14px;
          margin: 10px 0;
        }
        .uf-diverge {
          background: #fff7ed;
          color: #0f172a;
          border-left: 5px solid #ea580c;
          border-radius: 10px;
          padding: 12px 14px;
          margin: 10px 0;
        }
        .uf-code {
          background: var(--uf-code);
          color: var(--uf-code-text) !important;
          border: 1px solid #334155;
          border-radius: 10px;
          padding: 13px 14px;
          overflow-x: auto;
          white-space: pre;
          font-size: 0.91rem;
          line-height: 1.45;
        }
        .uf-code code { color: var(--uf-code-text) !important; }
        .uf-table-wrap { overflow-x: auto; margin-top: 8px; }
        table.uf-table {
          border-collapse: collapse;
          background: #ffffff;
          color: #0f172a !important;
          min-width: 440px;
          width: 100%;
        }
        .uf-table th {
          background: #e2e8f0;
          color: #0f172a !important;
          border: 1px solid #94a3b8;
          padding: 7px 9px;
          font-weight: 700;
        }
        .uf-table td {
          background: #ffffff;
          color: #0f172a !important;
          border: 1px solid #cbd5e1;
          padding: 7px 9px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          text-align: right;
        }
        .uf-help {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: #e2e8f0;
          color: #0f172a;
          border: 1px solid #64748b;
          font-weight: 800;
          cursor: help;
          margin-left: 6px;
        }
        .uf-sticky-controls {
          position: sticky;
          top: 0;
          z-index: 9999;
          background: rgba(248, 250, 252, 0.98);
          border: 1px solid #94a3b8;
          border-radius: 0 0 12px 12px;
          padding: 8px 10px;
          box-shadow: 0 5px 15px rgba(15, 23, 42, 0.14);
        }

        /* The floating controls panel is dark on purpose: marimo's UI widgets
           inherit readable foreground colors in both light and dark app themes. */
        .uf-controls-copy, .uf-controls-copy * {
          color: #f8fafc !important;
        }
        details.uf-details {
          background: #f8fafc;
          color: #0f172a;
          border: 1px solid #cbd5e1;
          border-radius: 10px;
          padding: 8px 10px;
          margin: 8px 0;
        }
        details.uf-details summary {
          cursor: pointer;
          color: #0f172a;
          font-weight: 700;
        }
        </style>
        """
    )
    return


@app.cell
def _(np, nx, sparse):
    class PlainFEATHERReference:
        """Exact node-level FEATHER logic supplied from FEATHER."""

        def __init__(self, theta_max=2.5, eval_points=25, order=5):
            self.theta_max = theta_max
            self.eval_points = eval_points
            self.order = order

        def _create_D_inverse(self, graph):
            index = np.arange(graph.number_of_nodes())
            values = np.array(
                [1.0 / graph.degree[node] for node in range(graph.number_of_nodes())]
            )
            shape = (graph.number_of_nodes(), graph.number_of_nodes())
            return sparse.coo_matrix((values, (index, index)), shape=shape)

        def _create_A_tilde(self, graph):
            A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
            D_inverse = self._create_D_inverse(graph)
            return D_inverse.dot(A)

        def fit(self, graph, X):
            theta = np.linspace(0.01, self.theta_max, self.eval_points)
            A_tilde = self._create_A_tilde(graph)
            X = np.outer(X, theta)
            X = X.reshape(graph.number_of_nodes(), -1)
            X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
            feature_blocks = []
            for _ in range(self.order):
                X = A_tilde.dot(X)
                feature_blocks.append(X)
            self._X = np.concatenate(feature_blocks, axis=1)

        def get_embedding(self):
            return self._X

    return (PlainFEATHERReference,)


@app.cell
def _(html, mo, np, plt):
    def _fmt(value, precision=4):
        value = float(value)
        if abs(value) < 1e-12:
            return "0"
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.{precision}f}"

    def code_block(code):
        escaped = html.escape(code.strip())
        return mo.Html(f"<pre class='uf-code'><code>{escaped}</code></pre>")

    def exact_code_details(title, code):
        escaped_title = html.escape(title)
        escaped_code = html.escape(code.strip())
        return mo.Html(
            f"""
            <details class='uf-details'>
              <summary>{escaped_title}</summary>
              <pre class='uf-code'><code>{escaped_code}</code></pre>
            </details>
            """
        )

    def callout(text, kind="info"):
        css = "uf-diverge" if kind == "diverge" else "uf-callout"
        return mo.Html(f"<div class='{css}'>{text}</div>")

    def matrix_html(matrix, row_labels, col_labels, title=None, precision=4):
        matrix = np.asarray(matrix, dtype=float)
        title_html = f"<h4>{html.escape(title)}</h4>" if title else ""
        head = "<th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in col_labels)
        body_rows = []
        for i, row in enumerate(matrix):
            cells = [f"<th>{html.escape(str(row_labels[i]))}</th>"]
            cells.extend(f"<td>{_fmt(v, precision)}</td>" for v in row)
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        return mo.Html(
            f"""
            <div class='uf-table-wrap'>
              {title_html}
              <table class='uf-table'>
                <thead><tr>{head}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
              </table>
            </div>
            """
        )

    def dataframe_html(df, title=None, precision=4):
        title_html = f"<h4>{html.escape(title)}</h4>" if title else ""
        cols = [str(c) for c in df.columns]
        rows = [str(i) for i in df.index]
        head = "<th></th>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = []
        for row_label, row in zip(rows, df.to_numpy(dtype=object)):
            cells = [f"<th>{html.escape(row_label)}</th>"]
            for value in row:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    rendered = _fmt(value, precision)
                else:
                    rendered = html.escape(str(value))
                cells.append(f"<td>{rendered}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return mo.Html(
            f"""
            <div class='uf-table-wrap'>
              {title_html}
              <table class='uf-table'>
                <thead><tr>{head}</tr></thead>
                <tbody>{''.join(body)}</tbody>
              </table>
            </div>
            """
        )

    def teaching_card(title, code, visual, meaning, accent="#334155", exact=None):
        items = [
            mo.Html(
                f"<div style='font-weight:800;font-size:1.12rem;color:#0f172a;"
                f"border-bottom:3px solid {accent};padding-bottom:6px'>{html.escape(title)}</div>"
            ),
            mo.Html("<strong style='color:#0f172a'>Teaching code</strong>"),
            code_block(code),
            exact if exact is not None else mo.Html(""),
            mo.Html("<strong style='color:#0f172a'>Computed result / math</strong>"),
            visual,
            mo.Html("<strong style='color:#0f172a'>What it means</strong>"),
            mo.Html(f"<div style='color:#0f172a;line-height:1.55'>{meaning}</div>"),
        ]
        return mo.vstack(items, gap=0.55).style(
            {
                "border": f"2px solid {accent}",
                "border-radius": "14px",
                "padding": "14px",
                "background": "#ffffff",
                "color": "#0f172a",
                "box-shadow": "0 2px 8px rgba(15,23,42,0.08)",
            }
        )

    def side_by_side(left, right):
        return mo.hstack([left, right], widths="equal", gap=1.2)

    def compare_heatmaps(left, right, row_labels, col_labels, left_title, right_title, diff_title="Urban - Plain"):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        diff = right - left
        combined = np.concatenate([left.ravel(), right.ravel()])
        vmin = float(np.min(combined))
        vmax = float(np.max(combined))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        dmax = float(np.max(np.abs(diff)))
        if dmax < 1e-12:
            dmax = 1.0

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        fig.patch.set_facecolor("white")
        for ax, data, title in zip(axes[:2], [left, right], [left_title, right_title]):
            ax.set_facecolor("white")
            im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title, color="#0f172a", fontweight="bold")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=75, ha="right", fontsize=8, color="#0f172a")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, color="#0f172a")
            if data.shape[0] <= 8 and data.shape[1] <= 14:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7, color="#0f172a", bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.78})
        dim = axes[2].imshow(diff, aspect="auto", cmap="coolwarm", vmin=-dmax, vmax=dmax)
        axes[2].set_title(diff_title, color="#0f172a", fontweight="bold")
        axes[2].set_xticks(range(len(col_labels)))
        axes[2].set_xticklabels(col_labels, rotation=75, ha="right", fontsize=8, color="#0f172a")
        axes[2].set_yticks(range(len(row_labels)))
        axes[2].set_yticklabels(row_labels, color="#0f172a")
        if diff.shape[0] <= 8 and diff.shape[1] <= 14:
            for i in range(diff.shape[0]):
                for j in range(diff.shape[1]):
                    axes[2].text(j, i, f"{diff[i, j]:.3f}", ha="center", va="center", fontsize=7, color="#0f172a", bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.78})
        fig.colorbar(im, ax=axes[:2].ravel().tolist(), shrink=0.82, label="value")
        fig.colorbar(dim, ax=axes[2], shrink=0.82, label="difference")
        return fig

    def single_heatmap(data, row_labels, col_labels, title):
        data = np.asarray(data, dtype=float)
        fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_title(title, color="#0f172a", fontweight="bold")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=75, ha="right", fontsize=8, color="#0f172a")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, color="#0f172a")
        fig.colorbar(im, ax=ax, shrink=0.85, label="value")
        return fig

    def help_icon(text):
        escaped = html.escape(text, quote=True)
        return mo.Html(f"<span class='uf-help' title='{escaped}'>?</span>")

    return (
        _fmt,
        callout,
        code_block,
        compare_heatmaps,
        dataframe_html,
        exact_code_details,
        help_icon,
        matrix_html,
        side_by_side,
        single_heatmap,
        teaching_card,
    )


@app.cell
def _(mo, project_root, urban_import_status):
    mo.Html(
        f"""
        <div class='uf-card uf-title'>
          <h1>FEATHER vs UrbanFEATHER - side-by-side math lab</h1>
          <p>
            This app uses the same four-node graph and the same continuous School/Shops
            feature matrix on both sides. The goal is to isolate the mathematical effect of
            UrbanFEATHER's distance-aware transition weighting.
          </p>
          <p><strong>Default FEATHER parameters:</strong> theta_max = 2.0, eval_points = 3, order = 3.</p>
          <p><strong>Default feature mode:</strong> features are derived from two Schools and two Shops using network shortest-path distances.</p>
          <p><strong>Important:</strong> amenity-to-feature conversion is an upstream teaching layer in this app. FEATHER itself receives the completed feature matrix X.</p>
          <p><strong>Tip:</strong> use the floating <strong>Graph</strong> button in the top-left to keep a compact copy of the current graph visible while you compare matrices farther down the page.</p>
          <hr>
          <p><strong>Repository root:</strong> {project_root}</p>
          <p><strong>UrbanFEATHER verification:</strong> {urban_import_status}</p>
        </div>
        """
    )
    return


@app.cell
def _(mo):
    controls_toggle = mo.ui.button(
        value=False,
        on_click=lambda value: not value,
        label="Controls",
    )
    graph_toggle = mo.ui.button(
        value=False,
        on_click=lambda value: not value,
        label="Graph",
    )

    feature_mode = mo.ui.radio(
        options={
            "Derived from amenity locations": "derived",
            "Manual continuous features": "manual",
        },
        value="Derived from amenity locations",
        label="Feature mode",
    )

    feature_rule = mo.ui.dropdown(
        options={
            "Sum of inverse distances (all amenities)": "sum_inverse",
            "Nearest amenity only": "nearest_inverse",
            "Mean of k nearest inverse distances": "mean_k_inverse",
            "Count within threshold": "count_threshold",
        },
        value="Sum of inverse distances (all amenities)",
        label="Derived feature rule",
    )

    k_nearest = mo.ui.slider(
        start=1,
        stop=2,
        step=1,
        value=2,
        label="k for k-nearest rule",
        show_value=True,
    )
    count_threshold = mo.ui.slider(
        start=0.5,
        stop=12.0,
        step=0.5,
        value=4.0,
        label="distance threshold",
        show_value=True,
    )

    theta_max = mo.ui.slider(start=0.5, stop=5.0, step=0.5, value=2.0, label="theta_max", show_value=True)
    eval_points = mo.ui.slider(start=2, stop=8, step=1, value=3, label="eval_points", show_value=True)
    order = mo.ui.slider(start=1, stop=5, step=1, value=3, label="order", show_value=True)
    inspect_stage = mo.ui.slider(start=1, stop=5, step=1, value=1, label="propagation stage to inspect", show_value=True)

    length_ab = mo.ui.slider(start=0.5, stop=10.0, step=0.5, value=1.0, label="A-B length", show_value=True)
    length_ac = mo.ui.slider(start=0.5, stop=10.0, step=0.5, value=3.0, label="A-C length", show_value=True)
    length_bc = mo.ui.slider(start=0.5, stop=10.0, step=0.5, value=5.0, label="B-C length", show_value=True)
    length_bd = mo.ui.slider(start=0.5, stop=10.0, step=0.5, value=2.0, label="B-D length", show_value=True)
    length_cd = mo.ui.slider(start=0.5, stop=10.0, step=0.5, value=1.0, label="C-D length", show_value=True)

    manual_school_a = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=1.0, label="A School", show_value=True)
    manual_school_b = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=3.0, label="B School", show_value=True)
    manual_school_c = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=2.0, label="C School", show_value=True)
    manual_school_d = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=4.0, label="D School", show_value=True)

    manual_shops_a = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=4.0, label="A Shops", show_value=True)
    manual_shops_b = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=2.0, label="B Shops", show_value=True)
    manual_shops_c = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=5.0, label="C Shops", show_value=True)
    manual_shops_d = mo.ui.slider(start=0.0, stop=6.0, step=0.25, value=1.0, label="D Shops", show_value=True)

    selected_node = mo.ui.dropdown(options={"A": 0, "B": 1, "C": 2, "D": 3}, value="A", label="Follow node")

    show_a_exact = mo.ui.checkbox(value=False, label="Show exact A values")
    show_d_exact = mo.ui.checkbox(value=False, label="Show exact D values")
    show_dinv_exact = mo.ui.checkbox(value=False, label="Show exact D-inverse values")
    show_atilde_exact = mo.ui.checkbox(value=False, label="Show exact A-tilde values")
    show_outer_exact = mo.ui.checkbox(value=False, label="Show exact feature x theta values")
    show_x0_exact = mo.ui.checkbox(value=False, label="Show exact X0 values")
    show_prop_exact = mo.ui.checkbox(value=False, label="Show exact propagation values")
    show_embedding_exact = mo.ui.checkbox(value=False, label="Show exact final embedding values")

    return (
        controls_toggle,
        graph_toggle,
        count_threshold,
        eval_points,
        feature_mode,
        feature_rule,
        inspect_stage,
        k_nearest,
        length_ab,
        length_ac,
        length_bc,
        length_bd,
        length_cd,
        manual_school_a,
        manual_school_b,
        manual_school_c,
        manual_school_d,
        manual_shops_a,
        manual_shops_b,
        manual_shops_c,
        manual_shops_d,
        order,
        selected_node,
        show_a_exact,
        show_atilde_exact,
        show_d_exact,
        show_dinv_exact,
        show_embedding_exact,
        show_outer_exact,
        show_prop_exact,
        show_x0_exact,
        theta_max,
    )


@app.cell
def _(
    controls_toggle,
    count_threshold,
    eval_points,
    feature_mode,
    feature_rule,
    help_icon,
    inspect_stage,
    k_nearest,
    length_ab,
    length_ac,
    length_bc,
    length_bd,
    length_cd,
    manual_school_a,
    manual_school_b,
    manual_school_c,
    manual_school_d,
    manual_shops_a,
    manual_shops_b,
    manual_shops_c,
    manual_shops_d,
    mo,
    order,
    selected_node,
    theta_max,
):
    _rule_help = help_icon(
        "Feature-rule alternatives: Nearest only uses the closest amenity; Sum of inverse distances uses every amenity and gives closer ones more influence; Mean of k nearest uses only the k closest; Count within threshold ignores exact distance and counts how many amenities are reachable inside the threshold."
    )

    _derived_controls = mo.vstack(
        [
            mo.hstack([feature_rule, _rule_help], gap=0.5),
            mo.hstack([k_nearest, count_threshold], widths="equal"),
        ],
        gap=0.4,
    )

    _manual_controls = mo.vstack(
        [
            mo.Html("<strong style='color:#f8fafc'>Manual School values</strong>"),
            mo.hstack([manual_school_a, manual_school_b, manual_school_c, manual_school_d], widths="equal"),
            mo.Html("<strong style='color:#f8fafc'>Manual Shops values</strong>"),
            mo.hstack([manual_shops_a, manual_shops_b, manual_shops_c, manual_shops_d], widths="equal"),
        ],
        gap=0.4,
    )

    _full_controls = mo.vstack(
        [
            mo.Html("<strong style='color:#f8fafc'>Experiment controls</strong>"),
            feature_mode,
            _derived_controls if feature_mode.value == "derived" else _manual_controls,
            mo.Html("<strong style='color:#f8fafc'>FEATHER parameters</strong>"),
            mo.hstack([theta_max, eval_points, order, inspect_stage, selected_node], widths="equal"),
            mo.Html("<strong style='color:#f8fafc'>Street lengths</strong>"),
            mo.hstack([length_ab, length_ac, length_bc, length_bd, length_cd], widths="equal"),
            mo.Html(
                "<div style='color:#cbd5e1;font-size:0.9rem'>Click <strong style='color:#ffffff'>Controls</strong> again to collapse this panel. All downstream calculations react automatically to changed values.</div>"
            ),
        ],
        gap=0.55,
    )

    _panel = mo.vstack(
        [
            controls_toggle,
            _full_controls if controls_toggle.value else mo.Html(""),
        ],
        gap=0.4,
    ).style(
        {
            "position": "fixed",
            "top": "10px",
            "right": "16px",
            "z-index": "9999",
            "width": "min(1180px, calc(100vw - 40px))" if controls_toggle.value else "auto",
            "max-height": "86vh",
            "overflow-y": "auto",
            "background": "rgba(17,24,39,0.98)",
            "padding": "8px 10px",
            "border": "1px solid #64748b",
            "border-radius": "12px",
            "box-shadow": "0 5px 15px rgba(15,23,42,0.35)",
            "color": "#f8fafc",
        }
    )
    _panel
    return


@app.cell
def _(
    count_threshold,
    eval_points,
    feature_mode,
    feature_rule,
    k_nearest,
    length_ab,
    length_ac,
    length_bc,
    length_bd,
    length_cd,
    manual_school_a,
    manual_school_b,
    manual_school_c,
    manual_school_d,
    manual_shops_a,
    manual_shops_b,
    manual_shops_c,
    manual_shops_d,
    np,
    nx,
    order,
    selected_node,
    theta_max,
):
    node_labels = ["A", "B", "C", "D"]
    feature_names = ["School", "Shops"]

    edge_lengths = {
        (0, 1): float(length_ab.value),
        (0, 2): float(length_ac.value),
        (1, 2): float(length_bc.value),
        (1, 3): float(length_bd.value),
        (2, 3): float(length_cd.value),
    }

    graph_lengths = nx.Graph()
    graph_lengths.add_nodes_from(range(4))
    for (_u, _v), _length in edge_lengths.items():
        graph_lengths.add_edge(_u, _v, weight=float(_length))

    graph_plain = nx.Graph()
    graph_plain.add_nodes_from(range(4))
    graph_plain.add_edges_from(edge_lengths.keys())

    # The amenities are POIs located directly on street edges. They are not FEATHER nodes.
    # fraction is measured from the first endpoint toward the second endpoint.
    amenities = [
        {
            "name": "School 1",
            "category": "School",
            "edge": (0, 2),
            # Place School 1 close to A but outside the node circle,
            # so it remains visible as a point on edge A-C.
            "fraction": 0.35 / edge_lengths[(0, 2)],
            "label_offset": (-0.12, 0.40),
        },
        {
            "name": "School 2",
            "category": "School",
            "edge": (2, 3),
            # Slightly below the midpoint of C-D to avoid overlapping the length label.
            "fraction": 0.62,
            "label_offset": (0.50, 0.00),
        },
        {
            "name": "Shop 1",
            "category": "Shops",
            "edge": (0, 1),
            # Place Shop 1 close to A but outside the node circle,
            # so it remains visible as a point on edge A-B.
            "fraction": 0.35 / edge_lengths[(0, 1)],
            "label_offset": (-0.52, -0.06),
        },
        {
            "name": "Shop 2",
            "category": "Shops",
            "edge": (1, 3),
            # Slightly right of the midpoint of B-D so it does not sit under the edge-length label.
            "fraction": 0.58,
            "label_offset": (0.00, -0.42),
        },
    ]

    all_pairs = dict(nx.all_pairs_dijkstra_path_length(graph_lengths, weight="weight"))

    amenity_distance_rows = []
    distances_by_category = {"School": {n: [] for n in range(4)}, "Shops": {n: [] for n in range(4)}}

    for _amenity in amenities:
        _u, _v = _amenity["edge"]
        _edge_length = edge_lengths[tuple(sorted((_u, _v)))]
        _from_u = _amenity["fraction"] * _edge_length
        _from_v = (1.0 - _amenity["fraction"]) * _edge_length
        for _node in range(4):
            _through_u = all_pairs[_node][_u] + _from_u
            _through_v = all_pairs[_node][_v] + _from_v
            _distance = min(_through_u, _through_v)
            distances_by_category[_amenity["category"]][_node].append((_amenity["name"], _distance))
            amenity_distance_rows.append(
                {
                    "node": node_labels[_node],
                    "category": _amenity["category"],
                    "amenity": _amenity["name"],
                    "network_distance": _distance,
                }
            )

    def _feature_from_distances(named_distances, rule, k_value, threshold):
        distances = sorted(float(distance) for _, distance in named_distances)
        if rule == "nearest_inverse":
            return 1.0 / distances[0]
        if rule == "mean_k_inverse":
            selected = distances[: max(1, min(int(k_value), len(distances)))]
            return float(np.mean([1.0 / d for d in selected]))
        if rule == "count_threshold":
            return float(sum(d <= float(threshold) for d in distances))
        return float(sum(1.0 / d for d in distances))

    derived_features = np.zeros((4, 2), dtype=float)
    for _node in range(4):
        derived_features[_node, 0] = _feature_from_distances(
            distances_by_category["School"][_node],
            feature_rule.value,
            k_nearest.value,
            count_threshold.value,
        )
        derived_features[_node, 1] = _feature_from_distances(
            distances_by_category["Shops"][_node],
            feature_rule.value,
            k_nearest.value,
            count_threshold.value,
        )

    manual_features = np.array(
        [
            [float(manual_school_a.value), float(manual_shops_a.value)],
            [float(manual_school_b.value), float(manual_shops_b.value)],
            [float(manual_school_c.value), float(manual_shops_c.value)],
            [float(manual_school_d.value), float(manual_shops_d.value)],
        ],
        dtype=float,
    )

    features = derived_features if feature_mode.value == "derived" else manual_features
    theta = np.linspace(0.01, float(theta_max.value), int(eval_points.value))
    selected_node_index = int(selected_node.value)
    actual_order = int(order.value)

    return (
        actual_order,
        amenities,
        amenity_distance_rows,
        derived_features,
        distances_by_category,
        edge_lengths,
        feature_names,
        features,
        graph_lengths,
        graph_plain,
        node_labels,
        selected_node_index,
        theta,
    )


@app.cell
def _(actual_order, edge_lengths, features, graph_plain, np, nx, theta):
    A_plain = nx.to_numpy_array(graph_plain, nodelist=range(4), weight=None, dtype=float)

    strength_graph = nx.Graph()
    strength_graph.add_nodes_from(range(4))
    for (_u, _v), _length in edge_lengths.items():
        strength_graph.add_edge(_u, _v, weight=1.0 / float(_length))

    A_urban = nx.to_numpy_array(strength_graph, nodelist=range(4), weight="weight", dtype=float)

    degree_plain = np.array([graph_plain.degree(_node) for _node in range(4)], dtype=float)
    strength_urban = np.array(
        [strength_graph.degree(_node, weight="weight") for _node in range(4)],
        dtype=float,
    )

    D_plain = np.diag(degree_plain)
    D_urban = np.diag(strength_urban)
    Dinv_plain = np.diag(1.0 / degree_plain)
    Dinv_urban = np.diag(1.0 / strength_urban)

    Atilde_plain = Dinv_plain @ A_plain
    Atilde_urban = Dinv_urban @ A_urban

    outer_raw = np.outer(features, theta)
    X_theta = outer_raw.reshape(4, -1)
    X0 = np.concatenate([np.cos(X_theta), np.sin(X_theta)], axis=1)

    plain_orders = []
    urban_orders = []
    current_plain = X0.copy()
    current_urban = X0.copy()
    for _ in range(actual_order):
        current_plain = Atilde_plain @ current_plain
        current_urban = Atilde_urban @ current_urban
        plain_orders.append(current_plain.copy())
        urban_orders.append(current_urban.copy())

    plain_embedding = np.concatenate(plain_orders, axis=1)
    urban_embedding = np.concatenate(urban_orders, axis=1)

    base_columns = []
    for _feature_name in ["School", "Shops"]:
        for _i in range(len(theta)):
            base_columns.append(f"{_feature_name} cos t{_i + 1}")
    for _feature_name in ["School", "Shops"]:
        for _i in range(len(theta)):
            base_columns.append(f"{_feature_name} sin t{_i + 1}")

    # np.outer flattens the 4x2 feature matrix node-by-node. After reshape,
    # columns are School theta..., Shops theta... for each node. Cos and sin
    # then duplicate that block in the same order.
    x0_columns = []
    for _part in ["cos", "sin"]:
        for _feature_name in ["School", "Shops"]:
            for _i in range(len(theta)):
                x0_columns.append(f"{_feature_name} {_part} t{_i + 1}")

    embedding_columns = []
    for _r in range(1, actual_order + 1):
        embedding_columns.extend([f"r{_r} {_name}" for _name in x0_columns])

    return (
        A_plain,
        A_urban,
        Atilde_plain,
        Atilde_urban,
        D_plain,
        D_urban,
        Dinv_plain,
        Dinv_urban,
        X0,
        X_theta,
        embedding_columns,
        plain_embedding,
        plain_orders,
        strength_graph,
        urban_embedding,
        urban_orders,
        x0_columns,
    )


@app.cell
def _(mo):
    get_step1, set_step1 = mo.state(True)
    get_step2, set_step2 = mo.state(True)
    get_step3, set_step3 = mo.state(True)
    get_step4, set_step4 = mo.state(True)
    get_step5, set_step5 = mo.state(True)
    get_step6, set_step6 = mo.state(True)
    get_step7, set_step7 = mo.state(True)
    get_step8, set_step8 = mo.state(True)
    get_step9, set_step9 = mo.state(True)
    return (
        get_step1,
        get_step2,
        get_step3,
        get_step4,
        get_step5,
        get_step6,
        get_step7,
        get_step8,
        get_step9,
        set_step1,
        set_step2,
        set_step3,
        set_step4,
        set_step5,
        set_step6,
        set_step7,
        set_step8,
        set_step9,
    )


@app.cell
def _(
    mo,
    set_step1,
    set_step2,
    set_step3,
    set_step4,
    set_step5,
    set_step6,
    set_step7,
    set_step8,
    set_step9,
):
    open_step1 = mo.ui.button(label="Open step 1 - graph and amenities", on_click=lambda _: set_step1(True))
    close_step1 = mo.ui.button(label="Close step 1", on_click=lambda _: set_step1(False))
    open_step2 = mo.ui.button(label="Open step 2 - adjacency matrix A", on_click=lambda _: set_step2(True))
    close_step2 = mo.ui.button(label="Close step 2", on_click=lambda _: set_step2(False))
    open_step3 = mo.ui.button(label="Open step 3 - degree/strength matrix D", on_click=lambda _: set_step3(True))
    close_step3 = mo.ui.button(label="Close step 3", on_click=lambda _: set_step3(False))
    open_step4 = mo.ui.button(label="Open step 4 - D inverse", on_click=lambda _: set_step4(True))
    close_step4 = mo.ui.button(label="Close step 4", on_click=lambda _: set_step4(False))
    open_step5 = mo.ui.button(label="Open step 5 - normalized matrix A-tilde", on_click=lambda _: set_step5(True))
    close_step5 = mo.ui.button(label="Close step 5", on_click=lambda _: set_step5(False))
    open_step6 = mo.ui.button(label="Open step 6 - theta and feature x theta", on_click=lambda _: set_step6(True))
    close_step6 = mo.ui.button(label="Close step 6", on_click=lambda _: set_step6(False))
    open_step7 = mo.ui.button(label="Open step 7 - cosine/sine X0", on_click=lambda _: set_step7(True))
    close_step7 = mo.ui.button(label="Close step 7", on_click=lambda _: set_step7(False))
    open_step8 = mo.ui.button(label="Open step 8 - propagation orders", on_click=lambda _: set_step8(True))
    close_step8 = mo.ui.button(label="Close step 8", on_click=lambda _: set_step8(False))
    open_step9 = mo.ui.button(label="Open step 9 - final embedding and verification", on_click=lambda _: set_step9(True))
    close_step9 = mo.ui.button(label="Close step 9", on_click=lambda _: set_step9(False))
    return (
        close_step1,
        close_step2,
        close_step3,
        close_step4,
        close_step5,
        close_step6,
        close_step7,
        close_step8,
        close_step9,
        open_step1,
        open_step2,
        open_step3,
        open_step4,
        open_step5,
        open_step6,
        open_step7,
        open_step8,
        open_step9,
    )


@app.cell
def _(np, plt):
    def plot_graph_with_amenities(edge_lengths, amenities, features, feature_names, compact=False):
        positions = {
            0: (-1.0, 1.0),
            1: (-1.0, -1.0),
            2: (1.0, 1.0),
            3: (1.0, -1.0),
        }
        labels = {0: "A", 1: "B", 2: "C", 3: "D"}
        edge_keys = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

        _figsize = (6.2, 5.2) if compact else (9.2, 7.8)
        fig, ax = plt.subplots(figsize=_figsize, constrained_layout=True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for u, v in edge_keys:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], linewidth=4.0, color="#174c2a", zorder=1)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dx = dy = 0.0
            key = tuple(sorted((u, v)))
            if key == (0, 2):
                dy = 0.16
            elif key == (1, 3):
                dy = -0.16
            elif key == (1, 2):
                dx, dy = 0.12, 0.02
            elif key == (0, 1):
                dx = -0.20
            elif key == (2, 3):
                dx = 0.20
            ax.text(
                mx + dx,
                my + dy,
                f"length={edge_lengths[key]:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                color="#0f172a",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#94a3b8"},
                zorder=5,
            )

        for node, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.27, facecolor="#df64b0", edgecolor="#174c2a", linewidth=3.2, zorder=4)
            ax.add_patch(circle)
            ax.text(x, y, labels[node], ha="center", va="center", fontsize=21, color="#174c2a", zorder=5)
            ax.text(
                x,
                y - 0.42,
                f"{feature_names[0]}={features[node, 0]:.3f}\n{feature_names[1]}={features[node, 1]:.3f}",
                ha="center",
                va="top",
                fontsize=9.5,
                color="#0f172a",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
                zorder=6,
            )

        for amenity in amenities:
            u, v = amenity["edge"]
            ux, uy = positions[u]
            vx, vy = positions[v]
            f = float(amenity["fraction"])
            attach_x = ux + f * (vx - ux)
            attach_y = uy + f * (vy - uy)
            is_school = amenity["category"] == "School"
            marker = "s" if is_school else "^"
            color = "#2563eb" if is_school else "#ea580c"

            # The amenity marker itself is the point on the street edge.
            ax.scatter(
                [attach_x],
                [attach_y],
                s=300,
                marker=marker,
                color=color,
                edgecolors="#111827",
                linewidths=1.6,
                zorder=10,
            )

            _dx, _dy = amenity["label_offset"]
            _label_x = attach_x + _dx
            _label_y = attach_y + _dy
            ax.annotate(
                amenity["name"],
                xy=(attach_x, attach_y),
                xytext=(_label_x, _label_y),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=10,
                color="#0f172a",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "linewidth": 1.2},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
                zorder=11,
            )

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.05, 2.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Shared teaching graph: roads, 2 Schools, 2 Shops, and current feature values",
            fontsize=11 if compact else 15,
            color="#0f172a",
            fontweight="bold",
        )
        return fig

    def plot_amenity_distance_bars(distance_rows, node_labels):
        school = {n: [] for n in node_labels}
        shops = {n: [] for n in node_labels}
        for row in distance_rows:
            target = school if row["category"] == "School" else shops
            target[row["node"]].append(row["network_distance"])

        x = np.arange(len(node_labels))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 4.7), constrained_layout=True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for idx in range(2):
            ax.bar(x - 1.5 * width + idx * width, [school[n][idx] for n in node_labels], width, label=f"School {idx + 1}")
            ax.bar(x + 0.5 * width + idx * width, [shops[n][idx] for n in node_labels], width, label=f"Shop {idx + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(node_labels)
        ax.set_ylabel("network distance")
        ax.set_title("Shortest network distance from each node to each amenity")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        return fig

    return plot_amenity_distance_bars, plot_graph_with_amenities


@app.cell
def _(
    amenities,
    edge_lengths,
    feature_names,
    features,
    graph_toggle,
    mo,
    plot_graph_with_amenities,
):
    _graph_panel = mo.vstack(
        [
            graph_toggle,
            plot_graph_with_amenities(
                edge_lengths, amenities, features, feature_names, compact=True
            ) if graph_toggle.value else mo.Html(""),
        ],
        gap=0.35,
    ).style(
        {
            "position": "fixed",
            "top": "10px",
            "left": "16px",
            "z-index": "9998",
            "width": "min(660px, calc(100vw - 40px))" if graph_toggle.value else "auto",
            "max-height": "88vh",
            "overflow-y": "auto",
            "background": "rgba(255,255,255,0.98)",
            "padding": "8px 10px",
            "border": "1px solid #94a3b8",
            "border-radius": "12px",
            "box-shadow": "0 5px 15px rgba(15,23,42,0.28)",
            "color": "#0f172a",
        }
    )
    _graph_panel
    return


@app.cell
def _(
    amenity_distance_rows,
    amenities,
    close_step1,
    code_block,
    dataframe_html,
    derived_features,
    feature_mode,
    feature_names,
    feature_rule,
    features,
    get_step1,
    help_icon,
    mo,
    node_labels,
    open_step1,
    pd,
    plot_amenity_distance_bars,
    plot_graph_with_amenities,
    edge_lengths,
):
    if not get_step1():
        _step_output = mo.vstack([open_step1])
    else:
        _rule_labels = {
            "sum_inverse": "Sum of inverse distances over all amenities",
            "nearest_inverse": "Nearest amenity only",
            "mean_k_inverse": "Mean of k nearest inverse distances",
            "count_threshold": "Count within threshold",
        }
        _help = help_icon(
            "Nearest only uses one closest amenity. Sum inverse uses every amenity. Mean of k nearest uses only the k closest. Count within threshold counts reachable amenities and ignores exact distance once inside the threshold."
        )
        _distance_df = pd.DataFrame(amenity_distance_rows)
        _feature_df = pd.DataFrame(features, index=node_labels, columns=feature_names)
        _derived_df = pd.DataFrame(derived_features, index=node_labels, columns=feature_names)
        _amenity_rows = []
        for _amenity in amenities:
            _start, _end = _amenity["edge"]
            _edge_length = edge_lengths[tuple(sorted((_start, _end)))]
            _fraction = float(_amenity["fraction"])
            _distance_from_start = _fraction * _edge_length
            _distance_from_end = (1.0 - _fraction) * _edge_length
            _amenity_rows.append(
                {
                    "amenity": _amenity["name"],
                    "category": _amenity["category"],
                    "road segment": f"{node_labels[_start]}-{node_labels[_end]}",
                    "start node": node_labels[_start],
                    "end node": node_labels[_end],
                    "position along edge from start node": f"{100.0 * _fraction:.1f}% toward {node_labels[_end]}",
                    "distance from start node": _distance_from_start,
                    "distance from end node": _distance_from_end,
                }
            )
        _amenity_df = pd.DataFrame(_amenity_rows)

        _mode_text = (
            "Derived from amenity locations"
            if feature_mode.value == "derived"
            else "Manual continuous features"
        )
        _rule_text = _rule_labels.get(feature_rule.value, str(feature_rule.value))

        _formula = """
For an amenity located part-way along edge (u,v):

    u and v are the two road nodes at the ends of that edge.
    Example: for School 1 on edge A-C, u=A and v=C.

    distance(node, amenity)
      = min(
          shortest_path(node, u) + distance(u, amenity along edge),
          shortest_path(node, v) + distance(v, amenity along edge)
        )

The amenity itself lies on the road segment, so the only distance is distance through the street network plus the distance along that segment.

Default derived rule:

    feature(node, category)
      = sum(1 / distance(node, amenity_i))
        for every amenity_i in that category
"""

        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 1 - One shared graph and one shared feature matrix</h2></div>"),
                mo.md(
                    f"""
                    The two FEATHER variants receive the **same four road nodes A-D** and the **same continuous feature matrix X**.

                    Current feature mode: **{_mode_text}**  
                    Current derived rule: **{_rule_text}**

                    """
                ),
                plot_graph_with_amenities(edge_lengths, amenities, features, feature_names),
                mo.Html(
                    "<div class='uf-callout'><strong>How multiple amenities work in the default rule.</strong> "
                    "This example intentionally starts with two Schools and two Shops. With the default sum-of-inverse-distances rule, both Schools contribute to every node's School feature and both Shops contribute to every node's Shops feature. "
                    "If there were only one School, the sum would contain one term. Adding another nearby School adds another positive contribution; a very distant School adds only a small contribution because 1/d is small.</div>"
                ),
                mo.hstack(
                    [
                        mo.md("**Feature rule help**"),
                        _help,
                        mo.md("The selector is in the fixed Controls panel."),
                    ],
                    gap=0.5,
                ),
                mo.Html("<h3 style='color:#0f172a'>How network distance to an on-edge amenity is computed</h3>"),
                code_block(_formula),
                plot_amenity_distance_bars(amenity_distance_rows, node_labels),
                mo.Html("<h3 style='color:#0f172a'>Where each amenity sits on the road network</h3>"),
                mo.Html(
                    "<div class='uf-callout'><strong>How to read this table:</strong> "
                    "<em>road segment</em> is the edge containing the amenity. <em>start node</em> and <em>end node</em> define the direction used to describe where on that edge the amenity sits. "
                    "The percentage is used only to locate the amenity along that street. For example, if School 1 is on A-C and is 6.7% toward C, then "
                    "<code>distance from start node = 0.067 × length(A-C)</code>, while the remaining 93.3% gives the distance from C. "
                    "These two distances are used when calculating the shortest network distance from every node to that amenity. "
                    "The percentage is not a FEATHER parameter and does not directly enter either adjacency matrix.</div>"
                ),
                dataframe_html(_amenity_df, title="Amenity positions on road segments", precision=4),
                mo.Html("<h3 style='color:#0f172a'>Current amenity distances</h3>"),
                dataframe_html(_distance_df.set_index(["node", "amenity"]), title="Network distances", precision=4),
                mo.Html("<h3 style='color:#0f172a'>Feature matrix X sent to both algorithms</h3>"),
                mo.Html(
                    "<div class='uf-callout'><strong>Why node A has large School/Shops values:</strong> "
                    "with the default derived rule, these are <em>inverse-distance affinity scores</em>, not raw distances. Each amenity contributes <code>1 / distance</code>. "
                    "School 1 and Shop 1 are each only 0.20 network-distance units from A, so each nearby amenity alone contributes <code>1 / 0.20 = 5</code> to A before the second amenity in that category is added. "
                    "Therefore <strong>closer amenities make the feature value larger</strong>. This direction is the opposite of a raw distance score, where a smaller number would mean closer access.</div>"
                ),
                dataframe_html(_feature_df, title="Current shared feature matrix X", precision=4),
                mo.Html(
                    "<div class='uf-callout'><strong>Controlled comparison:</strong> even though UrbanFEATHER was also used with binary amenity features in the predecessor project, this workbook intentionally supplies the exact same continuous X to both algorithms. That isolates the effect of the transition weighting.</div>"
                ),
                close_step1,
            ],
            gap=0.7,
        )
    _step_output
    return


@app.cell
def _(
    A_plain,
    A_urban,
    callout,
    close_step2,
    compare_heatmaps,
    exact_code_details,
    get_step2,
    matrix_html,
    mo,
    node_labels,
    open_step2,
    show_a_exact,
    side_by_side,
    teaching_card,
):
    if not get_step2():
        _step_output = mo.vstack([open_step2])
    else:
        _plain_teaching = """
# FEATHER expects an unweighted graph.
# In this comparison the street-length attributes are removed
# before the graph reaches plain FEATHER.
A = adjacency_matrix(G_plain)
"""
        _plain_exact = """
A = nx.adjacency_matrix(
    graph,
    nodelist=range(graph.number_of_nodes())
)
"""
        _urban_teaching = """
G = graph.copy()
for each edge:
    strength = 1 / street_length
A = adjacency_matrix(G, weight="weight")
"""
        _urban_exact = """
G = graph.copy()
for u, v, d in G.edges(data=True):
    length = d.get("weight", 1.0)
    if length <= 0:
        length = 1e-9
    d["weight"] = 1.0 / length

A = nx.adjacency_matrix(
    G,
    nodelist=range(G.number_of_nodes()),
    weight='weight'
)
"""

        _plain_card = teaching_card(
            "FEATHER - adjacency A",
            _plain_teaching,
            matrix_html(A_plain, node_labels, node_labels, title="FEATHER A"),
            "A connected pair gets value 1 and an unconnected pair gets 0. The real-world length labels are not part of the original unweighted FEATHER input representation.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER code", _plain_exact),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - weighted adjacency A",
            _urban_teaching,
            matrix_html(A_urban, node_labels, node_labels, title="UrbanFEATHER A after 1/length"),
            "UrbanFEATHER first turns each length into a connection strength. Shorter edges therefore have larger entries in A.",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER code", _urban_exact),
        )

        _fig = compare_heatmaps(
            A_plain,
            A_urban,
            node_labels,
            node_labels,
            "FEATHER A",
            "UrbanFEATHER weighted A",
            "A difference: Urban - Plain",
        )

        _exact = (
            mo.vstack(
                [
                    side_by_side(
                        matrix_html(A_plain, node_labels, node_labels, title="FEATHER A exact"),
                        matrix_html(A_urban, node_labels, node_labels, title="UrbanFEATHER A exact"),
                    ),
                    matrix_html(A_urban - A_plain, node_labels, node_labels, title="Urban - Plain"),
                ]
            )
            if show_a_exact.value
            else mo.Html("")
        )

        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 2 - Build the adjacency matrix A</h2></div>"),
                mo.md(
                    "Both algorithms start from the same road topology, but their **algorithm input representation is different**. "
                    "FEATHER was designed for an unweighted graph, so this controlled comparison gives it a copy without the street-length attributes. "
                    "UrbanFEATHER keeps those lengths and transforms them into inverse-length strengths."
                ),
                side_by_side(_plain_card, _urban_card),
                callout(
                    "<strong>Important edge case in the UrbanFEATHER code:</strong> the guard <code>if length &lt;= 0: length = 1e-9</code> is applied only while iterating over <em>existing edges</em>. "
                    "It prevents division by zero or a negative length. The next line then computes <code>1 / length</code>, so an invalid zero-length edge becomes a very large strength of <code>1e9</code>; it does <strong>not</strong> become a weight of <code>1e-9</code>. "
                    "The diagonal of the adjacency matrix still stays 0 because A-A, B-B, C-C and D-D are not edges in this graph. NetworkX only puts a nonzero diagonal entry there when the graph actually contains a self-loop. "
                    "Likewise, missing connections stay 0: the guard changes bad lengths on existing edges; it does not create new edges or fill zero cells in the matrix. "
                    "For a later production refactor, rejecting or explicitly handling non-positive street lengths would be safer than silently turning them into an overwhelmingly strong connection.",
                    kind="note",
                ),
                callout(
                    "<strong>First divergence:</strong> this is the first place where the numeric matrices differ. "
                    "The later cosine/sine code is almost the same; the different transition matrix is what causes different propagated embeddings.",
                    kind="diverge",
                ),
                _fig,
                close_step2,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    D_plain,
    D_urban,
    callout,
    close_step3,
    compare_heatmaps,
    exact_code_details,
    get_step3,
    matrix_html,
    mo,
    node_labels,
    open_step3,
    show_d_exact,
    side_by_side,
    teaching_card,
):
    if not get_step3():
        _step_output = mo.vstack([open_step3])
    else:
        _plain_teaching = """
degree[node] = number of connected neighbours
D = diag(degree[A], degree[B], degree[C], degree[D])
"""
        _urban_teaching = """
strength_sum[node] = sum(1 / edge_length)
D = diag(strength_sum[A], strength_sum[B], ...)
"""
        _plain_exact = """
# D is not explicitly stored in FEATHER.
# Its diagonal is implied by graph.degree[node].
values = np.array([
    1.0 / graph.degree[node]
    for node in range(graph.number_of_nodes())
])
"""
        _urban_exact = """
# D is not explicitly stored in UrbanFEATHER either.
# Its diagonal is implied by the weighted degree sum.
weights_sum = np.array([
    graph.degree(node, weight='weight')
    for node in range(graph.number_of_nodes())
], dtype=float)
"""

        _plain_card = teaching_card(
            "FEATHER - D contains node degree",
            _plain_teaching,
            matrix_html(D_plain, node_labels, node_labels, title="FEATHER D"),
            "The diagonal entry for a node is simply how many neighbours it has. Edge length does not affect D.",
            accent="#0369a1",
            exact=exact_code_details("Relevant FEATHER code", _plain_exact),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - D contains total outgoing strength",
            _urban_teaching,
            matrix_html(D_urban, node_labels, node_labels, title="UrbanFEATHER D"),
            "The diagonal entry is the sum of the transformed inverse-length strengths leaving that node.",
            accent="#7c3aed",
            exact=exact_code_details("Relevant UrbanFEATHER code", _urban_exact),
        )

        _fig = compare_heatmaps(
            D_plain,
            D_urban,
            node_labels,
            node_labels,
            "FEATHER D",
            "UrbanFEATHER D",
            "D difference: Urban - Plain",
        )
        _exact = (
            mo.vstack(
                [
                    side_by_side(
                        matrix_html(D_plain, node_labels, node_labels, title="FEATHER D exact"),
                        matrix_html(D_urban, node_labels, node_labels, title="UrbanFEATHER D exact"),
                    ),
                    matrix_html(D_urban - D_plain, node_labels, node_labels, title="Urban - Plain"),
                ]
            )
            if show_d_exact.value
            else mo.Html("")
        )

        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 3 - Understand the degree/strength matrix D</h2></div>"),
                mo.md("Neither implementation explicitly materializes D, but both compute exactly the values needed for its inverse. Showing D first makes the normalization easier to understand."),
                side_by_side(_plain_card, _urban_card),
                callout("FEATHER D counts neighbours. UrbanFEATHER D sums inverse-length connection strengths."),
                _fig,
                close_step3,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    Dinv_plain,
    Dinv_urban,
    close_step4,
    compare_heatmaps,
    exact_code_details,
    get_step4,
    matrix_html,
    mo,
    node_labels,
    open_step4,
    selected_node_index,
    show_dinv_exact,
    side_by_side,
    teaching_card,
):
    if not get_step4():
        _step_output = mo.vstack([open_step4])
    else:
        _plain_exact = """
index = np.arange(graph.number_of_nodes())
values = np.array([
    1.0/graph.degree[node]
    for node in range(graph.number_of_nodes())
])
shape = (graph.number_of_nodes(), graph.number_of_nodes())
D_inverse = sparse.coo_matrix(
    (values, (index, index)), shape=shape
)
"""
        _urban_exact = """
index = np.arange(graph.number_of_nodes())
weights_sum = np.array([
    graph.degree(node, weight='weight')
    for node in range(graph.number_of_nodes())
], dtype=float)
weights_sum[weights_sum == 0] = 1.0
values = 1.0 / weights_sum
shape = (graph.number_of_nodes(), graph.number_of_nodes())
D_inverse = sparse.coo_matrix(
    (values, (index, index)), shape=shape
)
"""
        _plain_card = teaching_card(
            "FEATHER - D inverse",
            "D_inverse[node,node] = 1 / number_of_neighbours",
            matrix_html(Dinv_plain, node_labels, node_labels, title="FEATHER D^-1"),
            f"For the currently followed node {node_labels[selected_node_index]}, this is 1 divided by its ordinary graph degree.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER _create_D_inverse", _plain_exact),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - weighted D inverse",
            "D_inverse[node,node] = 1 / sum(inverse_length_strengths)",
            matrix_html(Dinv_urban, node_labels, node_labels, title="UrbanFEATHER D^-1"),
            f"For node {node_labels[selected_node_index]}, this is the reciprocal of the total inverse-length strength leaving that node.",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER _create_D_inverse", _urban_exact),
        )
        _fig = compare_heatmaps(
            Dinv_plain,
            Dinv_urban,
            node_labels,
            node_labels,
            "FEATHER D inverse",
            "UrbanFEATHER D inverse",
            "D inverse difference",
        )
        _exact = (
            mo.vstack(
                [
                    side_by_side(
                        matrix_html(Dinv_plain, node_labels, node_labels, title="FEATHER D inverse exact"),
                        matrix_html(Dinv_urban, node_labels, node_labels, title="UrbanFEATHER D inverse exact"),
                    ),
                    matrix_html(Dinv_urban - Dinv_plain, node_labels, node_labels, title="Urban - Plain"),
                ]
            )
            if show_dinv_exact.value
            else mo.Html("")
        )
        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 4 - Create D inverse</h2></div>"),
                side_by_side(_plain_card, _urban_card),
                _fig,
                close_step4,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    Atilde_plain,
    Atilde_urban,
    close_step5,
    compare_heatmaps,
    exact_code_details,
    get_step5,
    matrix_html,
    mo,
    node_labels,
    open_step5,
    selected_node_index,
    show_atilde_exact,
    side_by_side,
    teaching_card,
):
    if not get_step5():
        _step_output = mo.vstack([open_step5])
    else:
        _plain_exact = """
A = nx.adjacency_matrix(
    graph, nodelist=range(graph.number_of_nodes())
)
D_inverse = self._create_D_inverse(graph)
A_tilde = D_inverse.dot(A)
"""
        _urban_exact = """
A = nx.adjacency_matrix(
    G,
    nodelist=range(G.number_of_nodes()),
    weight='weight'
)
D_inverse = self._create_D_inverse(G)
A_tilde = D_inverse.dot(A)
"""
        _plain_card = teaching_card(
            "FEATHER - A-tilde",
            "A_tilde = D_inverse @ A",
            matrix_html(Atilde_plain, node_labels, node_labels, title="FEATHER A-tilde"),
            f"Row {node_labels[selected_node_index]} distributes probability equally among its connected neighbours because each neighbour had equal edge strength in the unweighted graph.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER _create_A_tilde", _plain_exact),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - weighted A-tilde",
            "A_tilde = weighted_D_inverse @ weighted_A",
            matrix_html(Atilde_urban, node_labels, node_labels, title="UrbanFEATHER A-tilde"),
            f"Row {node_labels[selected_node_index]} distributes probability according to each inverse-length strength as a share of that node's total outgoing strength.",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER _create_A_tilde", _urban_exact),
        )
        _fig = compare_heatmaps(
            Atilde_plain,
            Atilde_urban,
            node_labels,
            node_labels,
            "FEATHER A-tilde",
            "UrbanFEATHER A-tilde",
            "Transition probability difference",
        )
        _exact = (
            mo.vstack(
                [
                    side_by_side(
                        matrix_html(Atilde_plain, node_labels, node_labels, title="FEATHER A-tilde exact"),
                        matrix_html(Atilde_urban, node_labels, node_labels, title="UrbanFEATHER A-tilde exact"),
                    ),
                    matrix_html(Atilde_urban - Atilde_plain, node_labels, node_labels, title="Urban - Plain"),
                ]
            )
            if show_atilde_exact.value
            else mo.Html("")
        )
        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 5 - Normalize into A-tilde: the transition/mixing matrix</h2></div>"),
                side_by_side(_plain_card, _urban_card),
                mo.Html(
                    "<div class='uf-diverge'><strong>This is the key mathematical difference.</strong> From this point onward, both implementations multiply their characteristic-function values by A-tilde in the same basic way. Because A-tilde differs, the propagated values differ.</div>"
                ),
                _fig,
                close_step5,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    X_theta,
    close_step6,
    compare_heatmaps,
    exact_code_details,
    feature_names,
    get_step6,
    matrix_html,
    mo,
    node_labels,
    open_step6,
    show_outer_exact,
    side_by_side,
    teaching_card,
    theta,
):
    if not get_step6():
        _step_output = mo.vstack([open_step6])
    else:
        _plain_exact = """
theta = np.linspace(
    0.01,
    self.theta_max,
    self.eval_points
)
X = np.outer(X, theta)
X = X.reshape(graph.number_of_nodes(), -1)
"""
        _urban_exact = """
theta = np.linspace(
    0.01,
    args.theta_max,
    args.eval_points
)
X = np.outer(X, theta)
X = X.reshape(graph.number_of_nodes(), -1)
"""
        _column_labels = []
        for _feature in feature_names:
            for _i in range(len(theta)):
                _column_labels.append(f"{_feature}*t{_i + 1}")

        _plain_card = teaching_card(
            "FEATHER - theta and feature x theta",
            "theta = linspace(0.01, theta_max, eval_points)\nX_theta = outer(X, theta).reshape(num_nodes, -1)",
            matrix_html(X_theta, node_labels, _column_labels, title="X multiplied by theta"),
            "FEATHER samples the continuous node features at the chosen theta evaluation points.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER fit code", _plain_exact),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - theta and feature x theta",
            "theta = linspace(0.01, theta_max, eval_points)\nX_theta = outer(X, theta).reshape(num_nodes, -1)",
            matrix_html(X_theta, node_labels, _column_labels, title="Same X multiplied by same theta"),
            "The mathematics is the same here. The code-level difference is that this UrbanFEATHER copy reads theta_max/eval_points/order from the args object passed to fit().",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER fit code", _urban_exact),
        )
        _fig = compare_heatmaps(
            X_theta,
            X_theta,
            node_labels,
            _column_labels,
            "FEATHER X * theta",
            "Urban X * theta",
            "Difference (should be zero)",
        )
        _exact = (
            side_by_side(
                matrix_html(X_theta, node_labels, _column_labels, title="FEATHER exact"),
                matrix_html(X_theta, node_labels, _column_labels, title="Urban exact"),
            )
            if show_outer_exact.value
            else mo.Html("")
        )
        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 6 - Create theta and multiply every feature by theta</h2></div>"),
                mo.md(f"Current theta values: **{[round(float(_theta_value), 6) for _theta_value in theta]}**"),
                side_by_side(_plain_card, _urban_card),
                mo.Html("<div class='uf-callout'><strong>No mathematical difference here.</strong> Because this experiment feeds both algorithms the same X and the same theta parameters, this intermediate matrix is identical.</div>"),
                _fig,
                close_step6,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    X0,
    close_step7,
    compare_heatmaps,
    exact_code_details,
    get_step7,
    matrix_html,
    mo,
    node_labels,
    open_step7,
    show_x0_exact,
    side_by_side,
    teaching_card,
    x0_columns,
):
    if not get_step7():
        _step_output = mo.vstack([open_step7])
    else:
        _exact_code = """
X = np.concatenate([
    np.cos(X),
    np.sin(X)
], axis=1)
"""
        _plain_card = teaching_card(
            "FEATHER - X0 characteristic-function values",
            "X0 = concatenate(cos(X_theta), sin(X_theta))",
            matrix_html(X0, node_labels, x0_columns, title="FEATHER X0"),
            "Each node now has cosine (real) and sine (imaginary) values for School and Shops at every theta evaluation point.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER code", _exact_code),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - X0 characteristic-function values",
            "X0 = concatenate(cos(X_theta), sin(X_theta))",
            matrix_html(X0, node_labels, x0_columns, title="UrbanFEATHER X0"),
            "Because both algorithms received the same continuous X and theta, X0 is still exactly the same before graph propagation begins.",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER code", _exact_code),
        )
        _fig = compare_heatmaps(
            X0,
            X0,
            node_labels,
            x0_columns,
            "FEATHER X0",
            "Urban X0",
            "Difference (zero before propagation)",
        )
        _exact = (
            side_by_side(
                matrix_html(X0, node_labels, x0_columns, title="FEATHER X0 exact"),
                matrix_html(X0, node_labels, x0_columns, title="UrbanFEATHER X0 exact"),
            )
            if show_x0_exact.value
            else mo.Html("")
        )
        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 7 - Convert to cosine and sine: X0</h2></div>"),
                side_by_side(_plain_card, _urban_card),
                mo.Html("<div class='uf-callout'><strong>Still identical.</strong> The algorithms only start producing different node embeddings when their different A-tilde matrices are used for propagation.</div>"),
                _fig,
                close_step7,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    Atilde_plain,
    Atilde_urban,
    X0,
    actual_order,
    close_step8,
    compare_heatmaps,
    dataframe_html,
    exact_code_details,
    get_step8,
    inspect_stage,
    matrix_html,
    mo,
    node_labels,
    np,
    open_step8,
    pd,
    plain_orders,
    selected_node_index,
    show_prop_exact,
    side_by_side,
    teaching_card,
    urban_orders,
    x0_columns,
):
    if not get_step8():
        _step_output = mo.vstack([open_step8])
    else:
        _stage = min(int(inspect_stage.value), actual_order)
        _plain_current = plain_orders[_stage - 1]
        _urban_current = urban_orders[_stage - 1]
        _plain_previous = X0 if _stage == 1 else plain_orders[_stage - 2]
        _urban_previous = X0 if _stage == 1 else urban_orders[_stage - 2]
        _component_index = 0
        _component_name = x0_columns[_component_index]

        _plain_rows = []
        _urban_rows = []
        for source in range(4):
            p_weight = float(Atilde_plain[selected_node_index, source])
            u_weight = float(Atilde_urban[selected_node_index, source])
            if p_weight > 0:
                previous = float(_plain_previous[source, _component_index])
                _plain_rows.append(
                    {
                        "source": node_labels[source],
                        "transition_share": p_weight,
                        "previous_value": previous,
                        "contribution": p_weight * previous,
                    }
                )
            if u_weight > 0:
                previous = float(_urban_previous[source, _component_index])
                _urban_rows.append(
                    {
                        "source": node_labels[source],
                        "transition_share": u_weight,
                        "previous_value": previous,
                        "contribution": u_weight * previous,
                    }
                )

        _plain_contrib = pd.DataFrame(_plain_rows).set_index("source")
        _urban_contrib = pd.DataFrame(_urban_rows).set_index("source")
        _plain_sum = float(_plain_contrib["contribution"].sum())
        _urban_sum = float(_urban_contrib["contribution"].sum())
        _plain_actual = float(_plain_current[selected_node_index, _component_index])
        _urban_actual = float(_urban_current[selected_node_index, _component_index])

        _loop_code = """
feature_blocks = []
for _ in range(order):
    X = A_tilde.dot(X)
    feature_blocks.append(X)
"""
        _urban_exact = """
feature_blocks = []
for _ in range(args.order):
    X = A_tilde.dot(X)
    feature_blocks.append(X)
"""
        _plain_exact = """
feature_blocks = []
for _ in range(self.order):
    X = A_tilde.dot(X)
    feature_blocks.append(X)
"""

        _plain_card = teaching_card(
            f"FEATHER - propagation order {_stage}",
            _loop_code,
            matrix_html(_plain_current, node_labels, x0_columns, title=f"FEATHER X{_stage}"),
            "Each row is a weighted mixture of the previous-step rows. In plain FEATHER the mixture weights come from ordinary graph degree normalization.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER loop", _plain_exact),
        )
        _urban_card = teaching_card(
            f"UrbanFEATHER - propagation order {_stage}",
            _loop_code,
            matrix_html(_urban_current, node_labels, x0_columns, title=f"Urban X{_stage}"),
            "The same matrix multiplication is performed, but the transition shares came from inverse street lengths, so the mixture is different.",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER loop", _urban_exact),
        )

        _fig = compare_heatmaps(
            _plain_current,
            _urban_current,
            node_labels,
            x0_columns,
            f"FEATHER order {_stage}",
            f"Urban order {_stage}",
            f"Order {_stage} difference",
        )

        _plain_power = np.linalg.matrix_power(Atilde_plain, _stage)
        _urban_power = np.linalg.matrix_power(Atilde_urban, _stage)
        _power_fig = compare_heatmaps(
            _plain_power,
            _urban_power,
            node_labels,
            node_labels,
            f"FEATHER A-tilde^{_stage}",
            f"Urban A-tilde^{_stage}",
            "Source-probability difference",
        )

        _exact = (
            side_by_side(
                matrix_html(_plain_current, node_labels, x0_columns, title=f"FEATHER X{_stage} exact"),
                matrix_html(_urban_current, node_labels, x0_columns, title=f"UrbanFEATHER X{_stage} exact"),
            )
            if show_prop_exact.value
            else mo.Html("")
        )

        _microscope = mo.vstack(
            [
                mo.Html(
                    f"<div class='uf-card'><h3>Matrix-multiplication microscope</h3>"
                    f"<p>Followed node: <strong>{node_labels[selected_node_index]}</strong>. "
                    f"Component: <strong>{_component_name}</strong>. Stage: <strong>{_stage}</strong>.</p>"
                    f"<p>The row calculation is: X{_stage}[node] = sum_j A_tilde[node,j] * X{_stage - 1}[j].</p></div>"
                ),
                side_by_side(
                    mo.vstack(
                        [
                            mo.Html("<h4 style='color:#0f172a'>FEATHER contributions</h4>"),
                            dataframe_html(_plain_contrib, title="FEATHER row terms", precision=6),
                            mo.md(f"Sum of terms: **{_plain_sum:.8f}**  | matrix result: **{_plain_actual:.8f}**"),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.Html("<h4 style='color:#0f172a'>Urban contributions</h4>"),
                            dataframe_html(_urban_contrib, title="Urban row terms", precision=6),
                            mo.md(f"Sum of terms: **{_urban_sum:.8f}**  | matrix result: **{_urban_actual:.8f}**"),
                        ]
                    ),
                ),
            ]
        )

        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 8 - Propagate through the graph order by order</h2></div>"),
                mo.md(
                    f"The Controls panel currently asks to inspect stage **{int(inspect_stage.value)}**. "
                    f"The FEATHER order is **{actual_order}**, so this section displays stage **{_stage}**."
                ),
                side_by_side(_plain_card, _urban_card),
                _fig,
                _microscope,
                mo.Html("<h3 style='color:#0f172a'>Where can the information originate after exactly this many steps?</h3>"),
                mo.md("A-tilde raised to the selected power summarizes the probability mass of all walks of exactly that length, including walks that revisit earlier nodes."),
                _power_fig,
                close_step8,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(
    PlainFEATHERReference,
    SimpleNamespace,
    UrbanFEATHER,
    actual_order,
    close_step9,
    compare_heatmaps,
    embedding_columns,
    eval_points,
    exact_code_details,
    features,
    get_step9,
    graph_lengths,
    graph_plain,
    matrix_html,
    mo,
    node_labels,
    np,
    open_step9,
    plain_embedding,
    show_embedding_exact,
    side_by_side,
    teaching_card,
    theta_max,
    urban_embedding,
):
    if not get_step9():
        _step_output = mo.vstack([open_step9])
    else:
        _plain_model = PlainFEATHERReference(
            theta_max=float(theta_max.value),
            eval_points=int(eval_points.value),
            order=int(actual_order),
        )
        _plain_model.fit(graph_plain, features.copy())
        _plain_repository_embedding = _plain_model.get_embedding()
        _plain_match = bool(np.allclose(_plain_repository_embedding, plain_embedding))

        _urban_match = None
        _urban_repository_embedding = None
        _urban_error = None
        if UrbanFEATHER is not None:
            try:
                _urban_model = UrbanFEATHER()
                _args = SimpleNamespace(
                    theta_max=float(theta_max.value),
                    eval_points=int(eval_points.value),
                    order=int(actual_order),
                )
                _urban_model.fit(graph_lengths, features.copy(), _args)
                _urban_repository_embedding = _urban_model.get_embedding()
                _urban_match = bool(np.allclose(_urban_repository_embedding, urban_embedding))
            except Exception as exc:
                _urban_error = str(exc)

        _plain_fit = """
def fit(self, graph, X):
    theta = np.linspace(0.01, self.theta_max, self.eval_points)
    A_tilde = self._create_A_tilde(graph)
    X = np.outer(X, theta)
    X = X.reshape(graph.number_of_nodes(), -1)
    X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
    feature_blocks = []
    for _ in range(self.order):
        X = A_tilde.dot(X)
        feature_blocks.append(X)
    self._X = np.concatenate(feature_blocks, axis=1)
"""
        _urban_fit = """
def fit(self, graph, X, args):
    theta = np.linspace(0.01, args.theta_max, args.eval_points)
    A_tilde = self._create_A_tilde(graph)
    X = np.outer(X, theta)
    X = X.reshape(graph.number_of_nodes(), -1)
    X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
    feature_blocks = []
    for _ in range(args.order):
        X = A_tilde.dot(X)
        feature_blocks.append(X)
    self._X = np.concatenate(feature_blocks, axis=1)
"""

        _plain_card = teaching_card(
            "FEATHER - final node embedding",
            "embedding = concatenate(X1, X2, ..., X_order, axis=1)",
            matrix_html(plain_embedding, node_labels, embedding_columns, title="FEATHER final embedding", precision=4),
            "Every node receives one long vector containing its characteristic-function representation after every requested propagation order.",
            accent="#0369a1",
            exact=exact_code_details("Actual FEATHER fit method", _plain_fit),
        )
        _urban_card = teaching_card(
            "UrbanFEATHER - final node embedding",
            "embedding = concatenate(X1, X2, ..., X_order, axis=1)",
            matrix_html(urban_embedding, node_labels, embedding_columns, title="UrbanFEATHER final embedding", precision=4),
            "The output has the same shape and layout, but its numbers differ because every propagation step used the weighted transition matrix.",
            accent="#7c3aed",
            exact=exact_code_details("Actual UrbanFEATHER fit method", _urban_fit),
        )

        _fig = compare_heatmaps(
            plain_embedding,
            urban_embedding,
            node_labels,
            embedding_columns,
            "FEATHER final embedding",
            "UrbanFEATHER final embedding",
            "Final embedding difference",
        )
        _exact = (
            side_by_side(
                matrix_html(plain_embedding, node_labels, embedding_columns, title="FEATHER exact", precision=6),
                matrix_html(urban_embedding, node_labels, embedding_columns, title="Urban exact", precision=6),
            )
            if show_embedding_exact.value
            else mo.Html("")
        )

        _values_per_node = features.shape[1] * int(eval_points.value) * 2 * actual_order
        _urban_status = (
            f"UrbanFEATHER repository check: {_urban_match}"
            if _urban_error is None and _urban_match is not None
            else f"UrbanFEATHER repository check could not run: {_urban_error or 'class not imported'}"
        )

        _step_output = mo.vstack(
            [
                mo.Html("<div class='uf-card'><h2>Step 9 - Concatenate all orders into the final node embedding</h2></div>"),
                side_by_side(_plain_card, _urban_card),
                mo.md(
                    f"With **{features.shape[1]} features x {int(eval_points.value)} theta points x 2 (cos/sin) x {actual_order} orders**, "
                    f"each node receives **{_values_per_node} values**. The final shape is **{plain_embedding.shape}** on both sides."
                ),
                _fig,
                mo.Html(
                    f"<div class='uf-card'><h3>Verification against the supplied implementations</h3>"
                    f"<p><strong>FEATHER reference check:</strong> {_plain_match}</p>"
                    f"<p><strong>{_urban_status}</strong></p>"
                    f"<p>The manual matrices in this workbook are therefore not a separate algorithm; they are an expanded explanation of the supplied code.</p></div>"
                ),
                mo.Html(
                    "<div class='uf-diverge'><strong>Code-level summary:</strong> the FEATHER uses ordinary degree normalization and stores theta/order on self. "
                    "UrbanFEATHER changes the graph normalization to inverse-length weighted normalization and changes fit(graph, X) to fit(graph, X, args). "
                    "After A-tilde has been constructed, the outer-product, cos/sin, repeated matrix multiplication, and concatenation steps are otherwise structurally the same.</div>"
                ),
                close_step9,
            ],
            gap=0.7,
        )
    _step_output
    return

@app.cell
def _(mo):
    mo.Html(
        """
        <div class='uf-card'>
          <h2>Suggested experiments</h2>
          <ol>
            <li>Set every street length equal. The two A-tilde matrices should become much closer, and for this topology they should match when all strengths are equal.</li>
            <li>Change only B-C from 5 to 1. FEATHER stays unchanged because its graph is unweighted; UrbanFEATHER changes.</li>
            <li>Switch the feature rule from Sum of inverse distances to Nearest amenity only. X changes for both algorithms in exactly the same way; the FEATHER-vs-UrbanFEATHER graph-weighting difference remains separate.</li>
            <li>Increase order from 1 to 3 and watch how a small A-tilde difference compounds through repeated propagation.</li>
            <li>Switch to Manual continuous features to isolate FEATHER math from the amenity-to-feature construction completely.</li>
          </ol>
        </div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
