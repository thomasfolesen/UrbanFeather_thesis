import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATHER_SRC = PROJECT_ROOT / "src" / "Utils" / "FEATHER" / "src"

sys.path.insert(0, str(FEATHER_SRC))

from feather import FEATHER


NODE_LABELS = ["A", "B", "C", "D"]


def build_teaching_graph():
    """Return the same 4-node teaching graph used in the marimo workbook.

    Edges / lengths:

        A-B = 1
        A-C = 3
        B-C = 5
        B-D = 2
        C-D = 1

    The B-C diagonal edge is intentionally included.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(4))

    edge_lengths = {
        (0, 1): 1.0,
        (0, 2): 3.0,
        (1, 2): 5.0,
        (1, 3): 2.0,
        (2, 3): 1.0,
    }

    for (u, v), length in edge_lengths.items():
        graph.add_edge(u, v, weight=length)

    return graph, edge_lengths


def build_teaching_features(graph, edge_lengths):
    """Build the School/Shops raw X used by the teaching example.

    Amenities sit on graph edges, matching the current marimo workbook:

    - School 1: A-C, 0.35 network units from A
    - School 2: 62% of the way from C to D
    - Shop 1: A-B, 0.35 network units from A
    - Shop 2: 58% of the way from B to D

    The feature rule is the workbook default:

        feature(node, category) = sum(1 / network_distance_to_amenity)

    Therefore larger X values mean better/closer accessibility, which is the
    direction required by UrbanFEATHER_edgeCost_accessScore.
    """
    amenities = [
        {
            "category": "School",
            "edge": (0, 2),
            "fraction": 0.35 / edge_lengths[(0, 2)],
        },
        {
            "category": "School",
            "edge": (2, 3),
            "fraction": 0.62,
        },
        {
            "category": "Shops",
            "edge": (0, 1),
            "fraction": 0.35 / edge_lengths[(0, 1)],
        },
        {
            "category": "Shops",
            "edge": (1, 3),
            "fraction": 0.58,
        },
    ]

    all_pairs = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    category_index = {"School": 0, "Shops": 1}
    distances = [[[], []] for _ in range(4)]

    for amenity in amenities:
        u, v = amenity["edge"]
        key = tuple(sorted((u, v)))
        length = edge_lengths[key]
        from_u = amenity["fraction"] * length
        from_v = (1.0 - amenity["fraction"]) * length
        feature_index = category_index[amenity["category"]]

        for node in range(4):
            distance = min(
                all_pairs[node][u] + from_u,
                all_pairs[node][v] + from_v,
            )
            distances[node][feature_index].append(distance)

    features = np.zeros((4, 2), dtype=float)
    for node in range(4):
        for feature_index in range(2):
            features[node, feature_index] = sum(
                1.0 / distance for distance in distances[node][feature_index]
            )

    return features


def expected_feather_transition():
    """Exact topology-only FEATHER probabilities for the teaching graph."""
    return np.array(
        [
            [0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0],
            [1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0],
            [0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0],
        ],
        dtype=float,
    )


def expected_urbanfeather_transition():
    """Exact inverse-edge-length probabilities for the teaching graph."""
    # Raw inverse-length strengths:
    # A-B=1, A-C=1/3, B-C=1/5, B-D=1/2, C-D=1
    return np.array(
        [
            [0.0, 3.0 / 4.0, 1.0 / 4.0, 0.0],
            [10.0 / 17.0, 0.0, 2.0 / 17.0, 5.0 / 17.0],
            [5.0 / 23.0, 3.0 / 23.0, 0.0, 15.0 / 23.0],
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0],
        ],
        dtype=float,
    )


def expected_access_transition(graph, features, edge_transition):
    """Independent calculation of destination-accessibility probabilities."""
    # IMPORTANT modelling choice: one destination accessibility score is the
    # MEAN of the raw feature vector for that node.
    access_scores = features.mean(axis=1)
    expected = np.zeros((4, 4), dtype=float)

    for source in range(4):
        neighbours = list(graph.neighbors(source))
        total = sum(access_scores[destination] for destination in neighbours)

        if total == 0:
            expected[source] = edge_transition[source]
            continue

        for destination in neighbours:
            expected[source, destination] = access_scores[destination] / total

    return expected, access_scores


def assert_probability_rows(matrix, label):
    row_sums = matrix.sum(axis=1)
    nonempty_rows = np.any(matrix != 0, axis=1)
    assert np.allclose(row_sums[nonempty_rows], 1.0), (
        f"{label}: non-empty transition rows must sum to 1, got {row_sums}"
    )


def run_mode(graph, features, weighting_mode, edge_alpha=0.5, access_beta=0.5):
    args = SimpleNamespace(
        theta_max=2.0,
        eval_points=3,
        order=2,
        weighting_mode=weighting_mode,
        # IMPORTANT: alpha + beta MUST equal 1.0.
        edge_alpha=edge_alpha,
        access_beta=access_beta,
    )

    model = FEATHER()
    model.fit(graph, features.copy(), args)

    return (
        model.get_transition_matrix().toarray(),
        model.get_embedding(),
        model.get_transition_components(),
    )


def main():
    graph, edge_lengths = build_teaching_graph()
    features = build_teaching_features(graph, edge_lengths)

    # ------------------------------------------------------------------
    # 1. FEATHER: topology only
    # ------------------------------------------------------------------
    feather_A, feather_embedding, _ = run_mode(
        graph,
        features,
        "FEATHER",
    )
    expected_feather = expected_feather_transition()

    # assert stops the test if the condition is False. np.allclose checks that
    # the calculated matrix matches the expected matrix while allowing tiny
    # floating-point rounding differences.
    assert np.allclose(feather_A, expected_feather)
    assert_probability_rows(feather_A, "FEATHER")

    # ------------------------------------------------------------------
    # 2. UrbanFEATHER: inverse edge length only
    # ------------------------------------------------------------------
    urban_A, urban_embedding, _ = run_mode(
        graph,
        features,
        "UrbanFEATHER",
    )
    expected_urban = expected_urbanfeather_transition()

    # assert stops the test if the condition is False. np.allclose checks that
    # the calculated matrix matches the expected matrix while allowing tiny
    # floating-point rounding differences.
    assert np.allclose(urban_A, expected_urban)
    assert_probability_rows(urban_A, "UrbanFEATHER")

    # ------------------------------------------------------------------
    # 3. UrbanFEATHER_edgeCost_accessScore
    # ------------------------------------------------------------------
    edge_alpha = 0.5
    access_beta = 0.5
    # IMPORTANT: edge_alpha + access_beta MUST equal 1.0.

    proposed_A, proposed_embedding, proposed_components = run_mode(
        graph,
        features,
        "UrbanFEATHER_edgeCost_accessScore",
        edge_alpha=edge_alpha,
        access_beta=access_beta,
    )

    expected_access, expected_access_scores = expected_access_transition(
        graph,
        features,
        expected_urban,
    )
    expected_proposed = (
        edge_alpha * expected_urban + access_beta * expected_access
    )

    assert np.allclose(
        proposed_components["edge_share"].toarray(),
        expected_urban,
    )
    assert np.allclose(
        proposed_components["access_share"].toarray(),
        expected_access,
    )
    assert np.allclose(
        proposed_components["access_scores"],
        expected_access_scores,
    )
    # assert stops the test if the condition is False. np.allclose checks that
    # the calculated matrix matches the expected matrix while allowing tiny
    # floating-point rounding differences.
    assert np.allclose(proposed_A, expected_proposed)
    assert_probability_rows(proposed_A, "UrbanFEATHER_edgeCost_accessScore")

    # 2 raw features x 3 theta evaluation points x 2 (cos/sin) x 2 orders
    expected_embedding_shape = (4, 24)

    for label, embedding in [
        ("FEATHER", feather_embedding),
        ("UrbanFEATHER", urban_embedding),
        ("UrbanFEATHER_edgeCost_accessScore", proposed_embedding),
    ]:
        assert embedding.shape == expected_embedding_shape, (
            f"{label}: expected {expected_embedding_shape}, got {embedding.shape}"
        )
        assert np.isfinite(embedding).all(), f"{label}: embedding is not finite"

    # The three methods should not accidentally collapse to the same transition
    # matrix on this deliberately asymmetric teaching graph.
    assert not np.allclose(feather_A, urban_A)
    assert not np.allclose(urban_A, proposed_A)

    # Backwards-compatible default: if weighting_mode is omitted, the class
    # must preserve current UrbanFEATHER behaviour.
    legacy_args = SimpleNamespace(
        theta_max=2.0,
        eval_points=3,
        order=2,
    )
    legacy_model = FEATHER()
    legacy_model.fit(graph, features, legacy_args)
    assert np.allclose(
        legacy_model.get_transition_matrix().toarray(),
        expected_urban,
    )

    # If every neighbouring destination has accessibility score 0, the
    # accessibility component falls back to the UrbanFEATHER edge probabilities.
    zero_features = np.zeros_like(features)
    zero_A, _, zero_components = run_mode(
        graph,
        zero_features,
        "UrbanFEATHER_edgeCost_accessScore",
        edge_alpha=0.5,
        access_beta=0.5,
    )
    assert np.allclose(zero_components["access_share"].toarray(), expected_urban)
    assert np.allclose(zero_A, expected_urban)

    # Validation checks for the experimental method.
    invalid_alpha_beta_args = SimpleNamespace(
        theta_max=2.0,
        eval_points=3,
        order=2,
        weighting_mode="UrbanFEATHER_edgeCost_accessScore",
        edge_alpha=0.7,
        access_beta=0.7,
    )
    try:
        FEATHER().fit(graph, features, invalid_alpha_beta_args)
    except ValueError as exc:
        assert "must equal 1.0" in str(exc)
    else:
        raise AssertionError("Expected invalid alpha/beta values to raise ValueError")

    negative_features = features.copy()
    negative_features[0, 0] = -1.0
    valid_proposed_args = SimpleNamespace(
        theta_max=2.0,
        eval_points=3,
        order=2,
        weighting_mode="UrbanFEATHER_edgeCost_accessScore",
        edge_alpha=0.5,
        access_beta=0.5,
    )
    try:
        FEATHER().fit(graph, negative_features, valid_proposed_args)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative accessibility to raise ValueError")

    print("UrbanFEATHER weighting smoke test: OK")
    print("graph nodes:", graph.number_of_nodes())
    print("graph edges:", graph.number_of_edges())
    print("raw feature shape:", features.shape)
    print("embedding shape:", expected_embedding_shape)
    print("FEATHER row A:", np.round(feather_A[0], 6))
    print("UrbanFEATHER row A:", np.round(urban_A[0], 6))
    print(
        "UrbanFEATHER_edgeCost_accessScore row A:",
        np.round(proposed_A[0], 6),
    )


if __name__ == "__main__":
    main()