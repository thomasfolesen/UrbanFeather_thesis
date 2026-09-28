import math
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy import sparse


class FEATHER:
    r"""An implementation of node-level unsupervised FEATHER.

    This project supports three transition-weighting modes:

    - ``FEATHER``:
      Original topology-only FEATHER transition probabilities. Edge lengths are
      ignored and every outgoing neighbour receives equal probability.

    - ``UrbanFEATHER``:
      Current UrbanFEATHER behaviour. Each edge length is converted to an
      inverse-length strength (1 / length), then row-normalized.

    - ``UrbanFEATHER_edgeCost_accessScore``:
      Experimental accessibility-aware UrbanFEATHER. The random-walk
      transition matrix combines:

          1. normalized inverse edge-length preference, and
          2. normalized destination-node accessibility preference.

      The two parts are combined additively as:

          A_tilde_new = edge_alpha * edge_share
                      + access_beta * access_share

      IMPORTANT: ``edge_alpha + access_beta`` must equal 1.0.

      Destination accessibility is computed from the ORIGINAL raw node feature
      matrix X, before theta/cosine/sine encoding. For each node, the current
      implementation takes the MEAN across all feature columns. This is an
      explicit modelling choice and can be replaced later if experiments justify
      a different aggregation rule.

    Args:
        theta_max (float): Maximal evaluation point. Default is 2.5.
        eval_points (int): Number of characteristic-function evaluation points.
            Default is 25.
        order (int): Scale - number of adjacency-matrix propagation steps.
            Default is 5.
    """

    FEATHER_MODE = "FEATHER"
    URBANFEATHER_MODE = "UrbanFEATHER"
    ACCESS_AWARE_MODE = "UrbanFEATHER_edgeCost_accessScore"

    def __init__(self, theta_max=2.5, eval_points=25, order=5):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self._X = None
        self._A_tilde = None
        self._transition_components = None
        self._weighting_mode = None

    def _validate_weighting_mode(self, weighting_mode):
        """Allow only the three weighting modes used in this project."""
        if weighting_mode is None:
            return self.URBANFEATHER_MODE

        valid_modes = [
            self.FEATHER_MODE,
            self.URBANFEATHER_MODE,
            self.ACCESS_AWARE_MODE,
        ]

        if weighting_mode not in valid_modes:
            raise ValueError(
                "weighting_mode must be 'FEATHER', 'UrbanFEATHER', or "
                "'UrbanFEATHER_edgeCost_accessScore'."
            )

        return weighting_mode

    def _row_normalize(self, matrix):
        """Return a CSR matrix whose non-empty rows sum to 1."""
        matrix = matrix.tocsr().astype(float)
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()

        inverse = np.zeros_like(row_sums, dtype=float)
        nonzero = row_sums > 0
        inverse[nonzero] = 1.0 / row_sums[nonzero]

        return sparse.diags(inverse).dot(matrix).tocsr()

    def _validate_alpha_beta(self, edge_alpha, access_beta):
        """Validate additive weighting parameters.

        Both weights must be non-negative and their total MUST be 1.0 so every
        combined non-empty row remains a valid probability distribution.
        """
        edge_alpha = float(edge_alpha)
        access_beta = float(access_beta)

        if edge_alpha < 0 or access_beta < 0:
            raise ValueError("edge_alpha and access_beta must be non-negative.")

        if not np.isclose(edge_alpha + access_beta, 1.0, atol=1e-9):
            raise ValueError(
                "edge_alpha + access_beta must equal 1.0. "
                f"Received {edge_alpha} + {access_beta} = "
                f"{edge_alpha + access_beta}."
            )

        return edge_alpha, access_beta

    def _prepare_raw_features(self, graph, X):
        """Validate and reshape the original raw node feature matrix."""
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D numeric array of node features.")

        if X.shape[0] != graph.number_of_nodes():
            raise ValueError(
                "X must contain exactly one row per graph node. "
                f"Graph has {graph.number_of_nodes()} nodes, X has {X.shape[0]} rows."
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        return X

    def _binary_topology_adjacency(self, graph):
        """Return a binary sparse adjacency matrix for graph connectivity.

        For MultiGraph/MultiDiGraph inputs NetworkX can accumulate multiple
        parallel edges into the same adjacency entry. For topology/accessibility
        preference we care about whether destination j is connected from source
        i, so every stored non-zero entry is converted to exactly 1.
        """
        adjacency = nx.adjacency_matrix(
            graph,
            nodelist=range(graph.number_of_nodes()),
            weight=None,
            dtype=float,
        ).tocsr()
        if adjacency.nnz:
            adjacency.data[:] = 1.0
        return adjacency

    def _create_D_inverse(self, graph):
        """Create the inverse weighted-degree matrix used by UrbanFEATHER.

        UrbanFEATHER works with weighted edges, so this helper builds D^-1 from
        the sum of the edge weights connected to each node.
        """
        index = np.arange(graph.number_of_nodes())
        weights_sum = np.array(
            [
                graph.degree(node, weight="weight")
                for node in range(graph.number_of_nodes())
            ],
            dtype=float,
        )
        weights_sum[weights_sum == 0] = 1.0
        values = 1.0 / weights_sum
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        return sparse.coo_matrix((values, (index, index)), shape=shape)

    def _create_feather_transition_matrix(self, graph):
        """Create the original FEATHER topology-only transition matrix."""
        adjacency = self._binary_topology_adjacency(graph)
        return self._row_normalize(adjacency)

    def _create_inverse_length_strength_matrix(self, graph):
        """Create a sparse adjacency matrix containing inverse edge lengths.

        The source graph is never modified.

        Existing UrbanFEATHER behaviour is preserved for non-positive lengths:
        a length <= 0 is replaced by 1e-9 before inversion. Therefore such an
        existing edge becomes an extremely large raw strength (1e9). This guard
        does not create new edges or self-loops.
        """
        graph_copy = graph.copy()

        for _, _, data in graph_copy.edges(data=True):
            length = float(data.get("weight", 1.0))

            if length <= 0:
                length = 1e-9

            data["weight"] = 1.0 / length

        return nx.adjacency_matrix(
            graph_copy,
            nodelist=range(graph_copy.number_of_nodes()),
            weight="weight",
            dtype=float,
        ).tocsr()

    def _create_urbanfeather_transition_matrix(self, graph):
        """Create current UrbanFEATHER inverse-edge-length probabilities."""
        inverse_length = self._create_inverse_length_strength_matrix(graph)
        return self._row_normalize(inverse_length)

    def _create_accessibility_transition_matrix(self, graph, raw_X, edge_transition):
        """Create destination-accessibility transition probabilities.

        Accessibility is computed from ORIGINAL raw X before FEATHER's
        characteristic-function transform.

        Current aggregation rule:

            accessibility(node) = mean(raw feature values for that node)

        This mean is intentionally simple and interpretable. It is highlighted
        in the marimo teaching app because it is a modelling choice, not a
        mathematical requirement of FEATHER.

        Higher values are assumed to mean MORE accessible / MORE desirable.
        Negative accessibility values are therefore rejected.

        If every accessible score among a node's neighbours is 0, the row falls
        back to the current UrbanFEATHER edge probabilities. This means missing
        accessibility information does not arbitrarily alter the random walk.
        """
        raw_X = self._prepare_raw_features(graph, raw_X)

        # Important feature-direction difference:
        # our teaching notebook's inverse-distance features are "higher = better,"
        # while the original Pandana-generated UrbanFEATHER features appear to
        # encode distance, where "lower = better." Pandana distance features
        # should therefore be transformed before they are used as access scores.

        if np.any(raw_X < 0):
            raise ValueError(
                "UrbanFEATHER_edgeCost_accessScore requires non-negative raw "
                "accessibility features where larger means more accessible."
            )

        # One scalar destination score is needed per node. We currently use the
        # MEAN across all raw feature columns (e.g. School and Shops).
        access_scores = raw_X.mean(axis=1)

        topology = self._binary_topology_adjacency(graph)

        # Multiplying on the right by diag(access_scores) means each existing
        # edge i->j receives the ACCESSIBILITY OF DESTINATION NODE j.
        raw_access = topology.dot(sparse.diags(access_scores)).tocsr()
        access_transition = self._row_normalize(raw_access)

        raw_access_row_sums = np.asarray(raw_access.sum(axis=1)).ravel()
        edge_row_sums = np.asarray(edge_transition.sum(axis=1)).ravel()

        fallback_rows = (raw_access_row_sums == 0) & (edge_row_sums > 0)
        if np.any(fallback_rows):
            # access_transition is all-zero on these rows, so adding the selected
            # rows from edge_transition cleanly implements the fallback.
            fallback_selector = sparse.diags(fallback_rows.astype(float))
            access_transition = (
                access_transition + fallback_selector.dot(edge_transition)
            ).tocsr()

        return access_transition, access_scores

    def _create_transition_components(
        self,
        graph,
        X=None,
        weighting_mode=URBANFEATHER_MODE,
        edge_alpha=0.5,
        access_beta=0.5,
    ):
        """Build the transition matrix and expose its interpretable components."""
        mode = self._validate_weighting_mode(weighting_mode)

        if mode == self.FEATHER_MODE:
            transition = self._create_feather_transition_matrix(graph)
            return {
                "weighting_mode": mode,
                "A_tilde": transition,
                "edge_share": None,
                "access_share": None,
                "access_scores": None,
                "edge_alpha": None,
                "access_beta": None,
            }

        edge_share = self._create_urbanfeather_transition_matrix(graph)

        if mode == self.URBANFEATHER_MODE:
            return {
                "weighting_mode": mode,
                "A_tilde": edge_share,
                "edge_share": edge_share,
                "access_share": None,
                "access_scores": None,
                "edge_alpha": 1.0,
                "access_beta": 0.0,
            }

        edge_alpha, access_beta = self._validate_alpha_beta(
            edge_alpha,
            access_beta,
        )

        if X is None:
            raise ValueError(
                "UrbanFEATHER_edgeCost_accessScore requires the original raw "
                "feature matrix X so destination accessibility can be computed."
            )

        access_share, access_scores = self._create_accessibility_transition_matrix(
            graph,
            X,
            edge_share,
        )

        transition = (
            edge_alpha * edge_share + access_beta * access_share
        ).tocsr()

        return {
            "weighting_mode": mode,
            "A_tilde": transition,
            "edge_share": edge_share,
            "access_share": access_share,
            "access_scores": access_scores,
            "edge_alpha": edge_alpha,
            "access_beta": access_beta,
        }

    def _create_A_tilde(
        self,
        graph,
        X=None,
        weighting_mode=URBANFEATHER_MODE,
        edge_alpha=0.5,
        access_beta=0.5,
    ):
        """Create a normalized transition matrix.

        The default remains ``UrbanFEATHER`` to preserve the behaviour of the
        current thesis code when old callers use ``_create_A_tilde(graph)``.
        """
        return self._create_transition_components(
            graph=graph,
            X=X,
            weighting_mode=weighting_mode,
            edge_alpha=edge_alpha,
            access_beta=access_beta,
        )["A_tilde"]

    def fit(self, graph, X, args):
        """Fit a FEATHER node embedding.

        Required legacy args:
            theta_max, eval_points, order

        Optional weighting args:
            weighting_mode:
                "FEATHER"
                "UrbanFEATHER" (default when omitted)
                "UrbanFEATHER_edgeCost_accessScore"

            edge_alpha:
                Weight given to normalized inverse-edge-length preference.
                Default 0.5 for the accessibility-aware mode.

            access_beta:
                Weight given to normalized destination-accessibility preference.
                Default 0.5 for the accessibility-aware mode.

        IMPORTANT: edge_alpha + access_beta MUST equal 1.0.
        """
        raw_X = self._prepare_raw_features(graph, X)

        weighting_mode = getattr(
            args,
            "weighting_mode",
            self.URBANFEATHER_MODE,
        )
        edge_alpha = getattr(args, "edge_alpha", 0.5)
        access_beta = getattr(args, "access_beta", 0.5)

        components = self._create_transition_components(
            graph=graph,
            X=raw_X,
            weighting_mode=weighting_mode,
            edge_alpha=edge_alpha,
            access_beta=access_beta,
        )

        A_tilde = components["A_tilde"]
        self._A_tilde = A_tilde
        self._transition_components = components
        self._weighting_mode = components["weighting_mode"]

        theta = np.linspace(0.01, args.theta_max, args.eval_points)

        # FEATHER characteristic-function encoding.
        encoded_X = np.outer(raw_X, theta)
        encoded_X = encoded_X.reshape(graph.number_of_nodes(), -1)
        encoded_X = np.concatenate(
            [np.cos(encoded_X), np.sin(encoded_X)],
            axis=1,
        )

        feature_blocks = []
        current_X = encoded_X

        for _ in range(args.order):
            current_X = A_tilde.dot(current_X)
            feature_blocks.append(current_X.copy())

        if feature_blocks:
            self._X = np.concatenate(feature_blocks, axis=1)
        else:
            self._X = np.empty((graph.number_of_nodes(), 0), dtype=float)

    def get_embedding(self):
        r"""Return the fitted node embedding."""
        return self._X

    def get_transition_matrix(self):
        """Return the transition matrix used by the most recent ``fit`` call."""
        if self._A_tilde is None:
            raise RuntimeError("fit() must be called before get_transition_matrix().")
        return self._A_tilde

    def get_transition_components(self):
        """Return transition components from the most recent ``fit`` call."""
        if self._transition_components is None:
            raise RuntimeError("fit() must be called before get_transition_components().")
        return self._transition_components


class FEATHERG:
    r"""An implementation of the graph level unsupervised FEATHER.

    This class is retained from the existing thesis code. The weighting-mode work
    in this branch focuses on the node-level FEATHER class used by the smoke test
    and marimo experiments.
    """

    def __init__(self, theta_max=2.5, eval_points=25, order=5, pooling="mean"):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self.pooling = pooling

    def _pooling(self, features):
        if self.pooling == "min":
            graph_embedding = np.min(features, axis=0)
        elif self.pooling == "max":
            graph_embedding = np.max(features, axis=0)
        else:
            graph_embedding = np.mean(features, axis=0)
        return graph_embedding

    def _fit_a_FEATHER(self, graph):
        sub_model = FEATHER(self.theta_max, self.eval_points, self.order)
        feature = np.array(
            [math.log(graph.degree(node) + 1) for node in range(graph.number_of_nodes())]
        )
        feature = feature.reshape(-1, 1)

        # Keep FEATHERG compatible with the thesis FEATHER.fit(graph, X, args)
        # signature while preserving the current UrbanFEATHER transition rule.
        args = SimpleNamespace(
            theta_max=self.theta_max,
            eval_points=self.eval_points,
            order=self.order,
            weighting_mode="UrbanFEATHER",
        )
        sub_model.fit(graph, feature, args)
        features = sub_model.get_embedding()
        graph_embedding = self._pooling(features)
        return graph_embedding

    def fit(self, graphs):
        self._X = np.array([self._fit_a_FEATHER(graph) for graph in tqdm(graphs)])

    def get_embedding(self):
        return self._X