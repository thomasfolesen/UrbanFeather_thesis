import math
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy import sparse

class FEATHER:
    r"""An implementation of the node level unsupervised FEATHER.
    
    Args:
        theta_max (float): Maximal evaluation point. Default is 2.5.
        eval_points (int): Number of characteristic function evaluation points. Default is 25.
        order (int): Scale - number of adjacency matrix powers. Default is 5.
    """
    def __init__(self, theta_max=2.5, eval_points=25, order=5):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        
        
    """ ORIGINAL IMplementation num of node probebility
    def _create_D_inverse(self, graph):
        
        Creating a sparse inverse degree matrix.
        
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        
        index = np.arange(graph.number_of_nodes())
        values = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse
    """ 
    
    def _create_D_inverse(self, graph):
        index = np.arange(graph.number_of_nodes())

        # Sum of transformed edge weights per node
        weights_sum = np.array([
            graph.degree(node, weight='weight')
            for node in range(graph.number_of_nodes())
        ], dtype=float)

        # Avoid division by zero
        weights_sum[weights_sum == 0] = 1.0

        values = 1.0 / weights_sum

        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix(
            (values, (index, index)),
            shape=shape
        )

        return D_inverse

    """
        def _create_A_tilde(self, graph):
            
            Creating a sparse normalized adjacency matrix.
            
            Arg types:
                * **graph** *(NetworkX graph)* - The graph to be embedded.
            Return types:
                * **A_tilde** *(Scipy array)* - The normalized adjacency matrix.
            
            A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
            D_inverse = self._create_D_inverse(graph) 
            A_tilde = D_inverse.dot(A)
            return A_tilde
    """ 
    def _create_A_tilde(self, graph):
        # Create a copy so original graph is untouched
        G = graph.copy()

        # Convert distance/length into similarity strength
        for u, v, d in G.edges(data=True):
            #grab the weight value for the edge, else set it to 1
            length = d.get("weight", 1.0)

            # Prevent division by zero
            if length <= 0:
                length = 1e-9

            # Short edges -> large influence
            d["weight"] = 1.0 / length

        A = nx.adjacency_matrix(
            G,
            nodelist=range(G.number_of_nodes()),
            weight='weight'
        )

        D_inverse = self._create_D_inverse(G)

        A_tilde = D_inverse.dot(A)

        return A_tilde

    #example model.fit(G, features, args)
    def fit(self, graph, X, args):
        """
        Fitting a FEATHER model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
            * **X** *(Numpy array)* - The matrix of node features.
        """
        # creates evenly spaced values from 0.01 up to theta_max.
        theta = np.linspace(0.01, args.theta_max, args.eval_points)
        # Example with:
            #   theta_max = 2.0
            #   eval_points = 3
            # theta = [0.01, 1.005, 2.0]
        """
        # Create the normalized weighted adjacency matrix A_tilde from the graph.
        # A_tilde describes:
            #   1. Which nodes are connected by edges.
            #   2. How strongly information should move across each edge.
        """
        A_tilde = self._create_A_tilde(graph)
        """
        For example for the graph
        #
        #   node 0 ---100m--- node 1 ---50m--- node 2 ---200m--- node 3
        #
        # _create_A_tilde() first converts each edge length into an
        # inverse-distance weight:
        #
        #   edge (0, 1): 1 / 100 = 0.01
        #   edge (1, 2): 1 / 50  = 0.02
        #   edge (2, 3): 1 / 200 = 0.005
        #
        # Shorter edges therefore get a larger weight / influence.
        #
        # These values are then normalized for each node so that the
        # outgoing transition weights from each node sum to 1.

                            TO NODE
                0      1      2      3
        FROM 0  [0.00,  1.00,  0.00,  0.00]
        FROM 1  [0.33,  0.00,  0.67,  0.00] edge 1---2 is half as short as 0---1 
        FROM 2  [0.00,  0.80,  0.00,  0.20]
        FROM 3  [0.00,  0.00,  1.00,  0.00]
        """

        # Multiply every node feature value with every theta evaluation point.
        X = np.outer(X, theta)
        """
        #  Multiply every value in X by every theta evaluation point.
            #
            # In our smoke test:
            #
            #   X =
            #   [[1.0],
            #    [2.0],
            #    [3.0],
            #    [4.0]]
            #
            #   theta = [0.01, 1.005, 2.0]
            #
            # np.outer(X, theta) gives:
            #
            #   [[0.01, 1.005, 2.0  ],
            #    [0.02, 2.010, 4.0  ],
            #    [0.03, 3.015, 6.0  ],
            #    [0.04, 4.020, 8.0  ]]
            #
            # Because this test has one feature per node,
            # each row currently corresponds to one node.
        """
        # Reshape X so that there is exactly one row per graph node.
        X = X.reshape(graph.number_of_nodes(), -1)
        # Create the real (cosine) and imaginary (sine) parts of the characteristic function for every node and evaluation point.
        X = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        """ feature matrix before random walk propogation
        [[cos0_1, cos0_2, cos0_3, sin0_1, sin0_2, sin0_3],   # node 0
        [cos1_1, cos1_2, cos1_3, sin1_1, sin1_2, sin1_3],   # node 1
        [cos2_1, cos2_2, cos2_3, sin2_1, sin2_2, sin2_3],   # node 2
        [cos3_1, cos3_2, cos3_3, sin3_1, sin3_2, sin3_3],   # node 3]
        """
        # node embeddings produced at each random-walk order.
        feature_blocks = []
        # Propagate the characteristic-function values one random-walk step
            # through the normalized weighted adjacency matrix (A_tilde).
            # A_tilde determines how much influence each neighboring node has.
        for _ in range(args.order):
            X = A_tilde.dot(X)
            # Save the node embedding for this specific random-walk order.
            feature_blocks.append(X)
        # Combine the embeddings from all random-walk orders into the final FEATHER node embedding.
        self._X = np.concatenate(feature_blocks, axis=1)

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._X

class FEATHERG:
    r"""An implementation of the graph level unsupervised FEATHER.
    
    Args:
        theta_max (float): Maximal evaluation point. Default is 2.5.
        eval_points (int): Number of characteristic function evaluation points. Default is 25.
        order (int): Scale - number of adjacency matrix powers. Default is 5.
        pooling (str): Pooling procedure (mean/max/min). Default is "mean".
    """
    def __init__(self, theta_max=2.5, eval_points=25, order=5, pooling="mean"):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self.pooling = pooling

    def _pooling(self, features):
        """
        Pooling a node level embedding to create a graph level embedding.
        
        Arg types:
            * **features* *(Numpy array)* - The node embedding array.
        Return types:
            * **graph_embedding** *(Numpy array)* - The whole graph embedding vector.
        """
        if self.pooling == "min":
            graph_embedding = np.min(features, axis=0)
        elif self.pooling == "max":
            graph_embedding = np.max(features, axis=0)
        else:
            graph_embedding = np.mean(features, axis=0)
        return graph_embedding

    def _fit_a_FEATHER(self, graph):
        """
        Creating a graph level embedding.
        
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        Return types:
            * **graph_embedding** *(Numpy array)* - The whole graph embedding vector.
        """
        sub_model = FEATHER(self.theta_max, self.eval_points, self.order)
        feature = np.array([math.log(graph.degree(node)+1) for node in range(graph.number_of_nodes())])
        feature = feature.reshape(-1, 1)
        sub_model.fit(graph, feature)
        features = sub_model.get_embedding()
        graph_embedding = self._pooling(features)
        return graph_embedding
    
    def fit(self, graphs):
        """
        Fitting a FEATHER model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._X = np.array([self._fit_a_FEATHER(graph) for graph in tqdm(graphs)])

    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._X
