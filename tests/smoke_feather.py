import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATHER_SRC = PROJECT_ROOT / "src" / "Utils" / "FEATHER" / "src"

sys.path.insert(0, str(FEATHER_SRC))

from feather import FEATHER


def main():
    #Create undirected graph with four vertices connected in a chain 1---2---3...---n
    G = nx.path_graph(4)
    #Nodes :[0, 1, 2, 3]
    #Edges [(0, 1), (1, 2), (2, 3)]


    # Create a feature matrix with one feature value for each node
    features = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    # node 0 -> feature value features[0] = 1.0
    # node 1 -> feature value features[0] = 2.0
    #[array([1.]), array([2.]), array([3.]), array([4.])]

    args = SimpleNamespace(
        # theta_max:
            #  The largest characteristic-function evaluation point (theta).
            #  FEATHER evaluates the node features at values between 0.01 and theta_max.
        theta_max=2.0,
        # eval_points :
            # use eval_points amount of theta values
        eval_points=3,
        order=2,
    )

    model = FEATHER()
    model.fit(G, features, args)

    embedding = model.get_embedding()

    print("shape:", embedding.shape)
    print("finite:", np.isfinite(embedding).all())

    assert embedding.shape == (4, 4)
    assert np.isfinite(embedding).all()


if __name__ == "__main__":
    main()