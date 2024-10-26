
from data.utils import load_graph_from_json
from src.utils  import visuailize_noc_application





if __name__ == "__main__" :

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    args = parser.parse_args()

    graph = load_graph_from_json(f"data/training_data/simulator/test/{args.idx}.json")
    visuailize_noc_application(graph)

