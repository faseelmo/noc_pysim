
import os 
import yaml
import torch

from src.utils              import visuailize_noc_application
from data.utils             import load_graph_from_json
from training.noc_dataset   import NocDataset
from training.model         import GNNHetero

if __name__ == "__main__" :

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    args = parser.parse_args()

    index = args.idx
    graph = load_graph_from_json(f"data/training_data/simulator/test/{index}.json")

    if not args.infer:
        visuailize_noc_application(graph)
        exit()

    dataset = NocDataset("data/training_data/simulator/test")
    data    = dataset[index]

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]

    model = GNNHetero( HIDDEN_CHANNELS, NUM_MPN_LAYERS, data.metadata() )  
    model(data)

    files = os.listdir(args.model_path)
    for file in files:
        if f"{args.epoch}.pth" in file:
            weights_path = os.path.join(args.model_path, file)

    model.load_state_dict(torch.load(weights_path))
    output = model(data)['task']

    visuailize_noc_application(graph, output.tolist())


