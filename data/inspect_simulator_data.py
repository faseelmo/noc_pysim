
import os 
import yaml
import torch
import argparse

from src.utils              import visuailize_noc_application
from data.utils             import load_graph_from_json, get_weights_from_directory
from training.noc_dataset   import NocDataset
from training.model         import GNNHetero

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    args = parser.parse_args()

    index = args.idx
    graph = load_graph_from_json(f"data/training_data/simulator/map_test/1/{index}.json")

    if not args.infer:
        visuailize_noc_application(graph)
        exit()

    dataset = NocDataset("data/training_data/simulator/map_test/1")
    data    = dataset[index]
    print(f"Edge index dict is {data.edge_index_dict['task', 'mapped_to', 'pe']}")
    print(f"data is {data['task'].y}")

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]

    model               = GNNHetero( 
                            HIDDEN_CHANNELS, 
                            NUM_MPN_LAYERS, data.metadata() )  
    model(data)

    weights_path        = get_weights_from_directory(
                            args.model_path, 
                            f"{args.epoch}.pth" )

    model.load_state_dict(torch.load(weights_path))

    output = model(data)['task']
    print(f"Output is {output}")

    visuailize_noc_application(graph, output.tolist())


