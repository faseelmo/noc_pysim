
import os 
import yaml
import torch
import argparse
import importlib.util 

from src.utils              import visuailize_noc_application
from data.utils             import load_graph_from_json, get_weights_from_directory

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    args = parser.parse_args()


    model_spec = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    dataset_spec = importlib.util.spec_from_file_location("noc_dataset", os.path.join(args.model_path, "noc_dataset.py"))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_module)

    # GNNHetero = model_module.GNNHetero
    HeteroGNN = model_module.HeteroGNN

    NocDataset = dataset_module.NocDataset

    index = args.idx
    graph = load_graph_from_json(f"data/training_data/simulator/map_test/9/{index}.json")

    if not args.infer:
        visuailize_noc_application(graph)
        exit()

    dataset = NocDataset("data/training_data/simulator/map_test/9")
    data    = dataset[index]
    print(f"Edge index dict is {data.edge_index_dict['task', 'mapped_to', 'pe']}")
    print(f"data is {data['task'].y}")

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]
    USE_HETERO_WRAPPER  = params["USE_HETERO_WRAPPER"]

    if not USE_HETERO_WRAPPER:
        model = GNNHetero( HIDDEN_CHANNELS, NUM_MPN_LAYERS, data.metadata() )
    else: 
        model = HeteroGNN( HIDDEN_CHANNELS, NUM_MPN_LAYERS )
    model(data)

    weights_path = get_weights_from_directory(args.model_path, f"{args.epoch}.pth" )
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path))

    output = model(data)['task']
    max_latency = torch.max(output).detach().cpu().numpy()  

    print(f"Output is {output}")
    print(f"Max latency is {max_latency}")

    visuailize_noc_application(graph, output.tolist())


