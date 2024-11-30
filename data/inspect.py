
import os 
import yaml
import torch
import argparse
import importlib.util 

from data.utils             import load_graph_from_json, get_weights_from_directory, visualize_application, visualize_noc_application

if __name__ == "__main__" :

    parser  = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--with_network", action="store_true", help="Use the model with network")   
    args    = parser.parse_args()

    # model_spec = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    # model_module = importlib.util.module_from_spec(model_spec)
    # model_spec.loader.exec_module(model_module)

    # dataset_spec = importlib.util.spec_from_file_location("noc_dataset", os.path.join(args.model_path, "noc_dataset.py"))
    # dataset_module = importlib.util.module_from_spec(dataset_spec)
    # dataset_spec.loader.exec_module(dataset_module)

    # HeteroGNN   = model_module.HeteroGNN
    # NocDataset  = dataset_module.NocDataset

    index = args.idx

    if args.with_network:
        graph_path = f"data/training_data/with_network/train/{index}.json"
        graph = load_graph_from_json( graph_path)
        visualize_noc_application(graph)

    else: 
        graph_path = f"data/training_data/without_network/train/{index}.json"
        graph = load_graph_from_json( graph_path)
        visualize_application(graph)
    



