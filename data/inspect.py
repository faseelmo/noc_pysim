
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
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    args    = parser.parse_args()

    model_spec = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    dataset_spec = importlib.util.spec_from_file_location("dataset", os.path.join(args.model_path, "dataset.py"))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_module)

    
    if args.with_network:
        dataset  = dataset_module.NocDataset("data/training_data/with_network/test", False)
    else:
        is_hetero = True
        has_scheduler = False
        has_task_depend = True 
        has_dependency = True 
        dataset  = dataset_module.CustomDataset("data/training_data/without_network/test", is_hetero, has_scheduler, has_task_depend, has_dependency, False) 
        data = dataset[0]
        model = model_module.MPNHetero( 64, 5, "graphconv", data.metadata())
        # data/training_data/with_network/test

    index = args.idx

    if args.with_network:
        graph_path = f"data/training_data/with_network/train/{index}.json"
        graph = load_graph_from_json( graph_path)
        visualize_noc_application(graph)

    else: 
        # graph_path = f"data/training_data/without_network/train/{index}.json"
        # graph = load_graph_from_json( graph_path)
        # visualize_application(graph)

        print(f"Data is {data}")

        data = dataset[index]
        output = model(data.x_dict, data.edge_index_dict)

        print(f"Real")
        print(f"Task is {data['task'].y}")
        print(f"Task Depend is {data['task_depend'].y}")
        print(f"")

        print(f"Predicted")
        print(f"Task is {output['task']}")
        print(f"Task Depend is {output['task_depend']}")

    



