import os
import yaml
import json

import argparse

from natsort import natsorted

from data.utils import (
    load_graph_from_json, 
    visualize_graph, 
    get_compute_list_from_json)

import torch
from training.model import GNN, GNNHetero, GNNHeteroPooling
from training.utils import get_metadata
from training.dataset import CustomDataset


if __name__ == "__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")

    args    = parser.parse_args()

    if args.test:
        data_dir            = os.path.join("data", "training_data", "test")
    else:
        data_dir            = os.path.join("data", "training_data")

    input_path          = os.path.join(data_dir, "input")
    target_path         = os.path.join(data_dir, "target")
    packet_list_path    = os.path.join(data_dir, "packet_list")

    input_files         = natsorted(os.listdir(input_path))
    target_files        = natsorted(os.listdir(target_path))
    packet_list_files   = natsorted(os.listdir(packet_list_path))

    if not args.infer:
        # Visualize the graphs without predicting the latencies

        for input_idx, target_idx, packet_list_idx in zip(input_files, target_files, packet_list_files):

            graph       = load_graph_from_json(os.path.join(input_path, input_idx))
            graph_json  = json.load(open(os.path.join(target_path, target_idx)))
            latency     = graph_json["latency"]

            compute_list    = get_compute_list_from_json(os.path.join(target_path, target_idx))
            packet_list     = json.load(open(os.path.join(packet_list_path, packet_list_idx)))
            visualize_graph(graph, latency, packet_list, compute_list)

    if args.infer:
        # Visualize the graphs and predict the latencies

        print(f"\nInference Mode")
        if not os.path.exists(args.model_path):
            print(f"Model path '{args.model_path}' does not exist.")
            exit()

        yaml_path       = os.path.join(args.model_path, "params.yaml")
        training_params = yaml.safe_load(open(yaml_path))
        device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        IS_HETERO       = training_params["IS_HETERO"]
        DO_POOLING      = training_params["DO_POOLING"]
        HIDDEN_CHANNELS = training_params["HIDDEN_CHANNELS"]
        NUM_MPN_LAYERS  = training_params["NUM_MPN_LAYERS"]
        MAX_CYCLE       = training_params["MAX_CYCLE"]
        HAS_WAIT_TIME   = training_params["HAS_WAIT_TIME"]

        print(f"\nModel Parameters: \n IS_HETERO: {IS_HETERO} \n "
              f"DO_POOLING: {DO_POOLING} \n HIDDEN_CHANNELS: {HIDDEN_CHANNELS} \n "
              f"NUM_MPN_LAYERS: {NUM_MPN_LAYERS}")   

        if IS_HETERO:
            metadata    = get_metadata(data_dir, HAS_WAIT_TIME)

            if DO_POOLING:
                print(f"\nGNNHeteroPooling Model Loaded")
                model   = GNNHeteroPooling(HIDDEN_CHANNELS, NUM_MPN_LAYERS, metadata).to(device)
            else:
                print(f"\nGNNHetero Model Loaded")
                model   = GNNHetero(HIDDEN_CHANNELS, NUM_MPN_LAYERS, metadata).to(device)

        elif not IS_HETERO:
            print(f"\nGNN Model Loaded")
            model       = GNN(HIDDEN_CHANNELS, NUM_MPN_LAYERS).to(device)

        # Initialize model since GraphConv is lazily initialized
        dataset             = CustomDataset(data_dir, is_hetero=IS_HETERO, return_graph=True, has_wait_time=HAS_WAIT_TIME)
        data_for_init, _    = next(iter(dataset))
        model(data_for_init)

        # Loading weights to the model
        weight_path         = os.path.join(args.model_path, f"LatNet_{args.epoch}_state_dict.pth")
        print(f"Loading weights from {weight_path}")
        model_state_dict    = torch.load(weight_path)
        model.load_state_dict(model_state_dict)

        for data, index_graph_tuple in dataset:

            data.to(device)
            # output      = model(data) * MAX_CYCLE
            output      = model(data) 
            
            task_output         = output["task"] * MAX_CYCLE
            task_truth          = data["task"].y * MAX_CYCLE

            if HAS_WAIT_TIME:
                task_depend_output  = output["task_depend"] * MAX_CYCLE
                task_depend_truth   = data["task_depend"].y * MAX_CYCLE


            nx_graph    = index_graph_tuple[1]
            graph_index = index_graph_tuple[0]

            truth_compute_list  = {}
            pred_compute_list   = {}

            for idx, node in nx_graph.nodes(data=True):

                if not HAS_WAIT_TIME:
                    if node["type"] == "task_depend":
                        node["type"] = "task"

                if node["type"] == "task":
                    tensor_idx              = graph_index["task"][idx] 

                    truth_compute_list[idx] = {
                        "start_cycle"   : int(task_truth[tensor_idx][0].item()),
                        "end_cycle"     : int(task_truth[tensor_idx][1].item())
                    }

                    pred_compute_list[idx]  = {
                        "start_cycle"   : int(task_output[tensor_idx][0].item()),
                        "end_cycle"     : int(task_output[tensor_idx][1].item())
                    }

                if node["type"] == "task_depend":
                    
                    tensor_idx              = graph_index["task_depend"][idx] 

                    truth_compute_list[idx] = {
                        "start_cycle"   : int(task_depend_truth[tensor_idx][0].item()),
                        "end_cycle"     : int(task_depend_truth[tensor_idx][1].item())
                    }

                    pred_compute_list[idx]  = {
                        "start_cycle"   : int(task_depend_output[tensor_idx][0].item()),
                        "end_cycle"     : int(task_depend_output[tensor_idx][1].item())
                    }

            visualize_graph(nx_graph, compute_list=truth_compute_list, pred_compute_list=pred_compute_list)





