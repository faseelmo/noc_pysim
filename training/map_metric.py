import os 
import re 
import yaml 
import torch
import argparse 
import numpy as np
import importlib.util 

from scipy.stats import kendalltau, spearmanr, pearsonr

from data.utils import get_weights_from_directory, get_all_weights_from_directory, extract_epoch    
from training.utils import print_parameter_count


def get_mapping_tau(model, NocDataset, epoch, show): 
    map_test_dir    = "data/training_data/simulator/map_test"
    num_dirs        = len(os.listdir(map_test_dir))

    tau_list        = []
    p_value_list    = []
    std_list        = []
    count = 0

    for i in range(num_dirs): 
        dir = os.path.join(map_test_dir, f"{i}")
        map_dataset = NocDataset(dir)

        truth_list = []
        pred_list  = []

        for j in range(len(map_dataset)): 
            data = map_dataset[j]
            output = model(data)['task']

            latency_truth   = torch.max(data['task'].y).detach().cpu().numpy()
            latency_pred    = torch.max(output).detach().cpu().numpy() 

            truth_list.append(latency_truth.item())
            pred_list.append(latency_pred.item())

        tau, p_val = kendalltau(truth_list, pred_list)
        tau_list.append(tau)
        p_value_list.append(p_val)

        max_truth = max(truth_list) 
        min_truth = min(truth_list)
        std_truth = np.std(truth_list)
        std_list.append(std_truth)
        range_truth = max_truth - min_truth
        if show:
            print(f"{count}. Tau: {round(tau, 2)}, \tp_val: {round(p_val, 2)}, \trange: {round(range_truth, 2)}, \tStd: {round(std_truth, 2)}")
        count += 1

    average_tau     = round(sum(tau_list)/len(tau_list), 2)
    average_p_val   = round(sum(p_value_list)/len(p_value_list), 2)
    std_tau         = round(np.std(tau_list), 2)

    print(f"Epoch: {epoch}\tAverage tau = {average_tau}\tAverage p_val = {average_p_val}\tStd p_val = {std_tau}")

    return average_tau, average_p_val

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    parser.add_argument("--find", action="store_true", help="Find the epoch with the best tau")
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

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]

    dataset = NocDataset("data/training_data/simulator/test")
    data    = dataset[0]

    model = HeteroGNN( HIDDEN_CHANNELS, NUM_MPN_LAYERS )
    model(data)

    print(f"Using HeteroGNN")
    print_parameter_count(model)

    weight_paths = []
    model_path = os.path.join(args.model_path, "models")
    if not args.find: 
        show_tau    = True
        weight_path = get_weights_from_directory( model_path, args.epoch )
        weight_paths.append(weight_path)

    else: 
        show_tau     = False
        weight_paths = get_all_weights_from_directory(model_path)
    
    max_tau     = 0
    best_epoch  = 0
    its_p_val   = 0

    for weight_path in weight_paths:
        print(f"\nLoading model from {weight_path}")
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        epoch = extract_epoch(weight_path)

        tau, p = get_mapping_tau(model, NocDataset, epoch, show=show_tau)

        if tau > max_tau: 
            max_tau     = tau
            best_epoch  = epoch
            its_p_val   = p

    result_str = f"Best epoch is {best_epoch} with tau = {max_tau} and p_val = {its_p_val}"
    print(f"{result_str}")

    file_path = os.path.join(args.model_path, f'results.txt')

    with open(file_path, 'w') as file:
        file.write(result_str)


