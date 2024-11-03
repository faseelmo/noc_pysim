import os 
import yaml 
import torch
import argparse 

from scipy.stats import kendalltau  

from training.model import GNNHetero, HeteroGNN
from training.noc_dataset import NocDataset
from data.utils import get_weights_from_directory

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    args = parser.parse_args()

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]
    USE_HETERO_WRAPPER  = params["USE_HETERO_WRAPPER"]

    dataset = NocDataset("data/training_data/simulator/test")
    data    = dataset[0]

    if not USE_HETERO_WRAPPER:
        model = GNNHetero( HIDDEN_CHANNELS, NUM_MPN_LAYERS, data.metadata() )  
        print(f"Using GNNHetero")
    else: 
        model = HeteroGNN( HIDDEN_CHANNELS, NUM_MPN_LAYERS )
        print(f"Using HeteroGNN")

    model(data)
    weights_path = get_weights_from_directory(args.model_path, f"{args.epoch}.pth" )
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path))

    map_test_dir    = "data/training_data/simulator/map_test"
    num_dirs        = len(os.listdir(map_test_dir))

    tau_list   = []

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
        print(f"Tau: {tau}, p_val: {p_val}")

    average_tau = sum(tau_list)/len(tau_list)   
    average_tau = round(average_tau, 2)
    print(f"Average tau is {average_tau}")

    # Create the file path
    file_path = os.path.join(args.model_path, f'avg_tau_{average_tau}.txt')

    # Write the average tau to the file
    with open(file_path, 'w') as file:
        file.write(f"Average tau is {average_tau}\n")





