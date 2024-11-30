import os 
import yaml
import torch
import argparse
import importlib.util 
import matplotlib.pyplot as plt

from data.utils import get_weights_from_directory, extract_epoch    

def infer(model, dataset) -> dict[int, dict[str, torch.Tensor]]: 

    """
    Return dict[ index, dict [ truth_all_nodes, pred_all_nodes ] ] where index is the index of the data in the dataset,
    """

    result_dict = {}
    for i in range(len(dataset)): 
        data = dataset[i]
        output = model(data)

        pred_latency    = output['task'][:,1]
        truth_latency   = data['task'].y[:,1]

        result_dict[i] = {}
        result_dict[i]["truth"] = truth_latency
        result_dict[i]["pred"] = pred_latency

    return result_dict

def get_execution_times(result_dict): 
    """
    Per graph, get the final execution time of the truth and pred
    """
    return_dict = {}
    for i, result in result_dict.items(): 
    
        argmax_truth = torch.argmax(result["truth"])

        truth_execution_time    = result["truth"][argmax_truth]    
        pred_execution_time     = result["pred"][argmax_truth]

        return_dict[i] = {}
        return_dict[i]["truth"] = truth_execution_time
        return_dict[i]["pred"] = pred_execution_time

    return return_dict


def get_map_accuracy(map_result_dict):

    data_list = []

    for i, result in map_result_dict.items():
        truth = result["truth"].item()
        pred = result["pred"].item()
        data_list.append((i, truth, pred))

    # Get least values and indices
    ordered_by_truth    = sorted(data_list, key=lambda x: x[1])
    ordered_by_pred     = sorted(data_list, key=lambda x: x[2])

    # Get the indices of the least values of truth and pred
    least_truth_index   = ordered_by_truth[0][0]
    least_pred_index    = ordered_by_pred[0][0]

    # Get the true latency values of the least truth and pred indices
    least_truth_value       = map_result_dict[least_truth_index]["truth"].item()
    least_pred_truth_value  = map_result_dict[least_pred_index]["truth"].item()

    return least_truth_index, least_pred_index, least_truth_value, least_pred_truth_value



def get_map_latency(model, dataset_obj): 
    map_test_dir = "data/training_data/simulator/map_test"
    num_dirs = len(os.listdir(map_test_dir))

    map_result_dict = {}
    for i in range(num_dirs):
        dir_path = os.path.join(map_test_dir, f"{i}")
        map_dataset = dataset_obj(dir_path)

        map_results         = infer(model, map_dataset)
        execution_results   = get_execution_times(map_results)
        map_result_dict[i]  = execution_results  
        least_truth_index, least_pred_index, least_truth_value, least_pred_truth_value = get_map_accuracy(execution_results)
        print(f"[{i}]Truth index {least_truth_index},\tPred index {least_pred_index},\t"
              f"True index value {least_truth_value},\tPred index value {least_pred_truth_value},\t"
              f"Error {least_pred_truth_value - least_truth_value}")

    return map_result_dict

def plot_all_map_pred(map_result_dict, plot_path, plot_name):    
    num_subplots = len(map_result_dict)  # Number of datasets
    cols = 3  # Number of columns in the subplot grid
    rows = (num_subplots + cols - 1) // cols  # Calculate rows needed for grid

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    for i, (key, dataset_results) in enumerate(map_result_dict.items()):
        row, col    = divmod(i, cols)
        ax          = axes[row, col]

        # Extract truth and pred lists from the result dictionary for this dataset
        truth   = [dataset_results[idx]["truth"].item() for idx in dataset_results]
        pred    = [dataset_results[idx]["pred"].item() for idx in dataset_results]

        # Plot the scatter plot for this dataset
        ax.scatter(truth, pred, alpha=0.7)
        ax.plot([min(truth), max(truth)], [min(truth), max(truth)], color='red', linestyle='--', label='y = x (Ideal)')
        ax.set_title(f"Dataset {key}")
        ax.set_xlabel('Truth Latency')
        ax.set_ylabel('Predicted Latency')
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis('off')

    # Save the plot
    save_path = os.path.join(plot_path, plot_name)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")


def plot_application_pred(result_dict, plot_path, plot_name, all_nodes=False):
    save_path = os.path.join(plot_path, plot_name)

    truth = []
    pred = []

    if all_nodes: 
        for i in result_dict:
            for j in range(len(result_dict[i]["truth"])):
                truth.append(result_dict[i]["truth"][j].item())
                pred.append(result_dict[i]["pred"][j].item())

    else: 
        for i in result_dict:
            # Extract truth and pred values from the result dictionary
            truth.append(result_dict[i]["truth"].item())
            pred.append(result_dict[i]["pred"].item())

    # Plot the truth vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(truth, pred, alpha=0.7)
    plt.plot([min(truth), max(truth)], [min(truth), max(truth)], color='red', linestyle='--', label='y = x (Ideal)')
    plt.xlabel('Truth Latency')
    plt.ylabel('Predicted Latency')
    plt.title('Truth vs Prediction Latency')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    args = parser.parse_args()

    plot_path = os.path.join(args.model_path, "plots")
    os.mkdir(plot_path) if not os.path.exists(plot_path) else None

    model_spec      = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    model_module    = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    dataset_spec    = importlib.util.spec_from_file_location("noc_dataset", os.path.join(args.model_path, "noc_dataset.py"))
    dataset_module  = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_module)

    HeteroGNN   = model_module.HeteroGNN
    NocDataset  = dataset_module.NocDataset

    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params_yaml_path    = os.path.join(args.model_path, "params.yaml")
    params              = yaml.safe_load(open(params_yaml_path))

    HIDDEN_CHANNELS     = params["HIDDEN_CHANNELS"]
    NUM_MPN_LAYERS      = params["NUM_MPN_LAYERS"]

    dataset = NocDataset("data/training_data/simulator/test")
    data    = dataset[0]

    model = HeteroGNN( HIDDEN_CHANNELS, NUM_MPN_LAYERS )
    model(data)

    model_path  = os.path.join(args.model_path, "models")
    weight_path = get_weights_from_directory( model_path, args.epoch )
    epoch       = extract_epoch(weight_path)
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    # Application Latency Plot 
    application_results = infer(model, dataset)
    plot_application_pred(application_results, plot_path, f"application_latency_all_nodes.png", all_nodes=True)
    application_execution_resutls = get_execution_times(application_results)
    plot_application_pred(application_execution_resutls, plot_path, f"application_latency.png")

    # Map Latency Plot 
    map_results = get_map_latency(model, NocDataset)
    plot_all_map_pred(map_results, plot_path, f"map_latency.png")

    

