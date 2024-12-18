import os 
import yaml
import torch
import argparse
import importlib.util 
import matplotlib.pyplot as plt

from data.utils import get_weights_from_directory, extract_epoch    
from training.train import get_true_pred_hetero, get_max_latency_hetero

def append_if_not_none(true_values, pred_values, true_list, pred_list):
    if true_values is not None and pred_values is not None:
        for true, pred in zip(true_values.flatten(), pred_values.flatten()):
            true_list.append(true)
            pred_list.append(pred)

def infer(model, dataset) -> dict[int, dict[str, list]]:
    """
    Perform inference on the dataset and return:
    dict[int, dict[str, list]], where the index is the dataset index, 
    and the inner dictionary contains lists of truth and prediction values.
    """
    all_node_true_list = []
    all_node_pred_list = []

    latency_true_list = []
    latency_pred_list = []
    
    for i in range(len(dataset)): 
        data = dataset[i]
        output = model(data.x_dict, data.edge_index_dict)
        true_task, pred_task, true_exit, pred_exit = get_true_pred_hetero(data, output)

        # Append if not None and convert tensors to detached NumPy arrays
        if true_task is not None and pred_task is not None:
            all_node_true_list.extend(true_task.detach().cpu().numpy().flatten())
            all_node_pred_list.extend(pred_task.detach().cpu().numpy().flatten())
        
        if true_exit is not None and pred_exit is not None:
            latency_true_list.extend(true_exit.detach().cpu().numpy().flatten())
            latency_pred_list.extend(pred_exit.detach().cpu().numpy().flatten())

        # Get max latency and append as NumPy arrays
        latency_true, latency_pred = get_max_latency_hetero(data, output)

        latency_true_list.append(latency_true)
        latency_pred_list.append(latency_pred)

    # Package results into a dictionary
    results = {
        "true_all_nodes": all_node_true_list,
        "pred_all_nodes": all_node_pred_list,
        "true_latency": latency_true_list,
        "pred_latency": latency_pred_list    
    }

    return results

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


def get_map_latency(model, dataset_obj, map_test_dir): 
    num_dirs = len(os.listdir(map_test_dir))

    map_result_dict = {}
    for i in range(num_dirs):
        dir_path = os.path.join(map_test_dir, f"{i}")
        map_dataset = dataset_obj(dir_path)

        results = infer(model, map_dataset)
        reduced_dict = {}
        reduced_dict['true_latency'] = results['true_latency']
        reduced_dict['pred_latency'] = results['pred_latency']
        
        map_result_dict[i]  = reduced_dict  

    return map_result_dict

def plot_all_map_pred(map_result_dict, plot_path, plot_name):    
    num_subplots = len(map_result_dict)  # Number of datasets
    cols = 3  # Number of columns in the subplot grid
    rows = (num_subplots + cols - 1) // cols  # Calculate rows needed for grid

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    for i, (key, dataset_results) in enumerate(map_result_dict.items()):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        # Extract truth and prediction values for this dataset
        truth = dataset_results['true_latency']
        pred = dataset_results['pred_latency']

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


def plot_application_pred(result_dict, plot_path, plot_name):
    
    save_path = os.path.join(plot_path, plot_name)

    # Extract truth and predictions for all nodes
    truth_all_nodes = result_dict["true_all_nodes"]
    pred_all_nodes = result_dict["pred_all_nodes"]

    # Extract latency truth and predictions
    truth_latency = result_dict["true_latency"]
    pred_latency = result_dict["pred_latency"]

    # Create the subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All Nodes
    axes[0].scatter(truth_all_nodes, pred_all_nodes, alpha=0.7)
    axes[0].plot([min(truth_all_nodes), max(truth_all_nodes)], 
                 [min(truth_all_nodes), max(truth_all_nodes)], 
                 color='red', linestyle='--', label='y = x (Ideal)')
    axes[0].set_xlabel('Truth (All Nodes)')
    axes[0].set_ylabel('Prediction (All Nodes)')
    axes[0].set_title('Truth vs Prediction (All Nodes inc. start and end)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Latency
    axes[1].scatter(truth_latency, pred_latency, alpha=0.7)
    axes[1].plot([min(truth_latency), max(truth_latency)], 
                 [min(truth_latency), max(truth_latency)], 
                 color='red', linestyle='--', label='y = x (Ideal)')
    axes[1].set_xlabel('Truth Latency')
    axes[1].set_ylabel('Predicted Latency')
    axes[1].set_title('Truth vs Prediction (Latency)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--with_network", action="store_true", help="Use the model with network")   
    parser.add_argument("--model_path", type=str, default="" ,help="Path to the results folder")
    parser.add_argument("--epoch", type=str, default="50" ,help="Epoch number of the model to load")
    args = parser.parse_args()

    plot_path = os.path.join(args.model_path, "plots")
    os.mkdir(plot_path) if not os.path.exists(plot_path) else None

    model_spec   = importlib.util.spec_from_file_location("model", os.path.join(args.model_path, "model.py"))
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    dataset_spec   = importlib.util.spec_from_file_location("dataset", os.path.join(args.model_path, "dataset.py"))
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    dataset_spec.loader.exec_module(dataset_module)
    
    params_yaml_path = os.path.join(args.model_path, "params.yaml")
    params_yaml_path = os.path.join(args.model_path, "params.yaml")
    params           = yaml.safe_load(open(params_yaml_path))

    channels    = params["HIDDEN_CHANNELS"]
    num_layers  = params["NUM_MPN_LAYERS"]
    mesh_size   = params["MESH_SIZE"]

    DATA_DIR     = params["DATA_DIR"]

    test_dir     = f"{DATA_DIR}/test"
    map_test_dir    = "data/training_data/with_network_4x4_v4/map_test"

    if args.with_network:
        dataset_obj = dataset_module.NocDataset
        # dataset     = dataset_obj(test_dir)
        model       = model_module.HeteroGNN( channels, num_layers, mesh_size )

    else: 
        is_hetero      = params["IS_HETERO"].strip().lower()      == "true" 
        has_dependency = params["HAS_DEPENDENCY"].strip().lower() == "true" 
        has_exit       = params["HAS_EXIT"].strip().lower()       == "true" 
        has_scheduler  = params["HAS_SCHEDULER"].strip().lower()  == "true" 
        conv_type      = params["CONV_TYPE"]

        dataset        = dataset_module.CustomDataset( "data/training_data/without_network/test", 
                                                        is_hetero, 
                                                        has_dependency,
                                                        has_exit, 
                                                        has_scheduler,
                                                        False ) 
        data    = dataset[0]
        model   = model_module.MPNHetero( channels, 
                                          num_layers, 
                                          conv_type, 
                                          data.metadata() ) 

    model_path  = os.path.join(args.model_path, "models")
    weight_path = get_weights_from_directory( model_path, args.epoch )
    epoch       = extract_epoch(weight_path)
    print(f"Weight path is {weight_path}, epoch is {epoch}")
    model.load_state_dict(torch.load(weight_path, weights_only=False))

    # results = infer(model, dataset)
    # plot_application_pred(results, plot_path, f"application_latency_all_nodes.png")

    if args.with_network:
        map_results = get_map_latency(model, dataset_obj, map_test_dir)
        plot_all_map_pred(map_results, plot_path, f"map_latency.png")


