
def log_hetero_data(data) -> None:

        from torch_geometric.data import HeteroData
        assert isinstance(data, HeteroData), "Data is not of type HeteroData"

        print(f"\n---HeteroData---")
        print(f"Data.x: {data.x_dict}")
        print(f"\nEdges")
        for edge_index in data.edge_index_dict:
            print(f"\nEdge index: {edge_index} \n{data.edge_index_dict[edge_index]}")


def does_path_exist(model_name):
    import os
    import yaml

    training_params = yaml.safe_load(open("training/params.yaml"))
    dir_path = os.path.join(training_params["RESULTS_DIR"], model_name)
    model_path = os.path.join(dir_path, "models")   

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Folder '{model_path}' created.")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Folder '{dir_path}' created.")
    else:
        print(f"Folder '{dir_path}' already exists.")
        continue_prompt = input("Do you want to continue? (yes/no): ")
        if continue_prompt.lower() != "yes":
            exit()


def copy_file(src_path, dest_path):
    import shutil

    shutil.copy2(src_path, dest_path)
    print(f"Copied {src_path} to {dest_path}")


def print_parameter_count(model):
    num_params = sum(
        p.numel() 
        for p in model.parameters() 
        if p.requires_grad
    )
    print(f"Number of parameters: {num_params}")


def get_metadata(dataset_path, has_wait_time, use_noc_dataset: False):

    if use_noc_dataset:
        from training.noc_dataset import NocDataset
        dataset = NocDataset(dataset_path)
        metadata = dataset[0].metadata()

    else: 
        from training.dataset import CustomDataset
        dataset = CustomDataset(dataset_path, 
                                is_hetero       = True, 
                                has_wait_time   = has_wait_time, 
                                return_graph    = False)
        metadata = dataset[0].metadata()

    return metadata

def initialize_model(model, dataloader, device):
    """Initialize the model by performing a dummy forward pass."""
    import torch
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = next(iter(dataloader))
        data = data.to(device)  # Ensure data is on the correct device
        model(data)  # Trigger lazy initialization
        # print(f"Data passed through the model is {data}")

    # Verify all parameters are initialized
    for name, param in model.named_parameters():
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            raise ValueError(f"Parameter {name} is still uninitialized.")
    print(f"Model initialized")


def plot_and_save_loss(train_loss, valid_loss, test_metric, model_name):
    import matplotlib.pyplot as plt
    import pickle

    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_yscale("log")
    ax1.plot(epochs, train_loss, label="Training Loss")
    ax1.plot(epochs, valid_loss, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(epochs, test_metric, "g-", label="Kendall's Tau")
    ax2.set_ylabel("Kendall's Tau")
    ax2.tick_params(axis="y")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Training and Validation Loss (Log Scale) with Kendall's Tau")
    plt.savefig(
        f"training/results/{model_name}/validation_plot_log_with_kendalls_tau.png"
    )
    plt.clf()

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "kendalls_tau": test_metric,
    }
    with open(
        f"training/results/{model_name}/loss_dict_with_kendalls_tau.pkl", "wb"
    ) as file:
        pickle.dump(loss_dict, file)

