
def log_hetero_data(data) -> None:

        from torch_geometric.data import HeteroData
        assert isinstance(data, HeteroData), "Data is not of type HeteroData"

        print(f"\n---HeteroData---")
        print(f"Data.x: {data.x_dict}")
        print(f"\nEdges")
        for edge_index in data.edge_index_dict:
            print(f"\nEdge index: {edge_index} \n{data.edge_index_dict[edge_index]}")


def does_model_dir_exit(dir_path) -> None:
    import os
    model_path = os.path.join(dir_path, "models")   
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Folder '{model_path}' created.")


def does_path_exist(dir_path) -> None:
    import os
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
    # print(f"Copied {src_path} to {dest_path}")


def print_parameter_count(model):
    num_params = sum(
        p.numel() 
        for p in model.parameters() 
        if p.requires_grad
    )
    print(f"Number of parameters: {num_params}")
    return num_params


def get_metadata(dataset_path, **kwargs): 
    
    use_noc_dataset = kwargs.get( "use_noc_dataset", False )

    if use_noc_dataset:
        from training.noc_dataset import NocDataset
        dataset  = NocDataset(dataset_path)
        metadata = dataset[0].metadata()

    else: 
        is_hetero       = kwargs.get( "is_hetero", False )
        has_scheduler   = kwargs.get( "has_scheduler", False )
        has_dependency  = kwargs.get( "has_dependency", False ) 
        has_exit        = kwargs.get( "has_exit", False )


        # print(f"Has task depend: {has_task_depend}")

        # print(f"Fetching metadata for dataset without_network")
        from training.dataset import CustomDataset
        dataset = CustomDataset( dataset_path, 
                                 is_hetero          = is_hetero, 
                                 has_scheduler      = has_scheduler, 
                                 has_exit           = has_exit,
                                 has_dependency     = has_dependency,
                                 return_graph       = False )

        metadata = dataset[0].metadata()

    return metadata

def initialize_model(model, dataloader, device, use_noc_dataset=False):
    """Initialize the model by performing a dummy forward pass."""
    import torch
    from torch_geometric.data import Data, HeteroData
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = next(iter(dataloader))
        data = data.to(device)  # Ensure data is on the correct device
        
        if isinstance(data, HeteroData):
            if use_noc_dataset:
                model(data)
            else: 
                model(data.x_dict, data.edge_index_dict)
        else:
            model(data.x, data.edge_index)  

    # Verify all parameters are initialized
    for name, param in model.named_parameters():
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            raise ValueError(f"Parameter {name} is still uninitialized.")
    # print(f"Model initialized")



def plot_and_save_loss(train_loss, valid_loss, test_metric, save_path):
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
        f"{save_path}/plot.png"
    )
    plt.clf()

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "kendalls_tau": test_metric,
    }
    with open(
        f"{save_path}/loss.pkl", "wb"
    ) as file:
        pickle.dump(loss_dict, file)


from itertools import combinations 

def adjusted_kendalls_tau(x: list, y: list, t_x: int=0, t_y: int=0) -> float:
    """
    Args:
        x, y        : lists of values
        t_x, t_y    : threshold values for x and y  
    """
    n = len(x)
    if n != len(y):
        raise ValueError("The two lists must have the same length.")

    C, D = 0.0, 0.0 

    for (i, j) in combinations(range(n), 2): 
        dx = x[i] - x[j]
        dy = y[i] - y[j]

        if ( dx > t_x and dy > t_y ) or ( dx < -t_x and dy < -t_y ): # Concordant
            C += 1

        elif ( dx > t_x and dy < -t_y ) or ( dx < -t_x and dy > t_y ): # Discordant
            D += 1
        
        else: # Tied
            if abs(dx) <= t_x and abs(dy) <= t_y: # Both pairs are effectively equal
                C += 1
            else: 
                D += 0.25

    total_pairs = n * (n - 1) / 2
    tau = (C - D) / total_pairs

    return round(tau, 2)
        


