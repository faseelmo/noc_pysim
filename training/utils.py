
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


def get_metadata(dataset_path, has_wait_time):
    from training.dataset import CustomDataset

    dataset = CustomDataset(
                dataset_path, 
                is_hetero       = True, 
                has_wait_time   = has_wait_time, 
                return_graph    = False)

    data    = dataset[0]

    return data.metadata()

def initialize_model(model, dataloader):
    """Necessary since GraphConv is lazily initialized"""
    data = next(iter(dataloader))
    model(data)

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


from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")
