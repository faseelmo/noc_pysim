import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_max_pool,
    SAGEConv,
    GraphConv,
    to_hetero,
    global_mean_pool,
    Set2Set,
    HeteroConv,
)
from training.utils import get_norm_adj
from torch_sparse import SparseTensor


class MPN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = torch.nn.Linear(input_channels, 5)
        self.lin2 = torch.nn.Linear(5, output_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mpn = MPN(hidden_channels)
        self.mlp = MLP(hidden_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.mpn(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.mlp(x)

        return x


class GNNHetero(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        mpn = MPN(hidden_channels)
        self.nodes_type = metadata[0]

        self.mpn_hetero = to_hetero(mpn, metadata, aggr="sum")

        self.mlp = nn.ModuleDict()
        for node_type in self.nodes_type:
            self.mlp[node_type] = MLP(hidden_channels)

        node_types = len(metadata[0])
        self.output_mlp = nn.Linear(node_types, 1)

    def forward(self, data):

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        batch_dict = data.batch_dict

        x_mpn_dict = self.mpn_hetero(x_dict, edge_index_dict)

        out_list = []

        for node_type in self.nodes_type:
            x_mpn = x_mpn_dict[node_type]
            batch = batch_dict[node_type]

            global_max = global_max_pool(x_mpn, batch)
            mlp_out = self.mlp[node_type](global_max)  # Forward pass through MLP

            out_list.append(mlp_out)

        out_tensor = torch.cat(out_list, dim=1)  # Concatenate the output of MLPs
        out = self.output_mlp(out_tensor)  # Forward pass through the output MLP

        return out


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = nn.Linear(input_dim, output_dim)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        # if self.adj_norm is None: # Commented because we have nodes with
        # different number of edges and nodes.
        # So, we need to recompute the adjacency matrix for each batch
        row, col = edge_index
        num_nodes = x.shape[0]
        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        self.adj_norm = get_norm_adj(adj, norm="dir")
        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
        self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        adj_norm_x = self.adj_norm @ x
        adj_t_norm_x = self.adj_t_norm @ x

        out = self.alpha * self.lin_src_to_dst(adj_norm_x) + (
            1 - self.alpha
        ) * self.lin_dst_to_src(adj_t_norm_x)

        return out


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(
            input_dim, output_dim, flow="source_to_target", root_weight=False
        )
        self.conv_dst_to_src = SAGEConv(
            input_dim, output_dim, flow="target_to_source", root_weight=False
        )
        self.lin_self = nn.Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class LinearModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(LinearModel, self).__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(num_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


if __name__ == "__main__":

    from training.dataset import CustomDataset, load_data

    IDX = 1000
    HIDDEN_CHANNELS = 5

    torch.manual_seed(0)

    print(f"\n----Homogenous GNN----")
    homogenous_dataset = CustomDataset("data/training_data", is_hetero=False)
    data = homogenous_dataset[IDX]
    print(f"Data is \n{data}")
    print(f"Edge index is \n{data.edge_index}")
    gnn_model = GNN(hidden_channels=HIDDEN_CHANNELS)
    output = gnn_model(data)
    print(f"Output of Homogenous GNN is {output}")

    print(f"\n----Heterogenous GNN----")
    hetero_dataset = CustomDataset("data/training_data", is_hetero=True)
    data = hetero_dataset[IDX]
    hetero_gnn_model = GNNHetero(
        hidden_channels=HIDDEN_CHANNELS,
        metadata=data.metadata(),
    )

    train_loader, _ = load_data(
        "data/training_data", is_hetero=True, batch_size=2, validation_split=0.1
    )

    train_data = next(iter(train_loader))
    output = hetero_gnn_model(train_data)

    assert (
        output.size()[0] == train_data.batch_size
    ), f"Batch size is {train_data.batch_size} != output size is {output.size()[0]}"
