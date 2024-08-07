import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, SAGEConv, GraphConv
from training.utils import get_norm_adj
from torch_sparse import SparseTensor

from training.utils import get_norm_adj
from torch_geometric.nn import (
    global_max_pool,
    SAGEConv,
    HeteroConv,
    to_hetero,
    HeteroDictLinear,
)


class GNNHomo(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = DirSageConv(num_features, hidden_channels)
        self.conv2 = DirSageConv(hidden_channels, hidden_channels)
        self.conv3 = DirSageConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, 5)
        self.lin2 = torch.nn.Linear(5, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = nn.Linear(input_dim, output_dim)  # W_1 (src to dst)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim)  # W_2 (dst to src)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        # Can make this run faster if we cache the adj_norm and adj_t_norm
        # But will cause issues with batching. Need to think about this
        row, col = edge_index
        num_nodes = x.shape[0]

        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        self.adj_norm = get_norm_adj(adj, norm="dir")

        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
        self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        adj_norm_x = self.adj_norm @ x  # matrix multiplication
        adj_t_norm_x = self.adj_t_norm @ x

        out = self.alpha * self.lin_src_to_dst(adj_norm_x) + (
            1 - self.alpha
        ) * self.lin_dst_to_src(adj_t_norm_x)

        return out


class GNNHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, node_types):
        super(GNNHetero, self).__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("task", "depends_on", "task"): DirGCNHetero(
                        hidden_channels, hidden_channels, node_types, alpha=0.5
                    ),
                    ("dependency", "depends_on", "task"): DirGCNHetero(
                        hidden_channels, hidden_channels, node_types, alpha=0.5
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):

        out_dict = {}

        for edge_type, edge_index in edge_index_dict.items():
            x = x_dict[edge_type[0]]
            edge_index = edge_index_dict[edge_type]
            print(f"X is \n{x}")
            print(f"Edge index is \n{edge_index}")
            # out_dict[edge_type] = self.convs[0]({edge_type: x}, edge_index


        # for conv in self.convs:
        #     x_dict = conv(x_dict, edge_index_dict)


class DirGCNHetero(torch.nn.Module):
    def __init__(self, input_dim, output_dim, node_types, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # W_1 (src to dst) for each node type
        self.lin_src_to_dst = HeteroDictLinear(-1, output_dim, types=node_types)
        # W_2 (dst to src) for each node type
        self.lin_dst_to_src = HeteroDictLinear(-1, output_dim, types=node_types)
        self.alpha = alpha

    def forward(self, x, edge_index):
        # Can make this run faster if we cache the adj_norm and adj_t_norm
        # But will cause issues with batching. Need to think about this
        row, col = edge_index
        num_nodes = x.shape[0]

        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        self.adj_norm = get_norm_adj(adj, norm="dir")

        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
        self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        adj_norm_x = self.adj_norm @ x  # matrix multiplication
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

    from training.dataset import load_data, CustomDataset

    INDEX = 2
    DATA_DIR = "data/training_data"

    hetero_dataset = CustomDataset(training_data_dir=DATA_DIR, is_hetero=True)
    homo_dataset = CustomDataset(training_data_dir=DATA_DIR, is_hetero=False)

    homo_model = GNNHomo(num_features=2, hidden_channels=32)

    print(f"-------Homogenous Graph-------")
    print(f"Homo Model is : {homo_model}")
    hetero_data = hetero_dataset[INDEX]
    homo_data = homo_dataset[INDEX]
    output = homo_model(homo_data.x, homo_data.edge_index, homo_data.batch)

    print(f"X_dict is \n{hetero_data.x_dict}")
    print(f"edge_dict is \n{hetero_data.edge_index_dict}")

    print(f"-------Heterogenous Graph-------")
    node_types = ["task", "dependency"]
    hetero_model = GNNHetero(hidden_channels=32, num_layers=3, node_types=node_types)
    output = hetero_model(hetero_data.x_dict, hetero_data.edge_index_dict)

    # print(f"Data is {data}")
    # print(f"Edge index is {data.edge_index_dict}")
    # hetero_model(data.x_dict, data.edge_index_dict)

    from tqdm import tqdm

    def train(loader, model, optimizer, criterion):
        model.train()
        loop = tqdm(loader, desc="Training")

        for idx, data in enumerate(loop):
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    def test(loader, model, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Valid loss: {avg_loss}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # criterion = torch.nn.MSELoss()

    # for epoch in range(1, 10000):
    #     print(f"Epoch {epoch}")
    #     train(train_loader)
    #     test(val_loader)
