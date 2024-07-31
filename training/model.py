import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, SAGEConv
from training.utils import get_norm_adj
from torch_sparse import SparseTensor


class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = DirGCNConv(num_features, hidden_channels, alpha=0.5)
        self.conv2 = DirGCNConv(hidden_channels, hidden_channels, alpha=0.5)
        self.conv3 = DirGCNConv(hidden_channels, hidden_channels, alpha=0.5)
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


if __name__ == "__main__":

    from training.dataset import load_data, HomogenousGraph

    homogenous_dataset = HomogenousGraph("data/training_data")
    train_loader, val_loader = load_data(homogenous_dataset, batch_size=64)
    model = GNN(2, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    from tqdm import tqdm

    def train(loader):
        model.train()
        loop = tqdm(loader, desc="Training")

        for idx, data in enumerate(loop):
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    def test(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                out = model(data.x, data.edge_index,  data.batch)
                loss = criterion(out, data.y)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Valid loss: {avg_loss}")

    for epoch in range(1, 10000):
        print(f"Epoch {epoch}")
        train(train_loader)
        test(val_loader)
