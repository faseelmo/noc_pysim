import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight, batch):

        print(f"Shape of x: {x.shape}")
        x = self.conv1(x, edge_index, edge_weight)

        print(f"Shape of x after conv1: {x.shape}")
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight)
        print(f"Shape of x after conv2: {x.shape}")

        x = global_mean_pool(x, batch)
        print(f"Shape of x after global_mean_pool: {x.shape}")

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


if __name__ == "__main__":

    from training.dataset import load_data

    train_loader, val_loader = load_data("data/training_data", batch_size=2)
    model = GNN(2, 16)

    for data in train_loader:
        print(f"Data is {data}")
        latency = model(data.x, data.edge_index, data.edge_attr, data.batch)
        print(f"Latency is {latency}")
        break 
