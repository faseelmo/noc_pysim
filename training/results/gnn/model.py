import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight, batch):

        # print(f"Shape of x: {x.shape}")
        x = self.conv1(x, edge_index, edge_weight)

        # print(f"Shape of x after conv1: {x.shape}")
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight)
        # print(f"Shape of x after conv2: {x.shape}")

        x = global_mean_pool(x, batch)
        # print(f"Shape of x after global_mean_pool: {x.shape}")

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class LinearModel(nn.Module):
    def __init__(self, num_features):
        super(LinearModel, self).__init__()
        self.lin1 = nn.Linear(num_features, 5)
        self.lin2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

if __name__ == "__main__":

    from training.dataset import load_data

    train_loader, val_loader = load_data("data/training_data", batch_size=64)
    model = GNN(2, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    from tqdm import tqdm


    def train(loader):
        model.train()
        loop = tqdm(loader, desc="Training")

        for idx, data  in enumerate(loop):
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())


    def test(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Valid loss: {avg_loss}")

    for epoch in range(1, 10000):
        print(f"Epoch {epoch}")
        train(train_loader)
        test(val_loader)
