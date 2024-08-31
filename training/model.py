import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch_sparse           import SparseTensor
from torch_geometric.nn     import (
                                global_max_pool,
                                SAGEConv,
                                GraphConv,
                                to_hetero,
                                global_mean_pool,
                                Set2Set,
                                HeteroConv
                            )

from training.utils         import get_norm_adj


class MPN(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels=None, num_conv_layers=3):
        """
        Message Passing Network (MPN)  
        Uses GraphConv to create a multi-layered model
        GraphConv is lazily initialized
        """
        super().__init__()
        torch.manual_seed(0)

        assert num_conv_layers >= 2, "Number of convolution layers should be at least 2."

        if output_channels is None:
            output_channels = hidden_channels

        self.conv_list = nn.ModuleList()
        self.conv_list.append(GraphConv(-1, hidden_channels))

        for _ in range(num_conv_layers - 2):
            self.conv_list.append(GraphConv(hidden_channels, hidden_channels))

        self.conv_list.append(GraphConv(hidden_channels, output_channels))

    def forward(self, x, edge_index):

        for conv in self.conv_list[:-1]:
            x = conv(x, edge_index).relu()

        x = self.conv_list[-1](x, edge_index)
        
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_channels, output_channels=1):
        """
        Muli-Layer Perceptron (MLP)  
        """
        super().__init__()
        torch.manual_seed(0)

        self.lin1 = torch.nn.Linear(input_channels, 5)
        self.lin2 = torch.nn.Linear(5, output_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_mpn_layers):
        """
        Note:  
        1. Message Passing Network (MPN) is lazily initialized  
        2. Global Max pooling  
        3. Forward pass requires torch_geometric.data.Data object
        5. Forward pass outputs a single value

        args:  
            hidden_channels : Number of channels (width) in MPN and MLP
            num_mpn_layers  : Number of layers (depth) in MPN   
        """
        super().__init__()
        torch.manual_seed(0)

        self.mpn = MPN(hidden_channels, num_mpn_layers)
        self.mlp = MLP(hidden_channels)

    def forward(self, data):
        x           = data.x
        edge_index  = data.edge_index
        batch       = data.batch

        x = self.mpn(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.mlp(x)

        return x


class GNNHeteroPooling(torch.nn.Module):
    def __init__(self, hidden_channels, num_mpn_layers, metadata):
        """
        Note:  
        1. Message Passing Network (MPN) is lazily initialized  
        2. Uses HeteroConv to create a multi-layered model  
        3. Global Max pooling  
        4. Forward pass requires torch_geometric.data.HeteroData object
        5. Forward pass outputs a single value

        args:  
            hidden_channels : Number of channels (width) in MPN and MLP
            num_mpn_layers  : Number of layers (depth) in MPN   
        """
        super().__init__()

        mpn             = MPN(hidden_channels, num_conv_layers=num_mpn_layers)
        self.mpn_hetero = to_hetero(mpn, metadata, aggr="sum")

        self.nodes_type = metadata[0]
        self.mlp        = nn.ModuleDict()

        for node_type in self.nodes_type:
            self.mlp[node_type] = MLP(hidden_channels)

        node_types      = len(metadata[0])
        self.output_mlp = nn.Linear(node_types, 1)

    def forward(self, data):

        x_dict          = data.x_dict
        edge_index_dict = data.edge_index_dict
        batch_dict      = data.batch_dict
        x_mpn_dict      = self.mpn_hetero(x_dict, edge_index_dict)

        out_list = []

        for node_type in self.nodes_type:
            x_mpn       = x_mpn_dict[node_type]
            batch       = batch_dict[node_type]

            global_max  = global_max_pool(x_mpn, batch)
            mlp_out     = self.mlp[node_type](global_max)  # Forward pass through MLP

            out_list.append(mlp_out)

        out_tensor  = torch.cat(out_list, dim=1)  # Concatenate the output of MLPs
        out         = self.output_mlp(out_tensor)  # Forward pass through the output MLP

        return out

class GNNHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_mpn_layers, metadata):
        """
        Note:  
        1. Message Passing Network (MPN) is lazily initialized  
        2. Uses HeteroConv to create a multi-layered model  
        3. Global Max pooling  
        4. Forward pass requires torch_geometric.data.HeteroData object
        5. Forward pass outputs start and end cycle for each task node  

        args:  
            hidden_channels : Number of channels (width) in MPN and MLP
            num_mpn_layers  : Number of layers (depth) in MPN   
        """
        super().__init__()

        num_output_features     = 2
        mpn                     = MPN(hidden_channels, num_output_features, num_mpn_layers)
        self.mpn_hetero         = to_hetero(mpn, metadata, aggr="sum")


    def forward(self, data):
        x_dict          = data.x_dict
        edge_index_dict = data.edge_index_dict

        out_dict = self.mpn_hetero(x_dict, edge_index_dict)

        return out_dict['task']

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

    IDX             = 1000
    BATCH_SIZE      = 1
    HIDDEN_CHANNELS = 5

    torch.manual_seed(0)

    print(f"\n----Homogenous GNN----")

    homogenous_dataset  = CustomDataset("data/training_data", is_hetero=False)
    data, _             = homogenous_dataset[IDX]

    print(f"Data is \n{data}")
    print(f"Edge index is \n{data.edge_index}")

    gnn_model   = GNN(hidden_channels=HIDDEN_CHANNELS, num_mpn_layers=HIDDEN_CHANNELS)
    output      = gnn_model(data)

    print(f"Output {output}")

    print(f"\n----Heterogenous Pooling GNN----")

    hetero_pooling_dataset      = CustomDataset("data/training_data", is_hetero=True)
    data, _                     = hetero_pooling_dataset[IDX]

    hetero_gnn_pooling_model    = GNNHeteroPooling(
                                    hidden_channels=HIDDEN_CHANNELS,
                                    num_mpn_layers=3, 
                                    metadata=data.metadata())

    train_loader, _             = load_data(
                                    "data/training_data", 
                                    is_hetero=True, 
                                    batch_size=BATCH_SIZE, 
                                    validation_split=0.1)

    input_data, _               = next(iter(train_loader))
    output                      = hetero_gnn_pooling_model(input_data)

    print(f"Input data is \n{input_data}")

    assert (
        output.size()[0] == input_data.batch_size
    ), f"Batch size is {input_data.batch_size} != output size is {output.size()[0]}"

    print(f"\n----Heterogenous GNN----")

    hetero_gnn_model            = GNNHetero(
                                    hidden_channels=HIDDEN_CHANNELS, 
                                    num_mpn_layers=3,
                                    metadata=data.metadata())

    output                      = hetero_gnn_model(input_data)

    print(f"Output is {output} of shape {output.shape}")

    
