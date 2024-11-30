import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch_sparse           import SparseTensor
from torch_geometric.nn     import (
                                global_max_pool,
                                SAGEConv,
                                GraphConv,
                                to_hetero,
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

        self.hidden_channels = hidden_channels  

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
        out_dict        = {}

        # Getting the batch size so that we can create a tensor of zeros
        # if 'task' node is not present in the batch
        batch_size     = torch.unique(batch_dict['dependency']).numel()
        print(f"Batch Size is {batch_size}")

        for node_type in self.nodes_type:
            x_mpn       = x_mpn_dict[node_type]
            batch       = batch_dict[node_type]

            print(f"\nNode type is {node_type}")
            print(f"X MPN shape {x_mpn.shape}")

            if node_type == 'task':
                target_shape = (batch_size, self.hidden_channels)

                if x_mpn.numel() == 0:
                    # print(f"Node type {node_type} is not present in the batch")
                    global_max = torch.zeros(target_shape, dtype=torch.float)
                else:
                    print(f"X MPN shape {x_mpn.shape}")
                    padding = torch.zeros((target_shape[0] - x_mpn.shape[0], target_shape[1]), dtype=x_mpn.dtype)
                    global_max = torch.cat((x_mpn, padding), dim=0)                

            else: 
                global_max = global_max_pool(x_mpn, batch)
                # if node_type == 'task':
                #     print(f"X MPN is {x_mpn}")
                #     print(f"Global Max is {global_max}")

            print(f"Global Max shape {global_max.shape}")

            out_dict[node_type] = self.mlp[node_type](global_max)  # Forward pass through MLP

            print(f"MLP output is {out_dict[node_type].shape}")


        
        # To ensure node order during the forward pass of the last MLP 
        out_list    = []
        node_order  = ['task', 'dependency', 'task_depend']

        for node_type in node_order:

            if node_type in out_dict:
                out_list.append(out_dict[node_type])

        # print(f"Out Dict is {out_dict}")
        # print(f"Out List is {out_list}")

        out_tensor  = torch.cat(out_list, dim=1)  # Concatenate the output of MLPs
        out         = self.output_mlp(out_tensor)  # Forward pass through the output MLP

        return out

class GNNHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_mpn_layers, metadata):
        """
        Note:  
        1. Message Passing Network (MPN) is lazily initialized  
        2. Uses HeteroConv to create a multi-layered model  
        3. Forward pass requires torch_geometric.data.HeteroData object
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

        return out_dict

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

    """
    Usage: python3 -m training.model True False False
    Conditions to Test: 
        1. Homogenous GNN Model
            python3 -m training.model 0 0 0 

        2. Heterogenous GNN Model
            python3 -m training.model 1 0 0 
            python3 -m training.model 1 0 1

        3. Heterogenous Pooling GNN Model
            [Works only for dataloader and not directly from CustomDataset. Issue with Batching]
            python3 -m training.model 1 1 0 
            python3 -m training.model 1 1 1
    """

    import sys

    from training.dataset import CustomDataset, load_data
    from data.utils import visualize_application

    if len(sys.argv) > 1:   
        HETERO_MODEL    = sys.argv[1].lower() in ['true', '1']
        DO_POOLING      = sys.argv[2].lower() in ['true', '1']
        HAS_WAIT_TIME   = sys.argv[3].lower() in ['true', '1']

    else:                   
        HETERO_MODEL    = False
        DO_POOLING      = False
        HAS_WAIT_TIME   = False

    print( f"Hetero Model   = {HETERO_MODEL}")
    print( f"Do Pooling     = {DO_POOLING}")
    print( f"Has Wait Time  = {HAS_WAIT_TIME}")

    IDX             = 10
    BATCH_SIZE      = 10
    HIDDEN_CHANNELS = 40

    torch.manual_seed(0)

    dataloader, _ = load_data(
                        "data/training_data",
                        is_hetero=HETERO_MODEL,
                        has_wait_time=HAS_WAIT_TIME,
                        batch_size=BATCH_SIZE)

    data = next(iter(dataloader))

    dataset     = CustomDataset(
                    "data/training_data", 
                    is_hetero=HETERO_MODEL, 
                    has_wait_time=HAS_WAIT_TIME, 
                    return_graph=True
                    )

    data_from_dataset, (index, graph) = dataset[IDX]
    visualize_application(graph)

    # data = data_from_dataset

    if not HETERO_MODEL:

        print(f"Using GNN Model (Homogenous)")

        model   = GNN(hidden_channels=HIDDEN_CHANNELS, 
                      num_mpn_layers=HIDDEN_CHANNELS)

    elif HETERO_MODEL: 

        if DO_POOLING:

            print(f"Using Heterogenous Pooling GNN Model")

            metadata = data_from_dataset.metadata()
            model = GNNHeteroPooling(
                        hidden_channels = HIDDEN_CHANNELS,
                        num_mpn_layers  = 3,
                        metadata        = metadata)

        else: 
        
            print(f"Using Heterogenous GNN Model")
    
            metadata = data_from_dataset.metadata()
            model = GNNHetero(
                        hidden_channels = HIDDEN_CHANNELS,
                        num_mpn_layers  = 3,
                        metadata        = metadata)

    if HETERO_MODEL:
        print(f"Metadata is {metadata}\n")
        print(f"\nData(Target) is ")

        print(f"[TASK]")
        print(f"Input {data['task'].x}")
        print(f"Output is {data['task'].y}\n")

        if HAS_WAIT_TIME:
            print(f"[TASK_DEPEND]")
            print(f"Input is {data['task_depend'].x}")
            print(f"Output is {data['task_depend'].y}\n")

    output = model(data)

    if DO_POOLING:
        print(f"\nPredicted Output is {output}")

    else: 
        print(f"\nPredicted Output is ")
        for key,value in output.items():
            print(f"{key}: {value}")

    
