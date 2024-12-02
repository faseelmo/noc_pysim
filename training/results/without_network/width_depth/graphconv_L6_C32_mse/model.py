
import importlib 

import torch
import torch.nn as nn

from torch_geometric.nn import to_hetero

class MPN(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels, model_str, num_conv_layers=3):
        super().__init__()
        torch.manual_seed(0)

        assert num_conv_layers >= 2, "Number of convolution layers should be at least 2."

        if model_str.lower() == "gcn":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "GCNConv")

        elif model_str.lower() == "graphconv":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "GraphConv")

        elif model_str.lower() == "gat":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "GATv2Conv")

        elif model_str.lower() == "sage":
            module = importlib.import_module("torch_geometric.nn")
            CONV = getattr(module, "SAGEConv")

        else: 
            raise ValueError(f"Model {model_str} not found.")

        self.conv_list = nn.ModuleList()
        self.conv_list.append(CONV(-1, hidden_channels))

        for _ in range(num_conv_layers - 2):
            self.conv_list.append(CONV(hidden_channels, hidden_channels))

        self.conv_list.append(CONV(hidden_channels, output_channels))

    def forward(self, data):
        x           = data.x
        edge_index  = data.edge_index   
        for conv in self.conv_list[:-1]:
            x = conv(x, edge_index).relu()

        x = self.conv_list[-1](x, edge_index)
        
        return x

class MPNHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_mpn_layers, metadata):
        super().__init__()

        num_output_features     = 2
        mpn                     = MPN(hidden_channels, num_output_features, num_mpn_layers)
        self.mpn_hetero         = to_hetero(mpn, metadata, aggr="sum")


    def forward(self, data):
        x_dict          = data.x_dict
        edge_index_dict = data.edge_index_dict

        out_dict = self.mpn_hetero(x_dict, edge_index_dict)

        return out_dict
