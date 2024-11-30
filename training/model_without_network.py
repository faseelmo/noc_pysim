import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch_geometric.nn     import ( GraphConv, to_hetero )
from torch_geometric.data   import HeteroData, Data


class MPN(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels=None, num_conv_layers=3):
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

if __name__ == "__main__":

    """
    Usage: python3 -m training.model True False False False
    Conditions to Test: 
        1. Homogenous GNN Model
            python3 -m training.model 0 0 0 0 

        2. Heterogenous GNN Model
            python3 -m training.model 1 0 0 0 
            python3 -m training.model 1 0 1 0 (w/ wait time)
            python3 -m training.model 1 0 1 1 (w/ scheduler node and wait time)

        3. Heterogenous Pooling GNN Model
            [Works only for dataloader and not directly from CustomDataset. Issue with Batching]
            python3 -m training.model 1 1 0 0 
            python3 -m training.model 1 1 1 0
    """


    from training.dataset   import load_data
    from training.noc_dataset import NocDataset
    from training.utils     import print_parameter_count, initialize_model

    IDX             = 10
    BATCH_SIZE      = 1
    HIDDEN_CHANNELS = 50

    torch.manual_seed(0)

    dataloader, _ = load_data(
                        "data/training_data/simulator/test",
                        batch_size          = BATCH_SIZE,
                        use_noc_dataset    = True 
                        )

    data        = next(iter(dataloader))
    print(f"Data shape is {data.x_dict['task'].shape}, {data.x_dict['pe'].shape}, {data.x_dict['router'].shape}")

    # model       = HeteroGNN(HIDDEN_CHANNELS, num_mpn_layers=3)
    # output      = model(data)

    # print(f"Output shape is {output['task'].shape}, {output['pe'].shape}, {output['router'].shape}")
    # print_parameter_count(model)