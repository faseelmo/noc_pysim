import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch_geometric.nn     import (
                                global_max_pool,
                                SAGEConv,
                                GraphConv,
                                to_hetero,
                                HeteroConv, 
                                GATConv, 
                                Linear, 
                                Sequential
                            )

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_mpn_layers: int): 
        super().__init__()

        assert num_mpn_layers >= 2, "Number of layers should be at least 2."

        self._convs = nn.ModuleList()

        for _ in range(num_mpn_layers - 1):
            intermediate_conv = self._get_hetero_conv(-1, hidden_channels, aggr="sum")
            self._convs.append(intermediate_conv)

        final_conv = self._get_hetero_conv(hidden_channels, 2, aggr="sum")
        self._convs.append(final_conv)

        projection_size         = 32
        self.pe_embedding       = nn.Embedding(9, projection_size)
        self.router_embedding   = nn.Embedding(9, projection_size)
        self._pos_project       = Linear(2, projection_size)


    def _get_hetero_conv(self, in_channels, out_channels, aggr): 
        conv = HeteroConv({
            ("task", "depends_on", "task"):         GraphConv(in_channels, out_channels, aggr="max"),
            ("task", "rev_depends_on", "task"):     GraphConv(in_channels, out_channels, aggr="add"),
            ("task", "mapped_to", "pe"):            GraphConv(in_channels, out_channels, aggr="add"), 
            ("pe", "rev_mapped_to", "task"):        GraphConv(in_channels, out_channels, aggr="add"), 
            ("router", "link", "router"):           GraphConv(in_channels, out_channels, aggr="add"), 
            ("router", "interface", "pe"):          GraphConv(in_channels, out_channels, aggr="add"), 
            ("pe", "rev_interface", "router"):      GraphConv(in_channels, out_channels, aggr="add"),
        }, aggr=aggr)

        return conv

    def forward(self, data):
        x_dict              = data.x_dict
        edge_index_dict     = data.edge_index_dict

        batch_size = x_dict['pe'].size(0) // 9

        # x_dict['pe']        = self.pe_embedding.weight.repeat(batch_size, 1)
        x_dict['router']    = self.router_embedding.weight.repeat(batch_size, 1)

        # print(f"\nData is {x_dict}")

        for conv in self._convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            
            for key, x in x_dict.items():
                x_dict[key] = x.relu()

        x_dict = self._convs[-1](x_dict, edge_index_dict)

        return x_dict



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


    IDX             = 10
    BATCH_SIZE      = 1
    HIDDEN_CHANNELS = 40

    torch.manual_seed(0)

    dataloader, _ = load_data(
                        "data/training_data/simulator/train",
                        batch_size          = BATCH_SIZE,
                        use_noc_dataset    = True 
                        )

    data        = next(iter(dataloader))
    print(f"Data is {data.x_dict}")

    model       = HeteroGNN(HIDDEN_CHANNELS, num_mpn_layers=3)
    output      = model(data)
