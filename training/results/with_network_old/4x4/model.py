import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch_geometric.nn     import (
                                GraphConv,
                                HeteroConv, 
                                Linear, 
                            )

from torch_geometric.data import HeteroData

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_mpn_layers: int, mesh_size: int = 3): 
        super().__init__()

        assert num_mpn_layers >= 2, "Number of layers should be at least 2."

        self._num_elems         = mesh_size**2

        projection_size         = 16
        self._pe_embedding      = nn.Embedding(self._num_elems, projection_size)
        self._router_embedding  = nn.Embedding(self._num_elems, projection_size)

        self._convs     = nn.ModuleList()
        self._conv_aggr = ["sum"] # ["sum", "mean", "max", "min"]

        for _ in range(num_mpn_layers-1):
            intermediate_convs = self._get_hetero_conv(-1, hidden_channels)
            self._convs.extend(intermediate_convs)

        self._final_conv = self._get_hetero_conv(-1, 2)[0]
        # self._feedforward = Linear(hidden_channels, 2)  


    def _get_hetero_conv(self, in_channels, out_channels): 
        
        aggr_list = ["mean", "max"] # ["sum", "mean", "max", "min"]
        conv_list = []

        for aggr in aggr_list:

            conv = HeteroConv({
                ("task", "generates_for", "task"): GraphConv(in_channels, out_channels, aggr="max"),
                ("task", "requires", "task"):      GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
                ("task", "mapped_to", "pe"):       GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("pe", "rev_mapped_to", "task"):   GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("router", "link", "router"):      GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("router", "interface", "pe"):     GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
                ("pe", "rev_interface", "router"): GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            }, aggr=aggr)

            conv_list.append(conv)

        return conv_list

    def forward(self, x_dict, edge_index_dict) -> HeteroData:

        batch_size = x_dict['pe'].size(0) // self._num_elems

        x_dict['pe']        = self._pe_embedding.weight.repeat(batch_size, 1)
        x_dict['router']    = self._router_embedding.weight.repeat(batch_size, 1)

        for conv in self._convs:


            x_dict = conv(x_dict, edge_index_dict)
            
            for key, x in x_dict.items():
                x_dict[key] = x.relu()

        x_dict = self._final_conv(x_dict, edge_index_dict)
        # x_dict['task'] = self._feedforward(x_dict['task'])

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
    from torch_scatter      import scatter_max

    IDX             = 10
    BATCH_SIZE      = 1
    HIDDEN_CHANNELS = 3
    MESH_SIZE       = 4

    torch.manual_seed(0)

    dataloader, _ = load_data( "data/training_data/with_network_4x4/test",
                                batch_size      = BATCH_SIZE,
                                use_noc_dataset = True )

    data        = next(iter(dataloader))
    print(f"Data shape is {data.x_dict['task'].shape}, {data.x_dict['pe'].shape}, {data.x_dict['router'].shape}")

    model       = HeteroGNN(HIDDEN_CHANNELS, num_mpn_layers=3, mesh_size=MESH_SIZE)
    output      = model(data.x_dict, data.edge_index_dict)

    # target_task = data['task'].y
    # batch = data['task'].batch  
    # _, max_indices = scatter_max(target_task[:, 1], batch)

    # print(f"indicees is {max_indices}")
    
    # print(f"Target is {data['task'].y}")
    # print(f"Target max is {data['task'].y[max_indices, 1]}")

    # print(f"Output is {output['task']}")
    # print(f"Output max is {output['task'][max_indices, 1]}")

    # abs_error = torch.abs(output['task'][max_indices, 1] - data['task'].y[max_indices, 1])


