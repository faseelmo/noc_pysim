import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv, Linear 

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_mpn_layers: int, classfiy_task_nodes: bool = False): 
        super().__init__()

        assert num_mpn_layers >= 2, "Number of layers should be at least 2."
        self._classify_task_nodes = classfiy_task_nodes 

        projection_size         = 16
        self._pe_embedding      = nn.Embedding(9, projection_size)
        self._router_embedding  = nn.Embedding(9, projection_size)

        self._convs     = nn.ModuleList()
        self._conv_aggr = ["sum"] # ["sum", "mean", "max", "min"]

        for _ in range(num_mpn_layers):
            intermediate_convs = self._get_hetero_conv(-1, hidden_channels)
            self._convs.extend(intermediate_convs)

        self._feedforward_task          = Linear(hidden_channels, 2)  
        # self._feedforward_task_depend   = Linear(hidden_channels, 2)


    def _get_hetero_conv(self, in_channels, out_channels): 
        
        aggr_list = ["mean", ] # ["sum", "mean", "max", "min"]
        conv_list = []

        classified_task_nodes = {
            ("dependency",  "generates_for", "task_depend") : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("task_depend", "generates_for", "task")        : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("task",        "generates_for", "task_depend") : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("pe",          "rev_mapped_to", "task_depend") : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("pe",          "rev_mapped_to", "dependency")  : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("task_depend", "mapped_to",     "pe")          : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("dependency",  "mapped_to",     "pe")          : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
        }

        noc_nodes = {
            ("task",    "generates_for", "task")   : GraphConv(in_channels, out_channels, aggr="max"),
            ("task",    "requires", "task")        : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("task",    "mapped_to",     "pe")     : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("pe",      "rev_mapped_to", "task")   : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
            ("router",  "link",          "router") : GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
            ("router",  "interface",     "pe")     : GraphConv(in_channels, out_channels, aggr=self._conv_aggr), 
            ("pe",      "rev_interface", "router") : GraphConv(in_channels, out_channels, aggr=self._conv_aggr),
        }

        if self._classify_task_nodes:
            hetero_dict = {**classified_task_nodes, **noc_nodes}
        else: 
            hetero_dict = noc_nodes

        for aggr in aggr_list:

            conv = HeteroConv(hetero_dict, aggr=aggr)
            conv_list.append(conv)

        return conv_list

    def forward(self, data: HeteroData) -> HeteroData:
        x_dict              = data.x_dict
        edge_index_dict     = data.edge_index_dict

        batch_size = x_dict['pe'].size(0) // 9

        x_dict['pe']        = self._pe_embedding.weight.repeat(batch_size, 1)
        x_dict['router']    = self._router_embedding.weight.repeat(batch_size, 1)

        for conv in self._convs:
            x_dict = conv(x_dict, edge_index_dict)
            
            for key, x in x_dict.items():
                x_dict[key] = x.relu()

        x_dict['task'] = self._feedforward_task(x_dict['task'])

        if self._classify_task_nodes:
            x_dict['task_depend'] = self._feedforward_task(x_dict['task_depend'])

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


    from training.dataset       import load_data
    from training.noc_dataset   import NocDataset
    from training.utils         import print_parameter_count, initialize_model

    IDX             = 10
    BATCH_SIZE      = 1
    HIDDEN_CHANNELS = 64

    classifiy_task_nodes = False    

    torch.manual_seed(0)

    dataloader, _ = load_data(
                        "data/training_data/with_network/test",
                        batch_size          = BATCH_SIZE,
                        use_noc_dataset     = True ,
                        classify_task_nodes = classifiy_task_nodes )

    data        = next(iter(dataloader))

    model  = HeteroGNN( HIDDEN_CHANNELS, num_mpn_layers=5, classfiy_task_nodes=classifiy_task_nodes )
    output = model(data)

    if classifiy_task_nodes:
        print(f"Task node classification output is {output['task'].shape}")
        print(f"Task depend node classification output is {output['task_depend'].shape}")
    else: 
        print(f"Output is {output['task'].shape}")

    print_parameter_count(model)