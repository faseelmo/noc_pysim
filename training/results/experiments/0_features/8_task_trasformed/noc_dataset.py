
import os 
import re
import yaml 
import torch

from natsort import natsorted

from data.utils import load_graph_from_json, visualize_application

from torch.utils.data import Dataset    
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class NocDataset(Dataset): 
    def __init__(self, training_data_dir): 

        self._file_dir              = training_data_dir  
        self._training_files        = natsorted(os.listdir(training_data_dir))

        training_parameters         = yaml.safe_load(open("training/params.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]

    def __len__(self) -> int: 
        return len(self._training_files)

    def __getitem__(self, index: int):

        file_path  = os.path.join(self._file_dir, self._training_files[index])
        graph      = load_graph_from_json(file_path)

        data = HeteroData()

        task_input_feature      = []
        task_target             = []
        router_input_feature    = []

        # Each node type has its own indexing
        global_to_local_indexing = { "task": {}, "router": {}, "pe": {} } 

        for node_id, node_data in graph.nodes(data=True): 
            node_type = node_data["type"]
            if node_type == "task": 
                generate        = node_data["generate"] / self.max_generate
                processing_time = node_data["processing_time"] / self.max_processing_time
                start_cycle     = node_data["start_cycle"] 
                end_cycle       = node_data["end_cycle"]

                task_input_feature.append([generate, processing_time])
                task_target.append([start_cycle, end_cycle])    

            elif node_type == "router":
                x_pos,y_pos     = self._extract_coordinates(node_id)
                norm_x_pos      = x_pos / 3 
                norm_y_pos      = y_pos / 3
                router_input_feature.append([norm_x_pos, norm_y_pos])

            global_to_local_indexing[node_type][node_id] = len(global_to_local_indexing[node_type])

        # Creating the input and target tensors
        data["task"].x = torch.tensor( task_input_feature, dtype=torch.float )
        data["task"].y = torch.tensor( task_target, dtype=torch.float )

        # Creating fake/empty inputs for routers and pes
        num_routers         = len(global_to_local_indexing["router"])
        num_pes             = len(global_to_local_indexing["pe"])

        data["router"].x    = torch.ones( num_routers, 1, dtype=torch.float )
        # data["router"].x    = torch.tensor( router_input_feature, dtype=torch.float )
        # data["router"].x    = torch.ones( num_pes, 1, dtype=torch.float )
        data["pe"].x        = torch.ones( num_pes, 1, dtype=torch.float )

        # Creating the edge index tensor
        task_edge       = "depends_on"
        rev_task_edge   = "rev_depends_on"
        task_pe_edge    = "mapped_to"
        router_edge     = "link"
        router_pe_edge  = "interface"

        pe_task_edge    = "rev_mapped_to"
        pe_router_edge  = "rev_interface"

        data["task",    task_edge,      "task"].edge_index     = [ [], [] ]
        data["task",    rev_task_edge,  "task"].edge_index     = [ [], [] ]
        data["task",    task_pe_edge,   "pe"].edge_index       = [ [], [] ]
        data["pe",      pe_task_edge,   "task"].edge_index     = [ [], [] ]
        data["router",  router_edge,    "router"].edge_index   = [ [], [] ]
        data["router",  router_pe_edge, "pe"].edge_index       = [ [], [] ]
        data["pe",      pe_router_edge, "router"].edge_index   = [ [], [] ]

        for edge in graph.edges(data=True):

            src_node, dst_node, edge_data = edge

            src_type = graph.nodes[src_node]["type"]
            dst_type = graph.nodes[dst_node]["type"]

            if src_type == "task" and dst_type == "task": 
                edge_type       = task_edge   
                rev_edge_type   = rev_task_edge

            elif src_type == "task" and dst_type == "pe": 
                edge_type       = task_pe_edge
                rev_edge_type   = pe_task_edge

            elif src_type == "router" and dst_type == "router":
                edge_type = router_edge

            elif src_type == "router" and dst_type == "pe":
                edge_type = router_pe_edge

            elif src_type == "pe" and dst_type == "router":
                edge_type = pe_router_edge  

            else:
                raise ValueError(f"Invalid edge type from {src_type} to {dst_type}")

            src_local_index = global_to_local_indexing[src_type][src_node]
            dst_local_index = global_to_local_indexing[dst_type][dst_node]  

            data[src_type, edge_type, dst_type].edge_index[0].append(src_local_index)
            data[src_type, edge_type, dst_type].edge_index[1].append(dst_local_index)

            if ( edge_type == task_pe_edge ) or ( edge_type == task_edge ) : 
                # Reversee edge for task-pe and task-task edges
                data[dst_type, rev_edge_type, src_type].edge_index[0].append(dst_local_index)
                data[dst_type, rev_edge_type, src_type].edge_index[1].append(src_local_index)   


        for edge_type in data.edge_types: 
            data[edge_type].edge_index = torch.tensor(data[edge_type].edge_index, dtype=torch.long).contiguous()

        # data = ToUndirected()(data) 
        self._do_checks(data)

        return data

    def _extract_coordinates(self, element_str: str) -> tuple: 
        x, y = tuple(map(int, re.findall(r'\d+', element_str)))
        return x, y


    def _do_checks( self, data: HeteroData ) -> None:

        assert data.validate() is True, "Data is invalid"
        assert data.has_isolated_nodes() is False, "Data contains isolated nodes"
        assert data.has_self_loops() is False, "Data contains self loops"

if __name__ == "__main__": 

    from training.dataset import load_data
    from training.model import  HeteroGNN

    import random 
    import torch
    random.seed(0)
    torch.manual_seed(0)

    # Dataset testing 
    dataset = NocDataset("data/training_data/simulator/test")
    print(f"Length of dataset is {len(dataset)}")
    data = dataset[0]

    # print(f"GNNHetero is ")
    # model = GNNHetero(hidden_channels=3, num_mpn_layers=3, metadata=dataset[0].metadata())
    # print(model)
    # output = model(data)
    # print(output['task'])

    # print(f"HeteroConv is ")
    # model = HeteroGNN(hidden_channels=3, num_mpn_layers=3)
    # print(model)
    # output = model(data)
    # print(output['task'])


    # print(f"X dict is \n{data.x_dict}")

    # print("\nEdge index dict is:")
    # for edge_type, edge_index in data.edge_index_dict.items():
    #     print(f"{edge_type}:")
    #     print(edge_index)

    
    # DataLoader testing
    # print(f"\n\nDataLoader testing")
    # dataloader, _   = load_data( 
    #                     training_data_dir   = "data/training_data/simulator/test", 
    #                     batch_size          = 2, 
    #                     use_noc_dataset     = True)

    
    # metadata    = dataset[0].metadata()
    # print(f"Metadata is {metadata}")
    # data        = next(iter(dataloader))
    # model       = GNNHetero(hidden_channels=3, num_mpn_layers=3, metadata=metadata)
    # print(model)
    # initialize_model(model, dataloader)

    # model = HeteroGNN(hidden_channels=3, num_mpn_layers=3)
    # model(data)






