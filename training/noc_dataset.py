
import os 
import yaml 
import torch

from data.utils import load_graph_from_json

from torch.utils.data import Dataset    
from torch_geometric.data import HeteroData


class NocDataset(Dataset): 
    def __init__(self, training_data_dir): 

        self._file_dir              = training_data_dir  
        self._training_files        = os.listdir(training_data_dir)

        training_parameters         = yaml.safe_load(open("training/params.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]

    def __len__(self) -> int: 
        return len(self._training_files)

    def __getitem__(self, index: int):

        file_path  = os.path.join(self._file_dir, self._training_files[index])
        graph      = load_graph_from_json(file_path)

        data = HeteroData()

        task_input_feature  = []
        task_target         = []

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

            global_to_local_indexing[node_type][node_id] = len(global_to_local_indexing[node_type])

        # Creating the input and target tensors
        data["task"].x = torch.tensor( task_input_feature, dtype=torch.float )
        data["task"].y = torch.tensor( task_target, dtype=torch.float )

        # Creating fake/empty inputs for routers and pes
        num_routers         = len(global_to_local_indexing["router"])
        num_pes             = len(global_to_local_indexing["pe"])

        data["router"].x    = torch.empty( (num_routers, 0), dtype=torch.float )
        data["pe"].x        = torch.empty( (num_pes, 0), dtype=torch.float )

        # Creating the edge index tensor
        task_edge       = "depends_on"
        task_pe_edge    = "mapped_to"
        router_edge     = "link"
        router_pe_edge  = "interface"

        data["task",    task_edge,      "task"].edge_index     = [ [], [] ]
        data["task",    task_pe_edge,   "pe"].edge_index       = [ [], [] ]
        data["router",  router_edge,    "router"].edge_index   = [ [], [] ]
        data["router",  router_pe_edge, "pe"].edge_index       = [ [], [] ]
        data["pe",      router_pe_edge, "router"].edge_index   = [ [], [] ]

        for edge in graph.edges(data=True):

            src_node, dst_node, edge_data = edge

            src_type = graph.nodes[src_node]["type"]
            dst_type = graph.nodes[dst_node]["type"]

            if src_type == "task" and dst_type == "task": 
                edge_type = task_edge   
            elif src_type == "task" and dst_type == "pe": 
                edge_type = task_pe_edge
            elif src_type == "router" and dst_type == "router":
                edge_type = router_edge
            elif src_type == "router" and dst_type == "pe":
                edge_type = router_pe_edge
            elif src_type == "pe" and dst_type == "router":
                edge_type = router_pe_edge
            else:
                raise ValueError(f"Invalid edge type from {src_type} to {dst_type}")

            src_local_index = global_to_local_indexing[src_type][src_node]
            dst_local_index = global_to_local_indexing[dst_type][dst_node]  

            data[src_type, edge_type, dst_type].edge_index[0].append(src_local_index)
            data[src_type, edge_type, dst_type].edge_index[1].append(dst_local_index)

        for edge_type in data.edge_types: 
            data[edge_type].edge_index = torch.tensor(data[edge_type].edge_index, dtype=torch.long).contiguous()

        print(f"Data is {data}")

        self._do_checks(data)

        return data


    def _do_checks( self, data: HeteroData ) -> None:

        assert data.validate() is True, "Data is invalid"
        assert data.has_isolated_nodes() is False, "Data contains isolated nodes"
        assert data.has_self_loops() is False, "Data contains self loops"

if __name__ == "__main__": 

    from training.dataset import load_data

    # Dataset testing 
    dataset = NocDataset("data/training_data/simulator/test")
    print(f"Length of dataset is {len(dataset)}")
    dataset[0]

    # DataLoader testing
    dataloader = load_data( 
                    training_data_dir   = "data/training_data/simulator/test", 
                    batch_size          = 10, 
                    is_noc_dataset      = True)

    


