
import os 
import re
import yaml 
import torch

from natsort import natsorted

from data.utils import load_graph_from_json

from torch.utils.data import Dataset    
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class NocDataset(Dataset): 
    def __init__(self, training_data_dir, classfiy_task_nodes: bool = False): 

        self._file_dir              = training_data_dir  
        self._training_files        = natsorted(os.listdir(training_data_dir))
        self._classify_task_nodes   = classfiy_task_nodes   

        training_parameters         = yaml.safe_load(open("training/config/params_with_network.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]

        self.return_graph           = False    

    def __len__(self) -> int: 
        return len(self._training_files)

    def get_graph(self, index: int):
        file_path = os.path.join(self._file_dir, self._training_files[index])
        graph = load_graph_from_json(file_path)
        return graph    

    def __getitem__(self, index: int):

        file_path  = os.path.join(self._file_dir, self._training_files[index])
        graph      = load_graph_from_json(file_path)

        # from data.utils import visualize_noc_application
        # visualize_noc_application(graph)

        data = HeteroData()

        dependency_feature      = []
        task_depend_feature     = []
        task_feature            = []

        task_target             = []
        task_depend_target      = []

        # Each node type has its own indexing
        global_to_local_indexing = { "router": {}, "pe": {}, "dependency": {}, "task": {}, "task_depend": {} } 

        for node_id, node_data in graph.nodes(data=True): 
            node_type = node_data["type"]
            if node_type == "task": 
                task_type       = node_data["task_type"]

                generate        = node_data["generate"] / self.max_generate 
                start_cycle     = node_data["start_cycle"] 
                end_cycle       = node_data["end_cycle"]

                processing_time = node_data["processing_time"] / self.max_processing_time

                if not self._classify_task_nodes:
                    task_feature.append([generate, processing_time])
                    task_target.append([start_cycle, end_cycle])

                else: 
                    if task_type == "dependency":
                        dependency_feature.append([generate])

                    elif task_type == "task_depend":
                        task_depend_feature.append([generate, processing_time])
                        task_depend_target.append([start_cycle, end_cycle])

                    elif task_type == "task":
                        task_feature.append([generate, processing_time])
                        task_target.append([start_cycle, end_cycle])   

                    else: 
                        raise ValueError(f"Invalid task type {task_type}")

                    node_type = task_type   

            global_to_local_indexing[node_type][node_id] = len(global_to_local_indexing[node_type])

        # Task (depend, task_depend, task) features
        # Creating the input and target tensors
        has_task_node = len(task_feature) > 0
        num_features_task_node = 2

        if has_task_node:
            data["task"].x = torch.tensor( task_feature, dtype=torch.float )
            data["task"].y = torch.tensor( task_target, dtype=torch.float ) 

        else: 
            data["task"].x = torch.empty( (0, num_features_task_node), dtype=torch.float )
            data["task"].y = torch.empty( (0, num_features_task_node), dtype=torch.float )

        if self._classify_task_nodes:
            data["dependency"].x = torch.tensor( dependency_feature, dtype=torch.float )
            data["task_depend"].x = torch.tensor( task_depend_feature, dtype=torch.float )
            data["task_depend"].y = torch.tensor( task_depend_target, dtype=torch.float )

        # Creating fake/empty inputs for routers and pes
        # This doesnt matter anyways since this feature is getting replaced with node embeddings
        num_elements = len(global_to_local_indexing["pe"])

        data["router"].x    = torch.ones( num_elements, 1, dtype=torch.float )
        data["pe"].x        = torch.ones( num_elements, 1, dtype=torch.float )

        # Creating the edge index tensor
        task_edge       = "generates_for"
        rev_task_edge   = "requires"

        task_pe_edge    = "mapped_to"
        pe_task_edge    = "rev_mapped_to"

        router_edge     = "link"

        router_pe_edge  = "interface"
        pe_router_edge  = "rev_interface"

        # NoC edges
        data["router",  router_edge,    "router"].edge_index   = [ [], [] ]
        data["router",  router_pe_edge, "pe"].edge_index       = [ [], [] ]
        data["pe",      pe_router_edge, "router"].edge_index   = [ [], [] ]

        # Task edge
        data["task", task_edge, "task"].edge_index     = [ [], [] ] 
        data["task", rev_task_edge, "task"].edge_index = [ [], [] ] 
        if self._classify_task_nodes:
            data["task",        task_edge, "task_depend"].edge_index    = [ [], [] ]
            data["task_depend", task_edge, "task"].edge_index           = [ [], [] ]
            data["task_depend", task_edge, "task_depend"].edge_index    = [ [], [] ]
            data["dependency",  task_edge, "task_depend"].edge_index    = [ [], [] ]

        # Map edges
        data["task", task_pe_edge, "pe"].edge_index   = [ [], [] ]
        data["pe",   pe_task_edge, "task"].edge_index = [ [], [] ]
        
        if self._classify_task_nodes:
            data["task_depend", task_pe_edge, "pe"].edge_index = [ [], [] ]
            data["dependency",  task_pe_edge, "pe"].edge_index = [ [], [] ]

            data["pe", pe_task_edge, "task_depend"].edge_index  = [ [], [] ]   
            data["pe", pe_task_edge, "dependency"].edge_index   = [ [], [] ]

        for edge in graph.edges(data=True):

            src_node, dst_node, _ = edge
            src_type = graph.nodes[src_node]["type"]
            dst_type = graph.nodes[dst_node]["type"]

            do_rev = False

            if src_type == "task" and dst_type == "task": 
                edge_type       = task_edge   
                rev_edge_type   = rev_task_edge 
                do_rev          = True
                
                if self._classify_task_nodes:
                    src_type = graph.nodes[src_node]["task_type"]
                    dst_type = graph.nodes[dst_node]["task_type"]

                

            elif src_type == "task" and dst_type == "pe": 
                edge_type       = task_pe_edge
                rev_edge_type   = pe_task_edge
                do_rev          = True

                if self._classify_task_nodes:
                    src_type = graph.nodes[src_node]["task_type"]

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

            if do_rev : 
                # Reversee edge for task-pe and task-task edges
                data[dst_type, rev_edge_type, src_type].edge_index[0].append(dst_local_index)
                data[dst_type, rev_edge_type, src_type].edge_index[1].append(src_local_index)   

        for edge_type in data.edge_types: 
            data[edge_type].edge_index = torch.tensor(data[edge_type].edge_index, dtype=torch.long).contiguous()

        # data = ToUndirected()(data) 
        self._do_checks(data)

        if self.return_graph:
            return data, graph

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
    from training.model_with_network import  HeteroGNN
    from data.utils import visualize_noc_application

    import random 
    import torch
    random.seed(0)
    torch.manual_seed(0)

    idx = 3 # 3 has all three task types 

    # Dataset testing 
    dataset = NocDataset("data/training_data/with_network/test")
    print(f"Length of dataset is {len(dataset)}")
    data = dataset[idx]  

    print(f"X is \n{data.x_dict}")
    
    for key, value in data.edge_index_dict.items(): 
        print(f"{key} edge index is \n{value}")




