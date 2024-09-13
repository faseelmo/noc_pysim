import os
import json
import torch
import yaml

import networkx                 as nx

from natsort                    import natsorted
from typing                     import Union

from torch_geometric.utils      import from_networkx
from torch.utils.data           import Dataset, random_split
from torch_geometric.loader     import DataLoader
from torch_geometric.data       import HeteroData, Data
from torch_geometric.transforms import ToUndirected


class CustomDataset(Dataset):
    def __init__(self, training_data_dir, is_hetero=False, has_wait_time=False, return_graph=False):
        """
        Args. 
        1. is_hetero     : If True, the dataset will return a HeteroData object, else a Data object  
        2. has_wait_time : If True, the dataset will contain wait time feature for task_depend nodes.
                           This is only relevant for heterogenous graphs. In homogenous graphs, all nodes
                           have the same features. 
        3. return_graph  : If True, the dataset will return the graph along with the data object. 
                           Useful for visualization.  
        """
        self.is_hetero              = is_hetero
        self.has_wait_time          = has_wait_time
        self.return_graph           = return_graph
        self.input_dir              = os.path.join(training_data_dir, "input")
        self.target_dir             = os.path.join(training_data_dir, "target")

        self.input_files            = natsorted(os.listdir(self.input_dir))
        self.output_files           = natsorted(os.listdir(self.target_dir))

        training_parameters         = yaml.safe_load(open("training/params.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]
        self.max_cycle              = training_parameters["MAX_CYCLE"]

        assert len(self.input_files) == len(
            self.output_files), f"Number of input files and output files must be the same. "
        f"{len(self.input_files)} != {len(self.output_files)}"

    def __len__(self):
        return len(self.input_files)

    def get_file_path(self, index) -> tuple[str, str]:

        input_file_path     = os.path.join(self.input_dir, self.input_files[index])
        target_file_path    = os.path.join(self.target_dir, self.output_files[index])

        return input_file_path, target_file_path

    def __getitem__(self, index):
        
        input_file_path, target_file_path = self.get_file_path(index)

        input_data  = self._load_json(input_file_path)
        target_data = self._load_json(target_file_path)

        graph = nx.node_link_graph(input_data)

        if self.is_hetero:
            return self._heterogenous_data(graph, target_data)
        else:
            return self._homogenous_data(graph, target_data)

    def _load_json(self, filename : str) -> dict: 
        with open(filename, "r") as file:
            data = json.load(file)
        return data

    def _homogenous_data(self, graph: nx.DiGraph, target_data: dict) -> tuple[Data, dict]:
        """
        Args: 
            1. graphs       : the directed graph of idx. From data/training_data/input/task_graph_idx.json 
            2. target_data  : the output features of the graph. From data/training_data/target/task_graph_idx.json

        returns a tuple of Data and an empty dictionary (for compatibility with heterogenous data)
        """
        data = from_networkx(graph)

        # data.x should contain all the node features with shape [num_nodes, num_node_features]
        data.x = torch.stack((data.generate, data.processing_time), dim=1).float() / 10 # Hardcoded normalization

        del data.generate
        del data.processing_time

        # data.edge_attr should contain all the edge features with shape [num_edges, num_edge_features]
        data.edge_attr = torch.stack((data.weight,), dim=1).float()
        del data.weight
        del data.type

        data.y = float(target_data["latency"])
        self._do_checks(data)

        if self.return_graph:
            return (data, ({}, graph))

        else: 
            return data, {}


    def _heterogenous_data(self, graph: nx.DiGraph, target_data: dict) -> Union[tuple[HeteroData, dict], tuple[HeteroData, tuple[dict, nx.DiGraph]]]:
        """
        Args: 
            1. graphs       : the directed graph of idx. From data/training_data/input/task_graph_idx.json 
            2. target_data  : the output features of the graph. From data/training_data/target/task_graph_idx.json

        returns a tuple of HeteroData and a dictionary containing global to local indexing
        """
        hetero_data = HeteroData()

        task_input_feature          = []
        task_depend_input_feature   = []
        dependency_input_feature    = []

        task_target                 = [] # list of start and end cycle for each task

        global_to_local_indexing    = {"task": {}, "dependency": {}}

        if self.has_wait_time:
            global_to_local_indexing["task_depend"] = {}    

        # Creating node features + global to local indexing
        for node_idx, node_data in graph.nodes(data=True):
            node_type = node_data["type"]

            if node_type == "task_depend" and not self.has_wait_time:
                # if wait time is not present, treat task_depend as task
                node_type = "task"

            if node_type not in hetero_data:
                hetero_data[node_type].x = []

            if node_type == "task" or node_type == "task_depend":
                generate = node_data["generate"] / self.max_generate

                processing_time = (
                    node_data["processing_time"] / self.max_processing_time)

                task_target_feature = target_data[str(node_idx)]
                target_start_cycle  = task_target_feature["start_cycle"] / self.max_cycle
                target_end_cycle    = task_target_feature["end_cycle"] / self.max_cycle

                # target is the same for both task and task_depend nodes
                task_target.append([target_start_cycle, target_end_cycle])

                if node_type == "task_depend":
                    wait_time = node_data["wait_time"] / self.max_cycle
                    task_depend_input_feature.append([generate, processing_time, wait_time])

                elif node_type == "task":
                    task_input_feature.append([generate, processing_time])

            if node_type == "dependency":
                generate = node_data["generate"] / self.max_generate
                dependency_input_feature.append([generate])

            global_to_local_indexing[node_type][node_idx] = len(
                global_to_local_indexing[node_type])

        # Converting list of node features to tensor
        hetero_data["task"].x           = torch.tensor(task_input_feature, dtype=torch.float)
        hetero_data["task"].y           = torch.tensor(task_target, dtype=torch.float)

        hetero_data["dependency"].x     = torch.tensor(dependency_input_feature, dtype=torch.float)

        if self.has_wait_time:
            hetero_data["task_depend"].x    = torch.tensor(task_depend_input_feature, dtype=torch.float)
            hetero_data["task_depend"].y    = torch.tensor(task_target, dtype=torch.float)

        # Creating edge indices
        require_edge_type = "requires"
        hetero_data["task", require_edge_type, "task"].edge_index                   = [[], []]

        if self.has_wait_time: 
            hetero_data["dependency", require_edge_type, "task_depend"].edge_index  = [[], []]
            hetero_data["task_depend", require_edge_type, "task"].edge_index        = [[], []]

        else: 
            hetero_data["dependency", require_edge_type, "task"].edge_index         = [[], []]

        for edge in graph.edges(data=True):
            src, dst, _ = edge

            src_type    = graph.nodes[src]["type"]
            dst_type    = graph.nodes[dst]["type"]
            
            if not self.has_wait_time: 
                # Converting task_depend to task if wait_time feature is not required
                src_type = "task" if src_type == "task_depend" else src_type
                dst_type = "task" if dst_type == "task_depend" else dst_type 

            hetero_data[src_type, require_edge_type, dst_type].edge_index[0].append(
                global_to_local_indexing[src_type][src])

            hetero_data[src_type, require_edge_type, dst_type].edge_index[1].append(
                global_to_local_indexing[dst_type][dst])

        # Converting edge indices [list] to tensors
        for edge_type in hetero_data.edge_types:
            hetero_data[edge_type].edge_index = torch.tensor(
                hetero_data[edge_type].edge_index, dtype=torch.long
            ).contiguous()

        hetero_data = ToUndirected()(hetero_data)  # To leverage message passing in both directions
        
        hetero_data.y = float(target_data["latency"])
        self._do_checks(hetero_data)

        
        if self.return_graph:
            return (hetero_data, (global_to_local_indexing, graph))

        else: 
            return (hetero_data, global_to_local_indexing)

    def _do_checks(self, data: Union[Data, HeteroData]) -> None:
        assert data.validate() is True, "Data is invalid"
        assert data.has_isolated_nodes() is False, "Data contains isolated nodes"
        assert data.has_self_loops() is False, "Data contains self loops"
        # assert data.is_directed() is True, "Data is not directed"


def load_data(
        training_data_dir, 
        is_hetero: bool, 
        has_wait_time: bool, 
        batch_size: int =32, 
        validation_split: float =0.1
    ) -> tuple[DataLoader, DataLoader]:

    print(f"[load_data] Is heterogenous graph: {is_hetero}")
    print(f"[load_data] Has wait time: {has_wait_time}")

    dataset             = CustomDataset(training_data_dir, is_hetero, has_wait_time)
    validation_size     = int(validation_split * len(dataset))

    if validation_size == 0: # ensure at least one validation sample
        validation_size = 1  

    train_size = len(dataset) - validation_size

    if train_size <= 0:
        raise ValueError("Training dataset size is too small after splitting.")

    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

    print(f"\nTraining dataset size: \t\t{len(train_dataset)}")
    print(f"Validation dataset size: \t{len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


if __name__ == "__main__":

    """
    Usage: python3 -m training.dataset 10, True
    """

    import sys

    from data.utils import visualize_graph, get_compute_list_from_json

    if len(sys.argv) > 1:   
        DATA_INDEX      = int(sys.argv[1])
        HAS_WAIT_TIME   = sys.argv[2].lower() in ['true', '1']

    else:                   
        DATA_INDEX      = 2
        HAS_WAIT_TIME   = False

    print( f"Data index is {DATA_INDEX}" )
    print( f"Has wait time is {HAS_WAIT_TIME}" )

    DATASET_DIR = "data/training_data"

    print(f"\n\n----------------------Homogenous Graph----------------------")

    is_hetero                   = False
    homogenous_dataset          = CustomDataset(DATASET_DIR, is_hetero=False, return_graph=True, has_wait_time=HAS_WAIT_TIME)
    data, (index_dict, graph)   = homogenous_dataset[DATA_INDEX]

    target_path     = homogenous_dataset.get_file_path(DATA_INDEX)[1]
    compute_list    = get_compute_list_from_json(target_path)

    print(f"Data is {data}")
    print(f"Feature matrix \n{data.x}\n")
    print(f"\nedge_index \n{data.edge_index}\n")
    print(f"edge_attr \n{data.edge_attr}\n")

    visualize_graph(graph=graph, compute_list=compute_list)

    print(f"\n\n----------------------Heterogenous Graph----------------------")

    is_hetero                           = True
    heterogenous_dataset                = CustomDataset(DATASET_DIR, is_hetero=True ,return_graph=True, has_wait_time=HAS_WAIT_TIME)
    hetero_data, (index_dict, graph)    = heterogenous_dataset[DATA_INDEX]

    print(f"\nHeteroData is {hetero_data}")
    print(f"\nOutput feature matrix \n{hetero_data['task'].y}")
    print(f"\nLatency is {torch.max(hetero_data['task'].y) * 100}")
    print(f"Latency is {hetero_data.y}")
    print(f"\nNode types are {hetero_data.node_types}")
    print(f"Edge types are {hetero_data.edge_types}")
    print(f"Meta information is {hetero_data.metadata}")

    for node_type in hetero_data.node_types:
        print(
            f"Node type {node_type} has feature "
            f"matrix \n{hetero_data[node_type].x}\n")

    for edge_type in hetero_data.edge_types:
        print(
            f"Edge type {edge_type} has edge index "
            "\n{hetero_data[edge_type].edge_index}\n")

    visualize_graph(graph=graph, compute_list=compute_list)


    print(f"\n\n----------------------DataLoader Test----------------------")

    homo_train_loader, val_loader           = load_data(
                                                DATASET_DIR, 
                                                is_hetero=False, 
                                                batch_size=32,
                                                has_wait_time=HAS_WAIT_TIME
                                            )

    hetero_train_loader, hetero_val_loader  = load_data(
                                                DATASET_DIR, 
                                                is_hetero=True, 
                                                batch_size=32,
                                                has_wait_time=HAS_WAIT_TIME
                                            )

    print(f"\n For homogenous graph")
    print(f"\nData loader information")
    print(f"Number of training batches:     {len(homo_train_loader)}")
    print(f"Batch size:                     {homo_train_loader.batch_size}")
    print(f"Training samples                {len(homo_train_loader) * homo_train_loader.batch_size}")
    print(f"Number of validation batches:   {len(val_loader)}")


    print(f"\n For homogenous graph")
    print(f"\nData loader information")
    print(f"Number of training batches:     {len(hetero_train_loader)}")
    print(f"Batch size:                     {hetero_train_loader.batch_size}")
    print(f"Training samples                {len(hetero_train_loader) * hetero_train_loader.batch_size}")
    print(f"Number of validation batches:   {len(hetero_val_loader )}")
