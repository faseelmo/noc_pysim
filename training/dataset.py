import os
import yaml
import json
import torch

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
        Args 
        1. is_hetero     : If True, the dataset will return a HeteroData object, else a Data object  
        2. has_wait_time : If True, the dataset will contain wait time feature for task_depend nodes.
                           This is only relevant for heterogenous graphs. In homogenous graphs, all nodes
                           have the same features. 
        3. return_graph  : If True, the __get__item__ will return the graph along with the data object. 
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

    def _homogenous_data(self, graph: nx.DiGraph, target_data: dict) -> Union[
            Data, 
            tuple[Data, tuple[dict, nx.DiGraph]]
        ]:
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
            return data


    def _heterogenous_data(self, graph: nx.DiGraph, target_data: dict) -> Union[
            HeteroData, 
            tuple[HeteroData, tuple[dict, nx.DiGraph]]
        ]:
        """
        Args: 
            1. graphs       : the directed graph of idx. From data/training_data/input/task_graph_idx.json 
            2. target_data  : the output features of the graph. From data/training_data/target/task_graph_idx.json

        returns a tuple of HeteroData and a dictionary containing global to local indexing

        The graph will have task and dependency nodes by default. 
        If has_wait_time is True, the graph will also have task_depend nodes.

        """
        hetero_data = HeteroData()

        task_input_feature          = []
        task_depend_input_feature   = []
        dependency_input_feature    = []

        task_target                 = [] # list of start and end cycle for each task
        task_depend_target          = [] # list of start and end cycle for each task_depend

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

                if node_type == "task_depend":
                    wait_time = node_data["wait_time"] / self.max_cycle
                    task_depend_input_feature.append([generate, processing_time, target_end_cycle])
                    task_depend_target.append([target_start_cycle, target_end_cycle])

                elif node_type == "task":
                    task_input_feature.append([generate, processing_time])
                    task_target.append([target_start_cycle, target_end_cycle])

            if node_type == "dependency":
                generate = node_data["generate"] / self.max_generate
                dependency_input_feature.append([generate])

            global_to_local_indexing[node_type][node_idx] = len(
                global_to_local_indexing[node_type])


        # Flag to check if data has no task node
        has_task_node = True
        if len(task_input_feature) == 0:
            has_task_node = False

        # Converting list of node features to tensor
        num_features_task_node = 2

        if has_task_node:
            hetero_data["task"].x = torch.tensor( task_input_feature, dtype=torch.float )
            hetero_data["task"].y = torch.tensor( task_target, dtype=torch.float )
        else: 
            hetero_data["task"].x = torch.empty( (0, num_features_task_node), dtype=torch.float )
            hetero_data["task"].y = torch.empty( (0, num_features_task_node), dtype=torch.float )

        hetero_data["dependency"].x = torch.tensor( dependency_input_feature, dtype=torch.float )

        if self.has_wait_time:
            hetero_data["task_depend"].x    = torch.tensor(task_depend_input_feature, dtype=torch.float)
            hetero_data["task_depend"].y    = torch.tensor(task_depend_target, dtype=torch.float)

        # Creating edge indices
        require_edge_type = "requires"

        hetero_data["task", require_edge_type, "task"].edge_index               = [[], []]

        if self.has_wait_time: 
            hetero_data["dependency", require_edge_type, "task_depend"].edge_index  = [[], []]
            hetero_data["task_depend", require_edge_type, "task_depend"].edge_index = [[], []]

            hetero_data["task_depend", require_edge_type, "task"].edge_index    = [[], []]
            hetero_data["task", require_edge_type, "task_depend"].edge_index    = [[], []]

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
                                                    hetero_data[edge_type].edge_index, 
                                                    dtype=torch.long
                                                ).contiguous()

        hetero_data = ToUndirected()(hetero_data)  # To leverage message passing in both directions
        
        hetero_data.y = float(target_data["latency"])
        self._do_checks(hetero_data)

        # [debugging] Uncomment to visualize the graph
        # from data.utils import visualize_graph
        # visualize_graph(graph=graph)    
        
        if self.return_graph:
            return (hetero_data, (global_to_local_indexing, graph))

        else: 
            return hetero_data



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

    print(f"\n[load_data] Is hetero graph: \t{is_hetero}")
    print(f"[load_data] Has wait time: \t{has_wait_time}")

    dataset             = CustomDataset(
                            training_data_dir=training_data_dir, 
                            is_hetero=is_hetero, 
                            has_wait_time=has_wait_time, 
                            return_graph=False)

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
    Usage: python3 -m training.dataset 10 True True
    Conditions to Test: 
        1. Homogenous 
            python3 -m training.dataset 10 False False

        2. Heterogenous
            a. Without wait time
                python3 -m training.dataset 10 True False
            b. With wait time
                python3 -m training.dataset 10 True True
    """

    import sys

    from data.utils import visualize_graph

    if len(sys.argv) > 1:   
        DATA_INDEX      = int(sys.argv[1])
        IS_HETERO       = sys.argv[2].lower() in ['true', '1']
        HAS_WAIT_TIME   = sys.argv[3].lower() in ['true', '1']

    else:                   
        DATA_INDEX      = 10
        IS_HETERO       = False
        HAS_WAIT_TIME   = False


    print( f"Data index is          {DATA_INDEX}" )
    print( f"Is heterogenous graph  {IS_HETERO}" )
    print( f"Has wait time is       {HAS_WAIT_TIME}" )

    DATASET_DIR = "data/training_data"

    dataset = CustomDataset(
                DATASET_DIR, 
                is_hetero       = IS_HETERO, 
                has_wait_time   = HAS_WAIT_TIME, 
                return_graph    = True)   

    def check_all_files_in_dataset():
        # Inside a def for scoping
        print(f"\n------Checking all files in the dataset------")
        list_of_edge_type = []

        for idx, (data, (index_dict, graph)) in enumerate(dataset):
        
            for src, dst, edge in graph.edges(data=True):
                src_type = graph.nodes[src]["type"]
                dst_type = graph.nodes[dst]["type"]
    
                edge_type = f"{src_type}_to_{dst_type}"
    
                if edge_type not in list_of_edge_type:
                    print(f"New edge type found: {edge_type}")
                    list_of_edge_type.append(edge_type)

        print(f"[Passed] All files are valid\n")

    # check_all_files_in_dataset()
    data, (index, graph) = dataset[DATA_INDEX]
    # visualize_graph(graph=graph)

    print(f"Data in index {DATA_INDEX} is \n{data}")

    print(f"\n\n----------------------DataLoader Test----------------------")

    BATCH_SIZE = 10
    print(f"\nLoading data with batch size {BATCH_SIZE}")

    train_loader, val_loader           = load_data(
                                                training_data_dir   = DATASET_DIR, 
                                                is_hetero           = IS_HETERO, 
                                                batch_size          = BATCH_SIZE,
                                                has_wait_time       = HAS_WAIT_TIME
                                        )

    print(f"\n For homogenous graph")
    print(f"Data from DataLoader            {next(iter(train_loader))}")
    print(f"DataLoader is                   {train_loader}")
    print(f"Number of training batches:     {len(train_loader)}")
    print(f"Batch size:                     {train_loader.batch_size}")
    print(f"Training samples                {len(train_loader) * train_loader.batch_size}")
    print(f"Number of validation batches:   {len(val_loader)}")
    


