import os
import yaml
import json
import torch

import networkx                 as nx

from typing                     import Union
from natsort                    import natsorted

from torch_geometric.loader     import DataLoader
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils      import from_networkx
from torch_geometric.data       import HeteroData, Data
from torch.utils.data           import Dataset, random_split

from data.utils                 import load_graph_from_json


class CustomDataset(Dataset):
    def __init__(self, training_data_dir, is_hetero=False, has_scheduler_node=False, return_graph=False):
        """
        Args 
        1. is_hetero            : If True, the dataset will return a HeteroData object, else a Data object  
        2. has_scheduler_node   : If True, the dataset will connect all task nodes ('task' and 'task_depend')  
                                  to one single node.   
        3. return_graph         : If True, the __get__item__ will return the graph along with the data object. 
                                  Useful for visualization.  
        """
        self._is_hetero             = is_hetero
        self._has_scheduler_node    = has_scheduler_node
        self._return_graph          = return_graph

        self._file_dir              = training_data_dir
        self._training_files        = natsorted(os.listdir(training_data_dir))

        training_parameters         = yaml.safe_load(open("training/params_without_network.yaml"))
        self.max_generate           = training_parameters["MAX_GENERATE"]
        self.max_processing_time    = training_parameters["MAX_PROCESSING_TIME"]
        self.max_cycle              = training_parameters["MAX_CYCLE"]

    def __len__(self):
        return len(self._training_files)

    def __getitem__(self, index):
        
        file_path = os.path.join(self._file_dir, self._training_files[index])
        graph     = load_graph_from_json(file_path)

        if not self._is_hetero:
            return self._homogenous_data(graph)
        else:
            return self._heterogenous_data(graph)

    def _homogenous_data(self, graph: nx.DiGraph) -> Union[ Data, tuple[Data, tuple[dict, nx.DiGraph]] ]:
        """
        Args: 
            1. graphs       : the directed graph of idx. From data/training_data/input/task_graph_idx.json 
            2. target_data  : the output features of the graph. From data/training_data/target/task_graph_idx.json

        returns a tuple of Data and an empty dictionary (for compatibility with heterogenous data)
        """
        
        data = Data()

        task_input_feature  = []
        task_target_feature = []

        for idx, node_data in graph.nodes(data=True):
            generate        = node_data["generate"]         / self.max_generate
            processing_time = node_data["processing_time"]  / self.max_processing_time
            start_cycle     = node_data["start_cycle"]      / self.max_cycle
            end_cycle       = node_data["end_cycle"]        / self.max_cycle

            task_input_feature.append([generate, processing_time])
            task_target_feature.append([start_cycle, end_cycle])

        data.x = torch.tensor(task_input_feature, dtype=torch.float)
        data.y = torch.tensor(task_target_feature, dtype=torch.float)

        edge_index = []
        for src, dst in graph.edges:
            edge_index.append([src, dst])

        data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        self._do_checks(data)

        if self._return_graph:
            return (data, ({}, graph))

        else: 
            return data



    def _heterogenous_data(self, graph: nx.DiGraph) -> Union[
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
        global_to_local_indexing["task_depend"] = {}    

        if self._has_scheduler_node:
            # Create a scheduler node to connect all task nodes
            global_to_local_indexing["scheduler"] = {}

            scheduler_node_id   = graph.number_of_nodes()
            scheduler_edge_type = "schedules"    

            graph.add_node(scheduler_node_id, type="scheduler")

            for node_idx, node_data in graph.nodes(data=True):

                if node_data["type"] in ["task", "task_depend"]:
                    graph.add_edge(scheduler_node_id, node_idx, type=scheduler_edge_type)

        # Creating node features + global to local indexing
        for node_idx, node_data in graph.nodes(data=True):
            node_type = node_data["type"]

            if node_type not in hetero_data:
                hetero_data[node_type].x = []

            if node_type == "task" or node_type == "task_depend":
                generate = node_data["generate"] / self.max_generate

                processing_time = (
                    node_data["processing_time"] / self.max_processing_time)

                target_start_cycle  = node_data["start_cycle"] / self.max_cycle
                target_end_cycle    = node_data["end_cycle"] / self.max_cycle

                # target is the same for both task and task_depend nodes
                if node_type == "task_depend":
                    # wait_time = node_data["wait_time"] / self.max_cycle
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

        hetero_data["task_depend"].x    = torch.tensor(task_depend_input_feature, dtype=torch.float)
        hetero_data["task_depend"].y    = torch.tensor(task_depend_target, dtype=torch.float)

        if self._has_scheduler_node: 
            hetero_data["scheduler"].x = torch.ones( (1, 1), dtype=torch.float)

        # Creating edge indices
        require_edge_type = "requires"

        hetero_data["task", require_edge_type, "task"].edge_index               = [[], []]

        hetero_data["dependency", require_edge_type, "task_depend"].edge_index  = [[], []]
        hetero_data["task_depend", require_edge_type, "task_depend"].edge_index = [[], []]
        hetero_data["task_depend", require_edge_type, "task"].edge_index    = [[], []]
        hetero_data["task", require_edge_type, "task_depend"].edge_index    = [[], []]

        if self._has_scheduler_node:
            hetero_data["scheduler", scheduler_edge_type, "task"].edge_index = [[], []]
            hetero_data["scheduler", scheduler_edge_type, "task_depend"].edge_index = [[], []]


        for edge in graph.edges(data=True):
            src, dst, _ = edge

            src_type    = graph.nodes[src]["type"]
            dst_type    = graph.nodes[dst]["type"]
            
            if src_type == "scheduler": 
                edge_type = scheduler_edge_type
            else: 
                edge_type = require_edge_type

            # print(f"Srctype: {src_type}, Dsttype: {dst_type}, Edge type: {edge_type}")
        
            hetero_data[src_type, edge_type, dst_type].edge_index[0].append(
                global_to_local_indexing[src_type][src])

            hetero_data[src_type, edge_type, dst_type].edge_index[1].append(
                global_to_local_indexing[dst_type][dst])

        # Converting edge indices [list] to tensors
        for edge_type in hetero_data.edge_types:

            hetero_data[edge_type].edge_index = torch.tensor(
                                                    hetero_data[edge_type].edge_index, 
                                                    dtype=torch.long
                                                ).contiguous()

        hetero_data = ToUndirected()(hetero_data)  # To leverage message passing in both directions
        
        self._do_checks(hetero_data)

        # [debugging] Uncomment to visualize the graph
        # from data.utils import visualize_graph
        # visualize_graph(graph=graph)    
        # from training.utils import log_hetero_data
        # log_hetero_data(hetero_data)
        
        if self._return_graph:
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
        batch_size          : int =32, 
        validation_split    : float =0.1,
        **kwargs

    ) -> tuple[DataLoader, DataLoader]:

    is_hetero           = kwargs.get( "is_hetero", False )
    has_scheduler_node  = kwargs.get( "has_scheduler_node", False )
    use_noc_dataset     = kwargs.get( "use_noc_dataset", False )


    if use_noc_dataset:    
        from training.noc_dataset import NocDataset
        dataset = NocDataset( training_data_dir )
        print(f"[load_data] Is NOC dataset: \t{use_noc_dataset}")

    else: 
        print(f"[load_data] Is hetero graph: \t{is_hetero}")
        dataset = CustomDataset(
                    training_data_dir   = training_data_dir, 
                    is_hetero           = is_hetero, 
                    has_scheduler_node  = has_scheduler_node,
                    return_graph        = False)

    validation_size = int( validation_split * len( dataset ) )

    if validation_size == 0: # ensure at least one validation sample
        validation_size = 1  

    train_size = len( dataset ) - validation_size

    if train_size <= 0:
        raise ValueError( "Training dataset size is too small after splitting." )

    train_dataset, val_dataset = random_split( dataset, [train_size, validation_size] )

    print( f"Training dataset size: \t\t{len(train_dataset)}" )
    print( f"Validation dataset size: \t{len(val_dataset)}" )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True )

    return train_loader, val_loader


if __name__ == "__main__":

    from data.utils import visualize_application

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of the file in training_data/simualtor/test to visualize")
    parser.add_argument("--is_hetero", action="store_true", help="Use the model with network")
    parser.add_argument("--has_scheduler", action="store_true", help="Use the model with network")
    parser.add_argument("--show_graph", action="store_true", help="Use the model with network")
    args = parser.parse_args()

    DATA_INDEX      = args.idx
    IS_HETERO       = args.is_hetero
    HAS_SCHEDULER   = args.has_scheduler
    SHOW_GRAPH      = args.show_graph   

    print( f"\n------Dataset Test------"               )
    print( f"Data index is          : {DATA_INDEX}"    )
    print( f"Is heterogenous graph  : {IS_HETERO}"     )
    print( f"Connecting task nodes  : {HAS_SCHEDULER}" )
    print( f"Show graph             : {SHOW_GRAPH}"    )

    DATASET_DIR = "data/training_data/without_network/train"

    dataset = CustomDataset( DATASET_DIR, 
                             is_hetero           = IS_HETERO, 
                             has_scheduler_node  = HAS_SCHEDULER,
                             return_graph        = SHOW_GRAPH )   

    def check_all_files_in_dataset():
        # Inside a def for scoping
        print( f"\n------Checking all files in the dataset------" )
        list_of_edge_type = []

        for idx, ( data, ( index_dict, graph ) ) in enumerate( dataset ):
        
            for src, dst, edge in graph.edges(data=True):
                src_type = graph.nodes[src]["type"]
                dst_type = graph.nodes[dst]["type"]
    
                edge_type = f"{src_type}_to_{dst_type}"
    
                if edge_type not in list_of_edge_type:
                    print( f"New edge type found: {edge_type}" )
                    list_of_edge_type.append( edge_type )

        print( f"[Passed] All files are valid\n" )

    if SHOW_GRAPH:
        data, ( index, graph ) = dataset[DATA_INDEX]
        check_all_files_in_dataset()
        visualize_application( graph=graph )

    print( f"\n\n----------------------DataLoader Test----------------------" )

    BATCH_SIZE = 10
    print( f"\nLoading data with batch size {BATCH_SIZE}" )

    train_loader, val_loader           = load_data( training_data_dir  = DATASET_DIR, 
                                                    is_hetero          = IS_HETERO, 
                                                    batch_size         = BATCH_SIZE,
                                                    has_scheduler_node = HAS_SCHEDULER )

    print( f"DataLoader is                   {train_loader}" )
    print( f"Number of training batches:     {len(train_loader)}" )  
    print( f"Batch size:                     {train_loader.batch_size}" ) 
    print( f"Training samples                {len(train_loader) * train_loader.batch_size}" ) 
    print( f"Number of validation batches:   {len(val_loader)}" )
    
    print( f"CustomDatasert and DataLoader looking good! 😎" )
    


