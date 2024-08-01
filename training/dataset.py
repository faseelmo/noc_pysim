import os
import json
import torch
import yaml
import networkx as nx

from natsort import natsorted

from data.utils import visualize_graph

from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData


class CustomDataset(Dataset):
    def __init__(self, training_data_dir, is_hetero=False):

        self.is_hetero = is_hetero
        self.input_dir = os.path.join(training_data_dir, "input")
        self.target_dir = os.path.join(training_data_dir, "target")

        self.input_files = natsorted(os.listdir(self.input_dir))
        self.output_files = natsorted(os.listdir(self.target_dir))

        training_parameters = yaml.safe_load(open("training/params.yaml"))
        self.max_generate = training_parameters["MAX_GENERATE"]
        self.max_processing_time = training_parameters["MAX_PROCESSING_TIME"]

        assert len(self.input_files) == len(
            self.output_files
        ), "Number of input files and output files must be the same"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file_path = os.path.join(self.input_dir, self.input_files[index])
        target_file_path = os.path.join(self.target_dir, self.output_files[index])

        input_data = self._load_json(input_file_path)
        target_data = self._load_json(target_file_path)

        graph = nx.node_link_graph(input_data)

        if self.is_hetero:
            return self._heterogenous_data(graph, target_data)
        else:
            return self._homogenous_data(graph, target_data)

    def _load_json(self, filename):
        with open(filename, "r") as file:
            data = json.load(file)
        return data

    def _homogenous_data(self, graph, target_data):

        data = from_networkx(graph)

        # data.x should contain all the node features with shape [num_nodes, num_node_features]
        data.x = torch.stack((data.generate, data.processing_time), dim=1).float() / 10
        del data.generate
        del data.processing_time

        # data.edge_attr should contain all the edge features with shape [num_edges, num_edge_features]
        data.edge_attr = torch.stack((data.weight,), dim=1).float()
        del data.weight

        # delete type for now
        del data.type

        # target
        data.y = float(target_data["latency"])

        return data

    def _heterogenous_data(self, graph, target_data):
        hetero_data = HeteroData()

        global_to_local = {'task': {}, 'dependency': {}}
        task_feature = []
        dependency_feature = []
        for node_idx, node_data in graph.nodes(data=True):
            node_type = node_data["type"]

            if node_type not in hetero_data:
                hetero_data[node_type].x = []

            if node_type == "task":
                generate = node_data["generate"] / self.max_generate
                processing_time = (
                    node_data["processing_time"] / self.max_processing_time
                )
                task_feature.append([generate, processing_time])

            if node_type == "dependency":
                generate = node_data["generate"] / self.max_generate
                dependency_feature.append([generate])

        hetero_data["task"].x = torch.tensor(task_feature, dtype=torch.float)
        hetero_data["dependency"].x = torch.tensor(
            dependency_feature, dtype=torch.float
        )

        global_to_local_indexing = {}

        for edge in graph.edges(data=True):
            src, dst, edge_data = edge
            src_type = graph.nodes[src]["type"]
            dst_type = graph.nodes[dst]["type"]

            edge_type = f"{src_type}_to_{dst_type}"
            if edge_type not in global_to_local_indexing:
                global_to_local_indexing[edge_type] = {}

            if src not in global_to_local_indexing[edge_type]:
                global_to_local_indexing[edge_type][src] = len(global_to_local_indexing[edge_type])

            if dst not in global_to_local_indexing[edge_type]:
                global_to_local_indexing[edge_type][dst] = len(global_to_local_indexing[edge_type])


        print(f"global_to_local_indexing: {global_to_local_indexing}")



        for edge in graph.edges(data=True):
            src, dst, edge_data = edge
            src_type = graph.nodes[src]["type"]
            dst_type = graph.nodes[dst]["type"]

            if src_type == "task" and dst_type == "task":
                edge_type = "no_delay"
            elif src_type == "dependency" and dst_type == "task":
                edge_type = "delay"
            else:
                raise ValueError(f"Unknown edge type from {src_type} to {dst_type}")

            if (src_type, edge_type, dst_type) not in hetero_data.edge_types:
                hetero_data[src_type, edge_type, dst_type].edge_index = [[], []]

            edge_type_str = f"{src_type}_to_{dst_type}"

            hetero_data[src_type, edge_type, dst_type].edge_index[0].append(src)
            hetero_data[src_type, edge_type, dst_type].edge_index[1].append(dst)

        for edge_type in hetero_data.edge_types:
            hetero_data[edge_type].edge_index = (
                torch.tensor(hetero_data[edge_type].edge_index, dtype=torch.long)
                # .t()
                .contiguous()
            )

        print(f"Global to local {global_to_local}")

        hetero_data.y = float(target_data["latency"])

        return hetero_data


def load_data(training_data_dir, is_hetero, batch_size=32, validation_split=0.1):
    dataset = CustomDataset(training_data_dir, is_hetero)
    validation_size = int(validation_split * len(dataset))

    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_dataset


if __name__ == "__main__":

    data_index = 2
    is_hetero = False
    homogenous_dataset = CustomDataset("data/training_data", is_hetero=False)
    data = homogenous_dataset[data_index]

    # nx_graph_path = f"data/training_data/input/task_graph_{data_index}.json"
    # nx_graph = nx.node_link_graph(json.load(open(nx_graph_path)))

    # Testing Custom Dataset
    print(f"\n\n----------------------Homogenous Graph----------------------")
    print(f"Data is {data}")
    print(f"Feature matrix \n{data.x}\n")
    print(f"\nedge_index \n{data.edge_index}\n")
    print(f"edge_attr \n{data.edge_attr}\n")
    print(f"data is valid: {data.validate(raise_on_error=True)}")
    print(f"is directed graph: {data.is_directed()}")

    # from data.utils import visualize_graph
    # visualize_graph(nx_graph)

    # Testing DataLoader
    # train_loader, val_loader = load_data(homogenous_dataset, batch_size=32)

    # print(f"\nData loader information")
    # print(f"Number of training batches: {len(train_loader)}")
    # print(f"Batch size: {train_loader.batch_size}")
    # print(
    #     f"Total number of training samples {len(train_loader) * train_loader.batch_size}"
    # )

    # print(f"Number of validation batches: {len(val_loader)}")

    """Testing for Heterogenous Graph"""

    is_hetero = True
    heterogenous_dataset = CustomDataset("data/training_data", is_hetero=True)
    hetero_data = heterogenous_dataset[data_index]

    print(f"\n\n----------------------Heterogenous Graph----------------------")
    print(f"\nHeteroData is {hetero_data}")
    print(f"y is {hetero_data.y}")
    print(f"\nNode types are {hetero_data.node_types}")
    print(f"Edge types are {hetero_data.edge_types}")

    for node_type in hetero_data.node_types:
        print(
            f"Node type {node_type} has feature matrix \n{hetero_data[node_type].x}\n"
        )
    for edge_type in hetero_data.edge_types:
        print(
            f"Edge type {edge_type} has edge index \n{hetero_data[edge_type].edge_index}\n"
        )

    print(f"data is valid: {hetero_data.validate(raise_on_error=True)}")
    print(f"has self loops: {hetero_data.has_self_loops()}")
    print(f"is directed graph: {hetero_data.is_directed()}")
