import os
import json
import torch 
import networkx as nx

from natsort import natsorted

from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader


class CustomDataset(Dataset):
    def __init__(self, training_data_dir):

        self.input_dir = os.path.join(training_data_dir, "input")
        self.target_dir = os.path.join(training_data_dir, "target")

        self.input_files = natsorted(os.listdir(self.input_dir))
        self.output_files = natsorted(os.listdir(self.target_dir))

        assert len(self.input_files) == len(
            self.output_files
        ), "Number of input files and output files must be the same"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file_path = os.path.join(self.input_dir, self.input_files[index])
        target_file_path = os.path.join(self.target_dir, self.output_files[index])

        with open(input_file_path, "r") as f:
            input_data = json.load(f)

        with open(target_file_path, "r") as f:
            target_data = json.load(f)

        graph = nx.node_link_graph(input_data)

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

def load_data(training_data_dir, batch_size=32, validation_split=0.1):
    dataset = CustomDataset(training_data_dir)
    validation_size = int(validation_split * len(dataset))

    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_dataset


if __name__ == "__main__":

    data_index = 2
    dataset = CustomDataset("data/training_data")
    data = dataset[data_index]

    nx_graph_path = f"data/training_data/input/task_graph_{data_index}.json"
    nx_graph = nx.node_link_graph(json.load(open(nx_graph_path)))


    # Testing Custom Dataset
    print(f"Data is {data}")
    print(f"Feature matrix \n{data.x}\n")
    print(f"\nedge_index \n{data.edge_index}\n")
    print(f"edge_attr \n{data.edge_attr}\n")
    print(f"data is valid: {data.validate(raise_on_error=True)}")
    print(f"is directed graph: {data.is_directed()}")

    # from data.utils import visualize_graph
    # visualize_graph(nx_graph)

    # Testing DataLoader
    train_loader, val_loader = load_data("data/training_data", batch_size=32)

    print(f"\nData loader information")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total number of training samples {len(train_loader) * train_loader.batch_size}")

    print(f"Number of validation batches: {len(val_loader)}")

