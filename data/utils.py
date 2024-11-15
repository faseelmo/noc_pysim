import networkx as nx

def save_graph_to_json(graph: nx.DiGraph, filename: str):
    import json
    data = nx.node_link_data(graph)
    with open(filename, "w") as file:
        json.dump(data, file)


def load_graph_from_json(filename: str) -> nx.DiGraph:
    import json
    with open(filename, "r") as file:
        data = json.load(file)
    return nx.node_link_graph(data)

def compute_list_to_node_dict(compute_list):
    """
    Converts compute list to a dictionary with task_id as key
    and start and end cycle as value
    """
    node_dict = {}
    for task in compute_list:
        node_dict[task.task_id] = {
            "start_cycle": task.start_cycle,
            "end_cycle": task.end_cycle
        }
    return node_dict


def get_compute_list_from_json(filename: str) -> dict:
    """
    Converts the node cycle information from the json file to a dictionary
    used in inspect_data.py
    """
    import json
    json_dict = json.load(open(filename))

    compute_list = {}
    for key in json_dict:
        if key == "latency":
            continue
        else: 
            compute_list[int(key)] = json_dict[key] 

    return compute_list

def get_weights_from_directory(directory: str, file_name: str):
    import os 
    files = os.listdir(directory)
    for file in files:
        if f"_{file_name}" in file:
            return os.path.join(directory, file)
    else: 
        raise Exception(f"File {file_name} not found in directory {directory}")


def get_all_weights_from_directory(directory: str):
    import os
    files = os.listdir(directory)
    weights_files = []
    for file in files:
        if ".pth" in file:
            path = os.path.join(directory, file)
            weights_files.append(path)
    return weights_files

def visualize_graph(
        graph: nx.DiGraph, 
        latency_value=None, 
        packet_list=None, 
        compute_list=None, 
        pred_compute_list=None):
    """
    args: 
        compute_list: list or dict. Used to display the start and end cycle (truth) of each task

    """
    import matplotlib.pyplot as plt
    import networkx as nx

    node_color_map = {"dependency": "skyblue", "task": "lightgreen", "task_depend": "yellow", "scheduler": "tomato"}

    node_colors = [
        node_color_map.get(graph.nodes[node].get("type", "task"), "lightgreen")
        for node in graph.nodes
    ]

    seed = 0
    pos = nx.spring_layout(graph, seed=seed)
    # color_light_gray = 
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=900,
        node_color=node_colors,
        arrows=True,
        edge_color="lightgray",
    )

    if isinstance(compute_list, dict): 
        node_cycle_dict = compute_list
    else: 
        node_cycle_dict = compute_list_to_node_dict(compute_list) if compute_list else {}

    custom_labels = {}
    for node in graph.nodes:
        label_parts = [f"id: {node}"]
        if "processing_time" in graph.nodes[node]:
            label_parts.append(f"P: {graph.nodes[node]['processing_time']}")
        if "generate" in graph.nodes[node]:
            label_parts.append(f"G: {graph.nodes[node]['generate']}")
        if "wait_time" in graph.nodes[node]:
            wait_time = graph.nodes[node]["wait_time"]
            if wait_time != 0:
                label_parts.append(f"W: {wait_time}")
        if node in node_cycle_dict:
            
            truth_start_cycle   = node_cycle_dict[node]['start_cycle']
            truth_end_cycle     = node_cycle_dict[node]['end_cycle']

            label_parts.append(f"T: ({truth_start_cycle}-{truth_end_cycle}) "
                               f"= {truth_end_cycle - truth_start_cycle}")

            if pred_compute_list:
                pred_start_cycle    = pred_compute_list[node]['start_cycle']
                pred_end_cycle      = pred_compute_list[node]['end_cycle']
    
                label_parts.append(f"P: ({pred_start_cycle}-{pred_end_cycle}) "
                                   f"= {pred_end_cycle - pred_start_cycle}")

        custom_labels[node] = "\n".join(label_parts)

    nx.draw_networkx_labels(graph, pos, labels=custom_labels)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    if latency_value is not None:
        plt.text(
            0.5,
            0.05,
            f"Latency: {latency_value}",
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    if packet_list is not None:
        plt.text(
            0.5,
            0.01,
            f"Packet list: ←{packet_list}",
            ha="center",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    plt.show()


def does_path_contains_files(path: str):
    import os

    files = os.listdir(path)

    if len(files) > 0:
        delete_prompt = input(
            f"Path '{path}' already contains files. Do you want to delete them? (yes/no): "
        )
        if delete_prompt.lower() == "yes":
            for file in files:
                os.remove(os.path.join(path, file))
            print(f"Files in '{path}' deleted.")
        else:
            print(f"Files in '{path}' not deleted. Appending new files.")

