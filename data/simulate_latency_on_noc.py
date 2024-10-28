
import networkx as nx

from src.simulator              import Simulator
from src.utils                  import get_mesh_network
from data.utils                 import load_graph_from_json, save_graph_to_json, visualize_graph
from data.create_graph_tasks    import generate_graph, modify_graph_to_application_graph

def simulate(num_nodes: int, idx: int) -> None: 

    graph       = generate_graph(num_nodes)
    graph       = modify_graph_to_application_graph(graph)

    debug_mode  = False 

    if debug_mode: 
        visualize_graph(graph)

    sim             = Simulator(
                        num_rows=3, 
                        num_cols=3, 
                        debug_mode=debug_mode)

    task_list       = sim.graph_to_task(graph)
    mapping_list    = sim.get_random_mapping(task_list)
    
    sim.map(mapping_list)
    sim.run()

    training_graph = get_mesh_network(3, graph, mapping_list)

    return training_graph


if __name__ == "__main__": 

    import random 
    import os

    random.seed(0)

    test_split          = 0.05
    training_data_count = 8000
    training_graphs     = []

    for i in range(training_data_count): 
        print( f"\rCreating Graph {i}", end='', flush=True )
        nodes           = random.randint(2, 5)
        training_graph  = simulate(nodes, i)
        training_graphs.append(training_graph)  

    test_graphs     = random.sample(training_graphs, int(test_split * training_data_count))
    training_graphs = [graph for graph in training_graphs if graph not in test_graphs]

    print(f"\nNumber of training graphs: {len(training_graphs)}, Number of test graphs: {len(test_graphs)}")

    test_data_dir   = os.path.join("data", "training_data", "simulator", "test")
    traing_data_dir = os.path.join("data", "training_data", "simulator", "train")

    if not os.path.exists(test_data_dir): 
        os.makedirs(test_data_dir)

    if not os.path.exists(traing_data_dir):
        os.makedirs(traing_data_dir)

    for i, graph in enumerate(test_graphs): 
        save_graph_to_json(graph, os.path.join(test_data_dir, f"{i}.json"))

    for i, graph in enumerate(training_graphs): 
        save_graph_to_json(graph, os.path.join(traing_data_dir, f"{i}.json"))






