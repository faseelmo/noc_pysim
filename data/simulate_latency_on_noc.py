
import networkx as nx

from src.simulator              import Simulator
from src.utils                  import get_mesh_network
from data.utils                 import load_graph_from_json, save_graph_to_json, visualize_graph
from data.create_graph_tasks    import generate_graph, modify_graph_to_application_graph

def simulate(num_nodes: int, map_count : int = 1) -> list: 

    graph       = generate_graph(num_nodes)
    graph       = modify_graph_to_application_graph(graph)

    debug_mode  = False 
    if debug_mode: visualize_graph(graph)

    graph_list = []
    for i in range(map_count): 

        sim             = Simulator(
                            num_rows=3, 
                            num_cols=3, 
                            debug_mode=debug_mode)

        task_list       = sim.graph_to_task(graph)
        mapping_list    = sim.get_random_mapping(task_list)

        sim.map(mapping_list)
        sim.run()

        output_graph = get_mesh_network(3, graph, mapping_list)
        graph_list.append(output_graph)

    return graph_list


if __name__ == "__main__": 

    import random 
    import os

    random.seed(0)

    test_split          = 400
    training_data_count = 8000
    maps_per_graph      = 1
    training_graphs     = []
    node_range          = (2, 5)

    # Simulating
    for i in range(training_data_count): 
        nodes           = random.randint(*node_range)
        graph_list      = simulate(nodes, map_count=maps_per_graph)
        training_graphs.extend(graph_list)  
        print( f"\rCreating Graph {i*len(graph_list)}", end='', flush=True )

    test_graphs     = random.sample(training_graphs, test_split)
    training_graphs = [graph for graph in training_graphs if graph not in test_graphs]

    print(f"\nNumber of training graphs: {len(training_graphs)}, Number of test graphs: {len(test_graphs)}")

    test_data_dir   = os.path.join("data", "training_data", "simulator", "test")
    traing_data_dir = os.path.join("data", "training_data", "simulator", "train")
    map_test_dir    = os.path.join("data", "training_data", "simulator", "map_test")

    # Creating test data dir 
    if not os.path.exists(test_data_dir): 
        os.makedirs(test_data_dir)

    for i, graph in enumerate(test_graphs): 
        save_graph_to_json(graph, os.path.join(test_data_dir, f"{i}.json"))

    # Creating training data dir 
    if not os.path.exists(traing_data_dir):
        os.makedirs(traing_data_dir)

    for i, graph in enumerate(training_graphs): 
        save_graph_to_json(graph, os.path.join(traing_data_dir, f"{i}.json"))

    # Creating data for mapping test metric 
    if not os.path.exists(map_test_dir): 
        os.makedirs(map_test_dir)

    count = 0
    for i in range(20): 
        
        nodes           = random.randint(*node_range)
        map_graph_list  = simulate(nodes, map_count=20)

        for j, graph in enumerate(map_graph_list): 
            dir = os.path.join(map_test_dir, f"{i}")

            if not os.path.exists(dir): 
                os.makedirs(dir)

            save_graph_to_json(graph, os.path.join(dir, f"{j}.json"))

            count += 1
            print( f"\rCreating Mapping Test grap {count}", end='', flush=True )

    print()







