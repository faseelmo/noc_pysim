
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
    save_graph_to_json(training_graph, f"data/training_data/simulator/test/{idx}.json")


if __name__ == "__main__": 

    import random 
    random.seed(0)
    training_data_count = 1000

    for i in range(training_data_count): 
        nodes = random.randint(2, 5)
        print( f"\rCreating Graph {i}", end='', flush=True )
        simulate(nodes, i)

    print(f"\n{training_data_count} graphs generated")
