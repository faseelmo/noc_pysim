import shutil

from src.utils                  import graph_to_task_list, simulate_application_on_pe, get_ordered_packet_list
from data.utils                 import visualize_graph, update_graph_with_computing_list, save_graph_to_json, generate_graph, modify_graph_to_task_graph

def simulate(num_nodes: int, debug_mode: bool, sjf_scheduling: bool ): 

    graph = generate_graph(num_nodes)
    graph = modify_graph_to_task_graph(graph) # Wait time is required for ordered packet injection

    computing_list  = graph_to_task_list(graph)
    packet_list     = get_ordered_packet_list(graph)

    packet_list_copy = []
    for packet in packet_list: 
        packet_list_copy.append( packet.get_source_task_id() )

    latency = simulate_application_on_pe( computing_list, 
                                          packet_list, 
                                          debug_mode      = debug_mode, 
                                          sjf_scheduling  = sjf_scheduling )

    graph = update_graph_with_computing_list(computing_list, graph)

    if debug_mode:
        print(f"---- Debug Mode ----")
        print(f"Number of nodes: {num_nodes}")
        print(f"\nPacket List")
        print(*packet_list)

        print(f"\nComputing List")
        for idx, task in enumerate(computing_list):
            print(f"{idx}. {task}\n")

    return graph

def create_and_clear_dir(directory_path):
    if os.path.exists(directory_path):
        print(f"The directory {directory_path} already exists. Remove it?")
        user_input = input("Enter Yes to remove: ")
        if user_input.lower() == "yes":
            print(f"Removing {directory_path}")
            shutil.rmtree(directory_path)
    os.makedirs(directory_path)
    print(f"Created directory: {directory_path}")

if __name__ == "__main__":

    import random 
    import os 

    random.seed(0)
    training_data_dir    = os.path.join("data", "training_data", "without_network", "train")
    test_data_dir       = os.path.join("data", "training_data", "without_network", "test")

    create_and_clear_dir(training_data_dir)
    create_and_clear_dir(test_data_dir)

    training_data_count = 8000
    test_split          = 400 
    training_graphs     = []
    node_range          = (2, 5)

    DEBUG_MODE          = False
    sjf_scheduling      = True

    # Simulating
    for i in range( training_data_count ): 
        nodes           = random.randint( *node_range ) 
        graph           = simulate( nodes, debug_mode=DEBUG_MODE, sjf_scheduling=sjf_scheduling)
        training_graphs.append(graph)  
        print( f"\rCreating graph {i}", end='', flush=True )

    test_graphs = random.sample( training_graphs, test_split )

    for i, graph in enumerate(training_graphs):
        save_graph_to_json(graph, os.path.join(training_data_dir, f"{i}.json"))
    print("\nCreated training data.")

    print("Saving Test graphs...")
    for i, graph in enumerate(test_graphs):
        save_graph_to_json(graph, os.path.join(test_data_dir, f"{i}.json"))




