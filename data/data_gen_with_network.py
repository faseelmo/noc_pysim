
import yaml
import shutil
import numpy    as np

from src.simulator import Simulator
from src.utils     import get_mesh_network
from data.utils    import ( save_graph_to_json, 
                            visualize_application, 
                            visualize_noc_application, 
                            generate_graph, 
                            modify_graph_to_application_graph )

PARAMS = yaml.safe_load(open("training/config/params_with_network.yaml"))  

def simulate(num_nodes: int, map_count : int = 1) -> list: 

    graph = generate_graph(num_nodes)
    graph = modify_graph_to_application_graph( graph, 
                                               ( PARAMS['MIN_GENERATE'] ,        PARAMS['MAX_GENERATE'] ), 
                                               ( PARAMS['MIN_PROCESSING_TIME'] , PARAMS['MAX_PROCESSING_TIME'] ) )

    debug_mode  = False 
    if debug_mode: visualize_application(graph)

    mesh_size = PARAMS['MESH_SIZE']

    graph_list = []
    max_latency = 0
    for i in range(map_count): 

        sim = Simulator( num_rows=mesh_size, 
                         num_cols=mesh_size, 
                         max_cycles= PARAMS['MAX_CYCLE'],
                         debug_mode=debug_mode )

        task_list    = sim.graph_to_task(graph)
        mapping_list = sim.get_random_mapping(task_list)

        sim.map(mapping_list)
        latency = sim.run()
        if latency > max_latency: 
            max_latency = latency

        output_graph = get_mesh_network(mesh_size, graph, mapping_list)
        graph_list.append(output_graph)

        # visualize_noc_application(output_graph)

    return graph_list, max_latency


if __name__ == "__main__": 

    import random 
    import os

    random.seed(0)

    test_split          = 1000      # 400
    training_data_count = 10000     # 12000
    training_graphs     = []
    node_range          = (2, 6)
    map_count           = 4  

    print(f"NoC Size: {PARAMS['MESH_SIZE']}")
    print(f"Saving data to {PARAMS['DATA_DIR']}")


    # Simulating
    max_latency = 0 
    for i in range(training_data_count): 
        nodes               = random.randint(*node_range)
        graph_list, latency = simulate(nodes, map_count=map_count)
        training_graphs.extend(graph_list)  
        if latency > max_latency: 
            max_latency = latency
            print(f"\nMax Latency: {max_latency}")
        print( f"\rCreating Graph {i*len(graph_list)}", end='', flush=False )

    test_graphs     = random.sample(training_graphs, test_split)
    training_graphs = [graph for graph in training_graphs if graph not in test_graphs]

    print(f"\nNumber of training graphs: {len(training_graphs)}, Number of test graphs: {len(test_graphs)}")

    test_data_dir   = os.path.join( PARAMS['DATA_DIR'], "test" )
    traing_data_dir = os.path.join( PARAMS['DATA_DIR'], "train" )

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

    # # # Creating data for mapping test metric 
    map_test_dir    = os.path.join( PARAMS['DATA_DIR'], "map_test" )
    if os.path.exists(map_test_dir): 
        shutil.rmtree(map_test_dir)

    os.makedirs(map_test_dir)
    # exit()

    num_metric_graph        = 10
    metric_maps_per_graph   = 100
    count                   = 0
    std_threshold           = 5    
    node_range              = (2, 5)
    # map_node_range          = [2, 3, 4]

    print(f"Creating Mapping Test graphs with std threshold {std_threshold}")
    map_count_dict = {}

    while count < num_metric_graph:
        
        # nodes                   = random.choice(map_node_range)
        nodes = random.randint(*node_range)
        map_graph_list, latency = simulate(nodes, map_count=metric_maps_per_graph)

        latency_list = []
        for _, graph in enumerate(map_graph_list): 
            # Checking the distribution of the latency
            end_cycle_list = []

            for node_id, node_data in graph.nodes(data=True): 
                if node_data["type"] == "task": 
                    end_cycle = node_data["end_cycle"] 
                    end_cycle_list.append(end_cycle)    

            max_latency = max(end_cycle_list)
            latency_list.append(max_latency)

        num_nodes = len(map_graph_list[0].nodes) - ((PARAMS['MESH_SIZE']**2) * 2) 

        std = np.std(latency_list)
        print( f"\rStd: \t{std}, num_nodes \t{num_nodes}", end='', flush=False ) 

        if std > std_threshold:
            dir     = os.path.join(map_test_dir, f"{count}")
            
            if map_count_dict.get(num_nodes) is None: 
                map_count_dict[num_nodes] = 0

            map_count_dict[num_nodes] += 1

            if num_nodes == 5:
                if map_count_dict[num_nodes] >= 1: 
                    node_range = (2, 4)


            if not os.path.exists(dir): 
                os.makedirs(dir)

            for j, graph in enumerate(map_graph_list):
                save_graph_to_json(graph, os.path.join(dir, f"{j}.json"))

            print( f"\nCreating Mapping Test graph {count}" )
            count   += 1

    print()







