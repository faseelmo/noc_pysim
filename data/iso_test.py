
import os 
import torch
import shutil 
import numpy as np 
from matplotlib import pyplot as plt    
import networkx as nx 

from src.utils import get_mesh_network
from src.simulator import Simulator, GraphMap

from training.train import ( get_max_latency_hetero, get_true_pred_hetero ) 

from data.utils import ( import_model_dataset_param, 
                         modify_graph_to_application_graph,
                         save_graph_to_json, 
                         visualize_application, 
                         get_weights_from_directory,
                         visualize_noc_application )

if __name__ == "__main__": 

    iso_test_path = "data/iso_test"

    model_path = "training/results/experiments/2_loss/3_weighted_mae_1000"
    model_epoch = 27
    model, dataset, params = import_model_dataset_param( model_path )

    # Loading the model
    model = model.HeteroGNN( params["HIDDEN_CHANNELS"], params["NUM_MPN_LAYERS"] )
    model_weights_path = get_weights_from_directory( f"{model_path}/models", model_epoch )
    model.load_state_dict( torch.load( model_weights_path, weights_only=False ) )

    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)

    mesh_size = 4
    debug_mode = False

    graph = modify_graph_to_application_graph( graph, 
                                               generate_range=(2, 2), 
                                               processing_time_range=(2, 2) )
    # visualize_application(graph)

    simulator = Simulator ( mesh_size, mesh_size, max_cycles=1000, debug_mode=debug_mode )
    task_list = simulator.graph_to_task(graph)
    

    # 
    mapping = [ GraphMap( task_id=0, assigned_pe=( 0,0 ) ), 
                GraphMap( task_id=1, assigned_pe=( 2,1 ) ), 
                GraphMap( task_id=2, assigned_pe=( 1,2 ) ), 
                GraphMap( task_id=3, assigned_pe=( 3,3 ) ) ]

    mapping_list = simulator.set_assigned_mapping_list(task_list, mapping)
    simulator.map(mapping_list)
    latency = simulator.run()
    print(f"First run Latency: {latency}")
    output_graph = get_mesh_network(mesh_size, graph, mapping_list)

    # visualize_noc_application(output_graph)
    simulator.clear()

    #


    mapping = [ GraphMap( task_id=0, assigned_pe=( 3,3 ) ), 
                GraphMap( task_id=1, assigned_pe=( 1,2 ) ), 
                GraphMap( task_id=2, assigned_pe=( 2,1 ) ), 
                GraphMap( task_id=3, assigned_pe=( 0,0 ) ) ]

    task_list = simulator.graph_to_task(graph)
    mapping_list = simulator.set_assigned_mapping_list(task_list, mapping)
    simulator.map(mapping_list)
    latency = simulator.run()
    output_graph = get_mesh_network(mesh_size, graph, mapping_list)
    # visualize_noc_application(output_graph)

    print(f"Second run Latency: {latency}")







    # output_graph = get_mesh_network(mesh_size, graph, mapping_list)
    # visualize_noc_application(output_graph)    

    # adj_matrix = nx.to_numpy_array(output_graph, weight=None)
    # print(f"Adjacency matrix: {adj_matrix}")
    # np.savetxt(f"{iso_test_path}/adjacency_matrix.txt", adj_matrix, fmt="%d")

    # num_nodes = adj_matrix.shape[0]
    # permutations = np.random.permutation(num_nodes)
    # iso_adj_matrix = adj_matrix[permutations][:, permutations]
    # iso_graph = nx.from_numpy_array(iso_adj_matrix, create_using=nx.DiGraph)

    # assert nx.is_isomorphic(output_graph, iso_graph), "Graph is not isomorphic"

    # original_node_dict = {}

    # for node_idx, node in enumerate(output_graph.nodes()):
    #     original_node_dict[node_idx] = node

    # for new_node_idx, original_node_idx in enumerate(permutations):
    #     original_node_id = original_node_dict[original_node_idx]
    #     original_node_attr = output_graph.nodes(data=True)[original_node_id]
    #     iso_graph.nodes[new_node_idx].update(original_node_attr)
    #     print(f"")
    #     print(f"Original Attr is {original_node_attr}")
    #     print(f"New Attr is {iso_graph.nodes(data=True)[new_node_idx]}")

    # new_to_original_mapping = {new_node_idx: original_node_dict[original_node_idx]
    #                        for new_node_idx, original_node_idx in enumerate(permutations)}

    # iso_graph = nx.relabel_nodes(iso_graph, new_to_original_mapping)

    # for node_id, node in iso_graph.nodes(data=True):
    #     print(f"Node {node_id} has attr {node}")

    # visualize_noc_application(output_graph)
    # visualize_noc_application(iso_graph)
    



    








    exit()

    os.makedirs(iso_test_path, exist_ok=True)  
    save_graph_to_json(output_graph, f"{iso_test_path}/graph.json")

    dataset = dataset.NocDataset(iso_test_path)
    data = dataset[0]

    output = model(data.x_dict, data.edge_index_dict)

    true, pred = get_max_latency_hetero( data, output )

    print(f"True latency: {int(true)}, Predicted latency: {int(pred)}")
    
    shutil.rmtree(iso_test_path)



