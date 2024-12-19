
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

    model_path = "training/results/with_network/4x4_v4"
    model_epoch = 13
    model, dataset, params = import_model_dataset_param( model_path )

    # Loading the model
    model = model.HeteroGNN( params["HIDDEN_CHANNELS"], params["NUM_MPN_LAYERS"], params["MESH_SIZE"] )
    model_weights_path = get_weights_from_directory( f"{model_path}/models", model_epoch )
    model.load_state_dict( torch.load( model_weights_path, weights_only=False ) )

    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_node(2)

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    mesh_size = 4
    debug_mode = False

    graph = modify_graph_to_application_graph( graph, 
                                               generate_range=(2, 2), 
                                               processing_time_range=(2, 2) )

    simulator = Simulator ( mesh_size, mesh_size, max_cycles=1000, debug_mode=debug_mode )
    task_list = simulator.graph_to_task( graph )

    mapping = [ GraphMap( task_id=0, assigned_pe=( 3,0 ) ), 
                GraphMap( task_id=1, assigned_pe=( 2,0 ) ),
                GraphMap( task_id=2, assigned_pe=( 1,0 ) ) ]

    mapping_list = simulator.set_assigned_mapping_list( task_list, mapping )
    simulator.map( mapping_list )
    latency = simulator.run()
    output_graph = get_mesh_network( mesh_size, graph, mapping_list )
    visualize_noc_application( output_graph )

    os.makedirs( iso_test_path, exist_ok=True )  
    save_graph_to_json( output_graph, f"{iso_test_path}/graph.json" )

    dataset = dataset.NocDataset( iso_test_path )
    data = dataset[0]

    output = model( data.x_dict, data.edge_index_dict )

    true, pred = get_max_latency_hetero( data, output )
    print(f"True latency: {int(true)}, Predicted latency: {int(pred)}")
    
    shutil.rmtree(iso_test_path)



