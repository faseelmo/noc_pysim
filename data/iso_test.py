
import os 
import torch
import shutil 
import networkx as nx 

from src.utils import get_mesh_network
from src.simulator import Simulator, GraphMap

from training.train import get_max_latency_hetero, get_true_pred_hetero

from data.utils import ( import_model_dataset_param, 
                         modify_graph_to_application_graph,
                         save_graph_to_json, 
                         visualize_application, 
                         get_weights_from_directory,
                         visualize_noc_application )

if __name__ == "__main__": 

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

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    graph = modify_graph_to_application_graph( graph, (2, 2), (4, 4) )

    simulator = Simulator ( 4, 4, max_cycles=1000, debug_mode=False )

    task_list = simulator.graph_to_task(graph)
    
    x_pos = 2
    mapping = [ GraphMap(task_id=0, assigned_pe=( x_pos,0 )), 
                GraphMap(task_id=1, assigned_pe=( x_pos,1 )), 
                GraphMap(task_id=2, assigned_pe=( x_pos,3 )) ]

    mapping_list = simulator.set_assigned_mapping_list(task_list, mapping)
    simulator.map(mapping_list)
    latency = simulator.run()
    print(f"Latency: {latency}")

    output_graph = get_mesh_network(4, graph, mapping_list)
    visualize_noc_application(output_graph)    

    iso_test_path = "data/iso_test"
    os.makedirs(iso_test_path, exist_ok=True)  
    save_graph_to_json(output_graph, f"{iso_test_path}/graph.json")

    dataset = dataset.NocDataset(iso_test_path)
    data = dataset[0]

    output = model(data.x_dict, data.edge_index_dict)

    true, pred = get_max_latency_hetero( data, output )

    print(f"True latency: {int(true)}, Predicted latency: {int(pred)}")

    
    shutil.rmtree(iso_test_path)



