
import networkx as nx

from src.simulator import Simulator, GraphMap
from data.utils import ( modify_graph_to_application_graph )

def test_symmetry(): 

    mesh_size = 4
    sim = Simulator( num_rows=mesh_size, 
                     num_cols=mesh_size, 
                     debug_mode=False, 
                     max_cycles=100 )

    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)

    graph = modify_graph_to_application_graph( graph, generate_range=(2,2), processing_time_range=(2,2) )
    task_list = sim.graph_to_task( graph )
    mapping = [ GraphMap( task_id=0, assigned_pe=( 0,0 ) ), 
                GraphMap( task_id=1, assigned_pe=( 2,1 ) ), 
                GraphMap( task_id=2, assigned_pe=( 1,2 ) ), 
                GraphMap( task_id=3, assigned_pe=( 3,3 ) ) ]
    mapping_list = sim.set_assigned_mapping_list( task_list, mapping )
    sim.map( mapping_list )
    latency_1 = sim.run()

    sim.clear()

    mapping = [ GraphMap( task_id=0, assigned_pe=( 3,3 ) ), 
                GraphMap( task_id=1, assigned_pe=( 1,2 ) ), 
                GraphMap( task_id=2, assigned_pe=( 2,1 ) ), 
                GraphMap( task_id=3, assigned_pe=( 0,0 ) ) ]

    task_list = sim.graph_to_task(graph)
    mapping_list = sim.set_assigned_mapping_list(task_list, mapping)
    sim.map(mapping_list)
    latency_2 = sim.run()

    assert latency_1 == latency_2