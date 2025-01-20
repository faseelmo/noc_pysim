
import networkx as nx
import random

from src.simulator import Simulator, GraphMap

def test_symmetry(): 

    r"""

       - 1 -
      /     \
     /       \ 
    0         3
     \       /
      \     /
       - 2 -
   
    """

    mesh_size = 4
    debug_mode = False
    sim = Simulator( num_rows=mesh_size, 
                     num_cols=mesh_size, 
                     debug_mode=debug_mode, 
                     max_cycles=1000 )

    graph = nx.DiGraph()


    proc_range = (2, 8)
    graph.add_node(0, processing_time=random.randint(*proc_range))
    graph.add_node(1, processing_time=random.randint(*proc_range))
    graph.add_node(2, processing_time=random.randint(*proc_range))
    graph.add_node(3, processing_time=random.randint(*proc_range), generate=2)

    require_range = (2, 10)
    graph.add_edge(0, 1, weight=random.randint(*require_range))  
    graph.add_edge(0, 2, weight=random.randint(*require_range))  
    graph.add_edge(1, 3, weight=random.randint(*require_range))  
    graph.add_edge(2, 3, weight=random.randint(*require_range))  

    task_list = sim.graph_to_task( graph )
    mapping = [ GraphMap( task_id=0, assigned_pe=( 1,1 ) ), 
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
                GraphMap( task_id=3, assigned_pe=( 1,1 ) ) ]

    task_list = sim.graph_to_task(graph)
    mapping_list = sim.set_assigned_mapping_list(task_list, mapping)
    sim.map(mapping_list)
    latency_2 = sim.run()

    print(f"Latency 1: {latency_1}, Latency 2: {latency_2}")

    assert latency_1 == latency_2

# test_symmetry()