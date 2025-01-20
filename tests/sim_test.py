from src.processing_element import TaskInfo, RequireInfo, TransmitInfo
from src.simulator import Simulator, Map, GraphMap

def test_sim_simple(debug_mode: bool = False):
    """
    R(0,0) to R(2,2)
    """
    sim = Simulator(num_rows=3, num_cols=3, debug_mode=debug_mode, max_cycles=100)

    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 2, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_list               = [TransmitInfo(id=1, require=2)]
            )

    task_1  = TaskInfo(
                task_id                     = 1, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [RequireInfo(
                                                require_type_id=0,
                                                required_packets=2)], 
                is_transmit_task            = False, 
            )

    mapping_list = [ Map( task=task_0, assigned_pe=(0,0) ), 
                     Map( task=task_1, assigned_pe=(2,2) )]

    sim.map(mapping_list)
    latency = sim.run()

    assert latency == 56


def test_sim_1(debug_mode: bool = False):    
    r"""
        1 PE(1,2)
       /
      /2
     /
    0 PE(0,1)
     \
      \3
       \ 
        2 PE(0,2)
    """
    sim = Simulator(num_rows=3, num_cols=3, debug_mode=debug_mode, max_cycles=100)

    task_0  = TaskInfo(
                task_id                     = 0,
                processing_cycles           = 5, 
                expected_generated_packets  = 5,
                require_list                = [],
                is_transmit_task            = True, 
                transmit_list               = [ TransmitInfo(
                                                    id = 1, 
                                                    require = 2), 
                                                TransmitInfo(
                                                    id = 2,
                                                    require = 3) ] ) 

    task_1  = TaskInfo(
                task_id                     = 1, 
                processing_cycles           = 3,
                expected_generated_packets  = 2,
                require_list                = [RequireInfo(
                                                require_type_id=0,
                                                required_packets=2)],
                is_transmit_task            = False )
    
    task_2  = TaskInfo(
                task_id                     = 2, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [RequireInfo(
                                                require_type_id=0,
                                                required_packets=3)],
                is_transmit_task            = False )

    mapping_list = [ Map( task=task_0, assigned_pe=(0,1) ),
                     Map( task=task_1, assigned_pe=(1,2) ),
                     Map( task=task_2, assigned_pe=(1,0) ) ]

    sim.map(mapping_list)
    latency = sim.run() 

    assert latency == 68

def test_sim_2(debug_mode: bool = False): 
    r"""
    1 PE(1,2)
     \
      \2
       \
        0 PE(0,1)
       /
      /2
     /
     2 PE(0,2)
    """    
    sim = Simulator(num_rows=3, num_cols=3, debug_mode=debug_mode, max_cycles=100)

    task_0  = TaskInfo(
                task_id                     = 0,
                processing_cycles           = 5, 
                expected_generated_packets  = 1,
                require_list                = [ RequireInfo(
                                                    require_type_id=1, 
                                                    required_packets=2), 
                                                RequireInfo(
                                                    require_type_id=2,
                                                    required_packets=2) ] ) 

    task_1  = TaskInfo(
                task_id                     = 1, 
                processing_cycles           = 3,
                expected_generated_packets  = 2,
                require_list                = [],
                is_transmit_task            = True , 
                transmit_list               = [ TransmitInfo(
                                                    id=0,
                                                    require=2) ] )
    
    task_2  = TaskInfo(
                task_id                     = 2, 
                processing_cycles           = 3, 
                expected_generated_packets  = 2, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_list               = [ TransmitInfo(
                                                    id=0,
                                                    require=2) ] )

    mapping_list = [ Map( task=task_0, assigned_pe=(0,1) ),
                     Map( task=task_1, assigned_pe=(1,2) ),
                     Map( task=task_2, assigned_pe=(1,0) ) ]

    sim.map(mapping_list)
    latency = sim.run() 

    assert latency == 47


def test_graph_to_task_fn(): 
    r""""
    Test for shortest transmit is given priority first  
        1 PE(1,2)
       /
      /2
     /
    0 PE(0,1)
     \
      \3
       \ 
        2 PE(0,2)
    """
    import networkx as nx 

    graph = nx.DiGraph()
    graph.add_node(0, type="task", processing_time=5)
    graph.add_node(1, type="task", processing_time=3, generate=2)
    graph.add_node(2, type="task", processing_time=4, generate=1)

    # First defined edge has higher weight than the second  
    # So, when graph is converted, the order should be changed. 
    graph.add_edge(0, 2, weight=3)
    graph.add_edge(0, 1, weight=2)

    sim = Simulator(num_rows=3, num_cols=3, debug_mode=False, max_cycles=100)    
    task_list = sim.graph_to_task(graph)

    for task in task_list: 
        if task.transmit_list:
            
            for i in range( len(task.transmit_list) - 1 ): 
                # Checking if the transmit list is sorted in ascending order
                assert task.transmit_list[i].require <= task.transmit_list[i+1].require, "Transmit list is not sorted"



def test_sim_graph(debug_mode: bool = False): 
    r"""
                    1 PE(2,1)
                    |
                    | 4
                    |
              3     |
         2  --------0 PE(2,0)
      PE(0,0)       |
                    |
                    | 9
                    |
                    3 PE(1,0)
    
    
    """

    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(1, type="task", processing_time=4)
    graph.add_node(2, type="task", processing_time=3)
    graph.add_node(3, type="task", processing_time=5)
    graph.add_node(0, type="task", processing_time=8, generate=1)

    graph.add_edge(2, 0, weight=3)
    graph.add_edge(1, 0, weight=4)
    graph.add_edge(3, 0, weight=9)  

    sim         = Simulator(num_rows=3, num_cols=3, debug_mode=debug_mode, max_cycles=1000)
    task_list   = sim.graph_to_task(graph)

    graph_map   = [ GraphMap(task_id=0, assigned_pe=(2,0)), 
                    GraphMap(task_id=1, assigned_pe=(2,1)), 
                    GraphMap(task_id=2, assigned_pe=(0,0)), 
                    GraphMap(task_id=3, assigned_pe=(1,0)) ]
                     
    mapping_list = sim.set_assigned_mapping_list(task_list, graph_map)

    sim.map(mapping_list)
    latency = sim.run()

    assert latency == 97

if __name__ == "__main__":

    # DEBUG_MODE = False
    # DEBUG_MODE = True   

    # test_sim_simple(DEBUG_MODE)
    # test_sim_1(DEBUG_MODE)
    # test_sim_2(DEBUG_MODE)
    # test_graph_to_task_fn()

    # test_sim_graph(DEBUG_MODE)

    pass