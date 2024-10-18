from src.processing_element import TaskInfo, RequireInfo, TransmitInfo
from src.simulator import Simulator, Map

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

    assert latency == 63 


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

    assert latency == 71

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
                                                    required_packets=2) ],
                ) 

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

    assert latency == 59


if __name__ == "__main__":

    # DEBUG_MODE = False
    DEBUG_MODE = True   

    # test_sim_simple(DEBUG_MODE)
    # test_sim_1(DEBUG_MODE)
    test_sim_2(DEBUG_MODE)
    pass