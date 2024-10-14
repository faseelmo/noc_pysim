from src.processing_element import TaskInfo, RequireInfo
from src.simulator import Simulator, Map


def test_sim_simple():
    """
    R(0,0) to R(2,2)
    """
    sim = Simulator(num_rows=3, num_cols=3, debug_mode=False, max_cycles=100)

    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 2, 
                require_list                = [], 
                is_transmit_task            = True, 
                transmit_id_list            = [1]
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

# test_sim_simple()


def test_sim_1(): 
    r"""
      1
     / 
    0
     \
      2
    """
    pass

def test_sim_2(): 
    r"""
    0 
     \
      2
     /
    1
    """
    pass