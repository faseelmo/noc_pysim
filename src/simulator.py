import networkx as nx

from dataclasses import dataclass

from src.router             import Router 
from src.processing_element import ProcessingElement, TaskInfo, RequireInfo, TransmitInfo

@dataclass 
class Map:
    task                : TaskInfo
    assigned_pe         : tuple[int, int] # (x, y)

    def __str__(self) -> str:
        return f"Task: {self.task.task_id} -> PE: {self.assigned_pe}"

@dataclass
class GraphMap:
    task_id     : int
    assigned_pe : tuple[int, int]

class Simulator: 
    def __init__(self, num_rows:int, num_cols:int, debug_mode: bool = False, max_cycles: int = 1000):
        self._debug_mode    = debug_mode   
        self._max_cycles    = max_cycles
        self._num_rows      = num_rows
        self._num_cols      = num_cols
        self._num_pes       = num_rows * num_cols

        self._routers       = self._create_routers()
        self._pes           = self._create_pes()

        self._task_list     = []
        self._mapping_list  = []

        self._pe_done_count     = 0    
        self._pe_active_count   = 0

        if self._debug_mode:
            self._visualizer = self._init_visualizer()

    def clear(self) -> None:
        self._pe_done_count     = 0    
        self._pe_active_count   = 0

        for pe in self._pes.values():
            pe.clear()

        for router in self._routers.values():
            router.clear()

        self._mapping_list.clear()
        self._task_list.clear()
        print("Simulation cleared. Ready for next run.")

    def run(self) -> int:
        assert self._mapping_list, "Tasks have not been assigned to PEs"
        self._debug_print(f"\nRunning simulation with {self._num_rows}x{self._num_cols} mesh PEs")

        cycle_count = 0
        
        while True: 

            self._debug_print(f"\n>{cycle_count}")

            cycle_count += 1
            status_list = [] # To check if simulation is done

            # Processing all the PEs
            for pe in self._pes.values():
                is_done = pe.process(None)
                status_list.append(is_done)

            
            # Process the output buffer of all the routers
            for router in self._routers.values():
                router.forward_output_buffer_flits( self._routers, self._pes )

            # Process the input buffer and receive of all the routers 
            for router in self._routers.values():
                router.process()

            if self._debug_mode:
                self._visualizer(cycle_count - 1)

            if self.is_stop_condition_met(status_list, cycle_count):
                return cycle_count - 1

    def graph_to_task(self, graph: nx.DiGraph) -> list[TaskInfo]:
        """
        Convert the graph to a list of TaskInfo objects. 
        In a transmit node, priority give to the task that requires the least number of packets.
        """
        task_list = []

        for node_id, node in graph.nodes(data=True):
            # Converting the graph to a list of TaskInfo objects
            predecessors = list(graph.predecessors(node_id))
            successors   = list(graph.successors(node_id))

            require_list    = []
            transmit_list   = []

            if len(predecessors) > 0:
                for predecessor in predecessors:
                    require_id      = predecessor
                    edge            = graph[predecessor][node_id]

                    if "weight" not in edge:
                        raise ValueError("Need to mention weight for all edges")

                    require_count   = graph[predecessor][node_id]["weight"]

                    require         = RequireInfo(
                                        require_type_id=require_id, 
                                        required_packets=require_count)
                    require_list.append(require)

            if "processing_time" not in node:
                raise ValueError("Need to mention processing_time for all nodes")

            is_transmit_node = False
            if len(successors) > 0:
                is_transmit_node = True
                for successor in successors:
                    transmit_id     = successor
                    edge            = graph[node_id][successor]

                    if "weight" not in edge:
                        raise ValueError("Need to mention weight for all edges")

                    transmit_count  = graph[node_id][successor]["weight"]
                    transmit        = TransmitInfo(
                                        id=transmit_id, 
                                        require=transmit_count)
                    transmit_list.append(transmit)

                # Sorting based on shortest transmit first
                transmit_list.sort(key=lambda transmit_info: transmit_info.require,)

                if "generate" in node:
                    raise ValueError( f"Node {node_id} should not have generate count." 
                                      f"Generate here is calculated based on edge weights.")

                generate_count = 0
                for successor in successors:
                    generate_count += graph[node_id][successor]["weight"]

            else: 
                if "generate" not in node:
                    raise ValueError("Need to mention generate count for terminal nodes")

                else: 
                    generate_count = node["generate"]

            task = TaskInfo(
                task_id                     = node_id, 
                processing_cycles           = node["processing_time"], 
                expected_generated_packets  = generate_count, 
                require_list                = require_list, 
                is_transmit_task            = is_transmit_node, 
                transmit_list               = transmit_list, 
            )

            task_list.append(task)

        self._task_list = task_list

        return task_list

    def get_random_mapping(self, tasks: list[TaskInfo] = None, do_map: bool = False) -> list[Map]:
        """
        One-to-one mapping of tasks to PE. 
        As of now, only one task per PE is supported. 
        """
        import random

        if not tasks : 
            tasks = self._task_list
            assert tasks, "Tasks have not been defined"

        self._debug_print(f"\nRandomly mapping {len(tasks)} tasks to PEs")

        mapping_list    = []
        list_of_pes     = list(self._pes.keys())

        for task in tasks:
            random_pe = random.choice(list_of_pes)
            list_of_pes.remove(random_pe)
            map = Map(task=task, assigned_pe=random_pe)
            mapping_list.append(map)
            self._debug_print(f"Mapping {map}")

        self._debug_print("")

        if do_map:
            self.map(mapping_list)

        return mapping_list

    def set_assigned_mapping_list(self, tasks: list[TaskInfo], mapping: list[ GraphMap ] ) -> list[Map]:
        """
        Get mapping list when tasks are defined as graphs. 
        """
        assert tasks, "Task list is empty"
        assert mapping, "Mapping list is empty"

        mapping_list = []

        for task in tasks: 
            for graph_map in mapping: 

                if task.task_id == graph_map.task_id:
                    pe  = graph_map.assigned_pe
                    map = Map(task=task, assigned_pe=pe)
                    mapping_list.append(map)
                    break

        return mapping_list 

    def map(self, mapping_list: list[Map]) -> None:
        """
        Assign tasks to PEs based on the mapping list. 
        """

        self._mapping_list      = mapping_list
        # router_order_list       = []

        for router in self._routers.values():
            router.set_mapping_list(mapping_list)

        active_pes = set() # Set will handle duplicates
        for map in mapping_list:
            pe = self._pes[map.assigned_pe]
            pe.assign_task([map.task])
            active_pes.add(map.assigned_pe)
            # router_order_list.append(map.assigned_pe)

        self._pe_active_count = len(active_pes)

        if self._debug_mode:
            self._visualizer.init_mapping(mapping_list)

    def _create_routers(self) -> dict[tuple[int, int], Router]:
        router_lookup = {}
        for x in range(self._num_cols):
            for y in range(self._num_rows):
                router = Router( pos=(x, y), debug_mode=self._debug_mode )
                router_lookup[(x, y)] = router
        return router_lookup

    def _create_pes(self) -> dict[tuple[int, int], ProcessingElement]:
        pe_lookup = {}
        for x in range(self._num_cols):
            for y in range(self._num_rows):
                pe = ProcessingElement( xy=(x, y), debug_mode=self._debug_mode, router_lookup=self._routers )
                pe_lookup[(x, y)] = pe
        return pe_lookup


    def _get_required_flit(self, flit_list: list, router: Router) -> list:
        """
        `flit_list` is a list of flits that are in the network.
        This function checks the `flit_list` and returns the flits that are required by `router`.
        Also updates the `flit_list` by removing the flits used by the `router`.
        """
        if len(flit_list) == 0:
            return []

        required_flit = []
        router_xy = router.get_pos()    
        for flit in flit_list:
            flit_next_router = flit.get_routing_info().x, flit.get_routing_info().y
            if flit_next_router == router_xy:
                required_flit.append(flit)

        for flit in required_flit:
            flit_list.remove(flit)

        return required_flit

    def get_mapping_list(self) -> list[Map]:    
        return self._mapping_list

    def get_tasks_status(self, show: bool= False) -> list[TaskInfo]:
        """
        Reports the start_cycle and end_cycle of each tasks in the mapping list. 
        Also returns compute_list for data.utils.visualize_application
        """
        compute_list = []

        if show or self._debug_mode:
            print("---------Final Report---------")
            print(f"Task \t PE \t Start \t End")


        for map in self._mapping_list:
            task        = map.task
            compute_list.append(task)
            if show or self._debug_mode:
                print
                print(f" {task.task_id}\t{map.assigned_pe} \t {task.start_cycle} \t {task.end_cycle}")

        return compute_list 

    def is_stop_condition_met(self, status_list: list[bool], cycle_count: int) -> bool:
        assert cycle_count < self._max_cycles, f"Simulation did not finish in {self._max_cycles} cycles"

        for status in status_list:
            if status == True:
                self._pe_done_count += 1

        if self._pe_done_count == self._pe_active_count:
            self._debug_print(f"\nSimulation finished in {cycle_count - 1} cycles")
            return True

        return False

    def _init_visualizer(self) -> None:
        from src.visualizer import Visualizer

        visualizer = Visualizer(
                        num_rows=self._num_rows, 
                        num_cols=self._num_cols, 
                        routers =self._routers,
                        pes     =self._pes)

        return visualizer

    def _debug_print(self, message: str) -> None:
        if self._debug_mode:
            print(message)

if __name__ == "__main__":

    import random

    from src.processing_element import TaskInfo, RequireInfo, TransmitInfo
    from src.utils              import ( visualize_noc_application, 
                                         get_mesh_network,  
                                         visualize_application )

    random.seed(0)

    mesh_size   = 4
    debug_mode  = False
    sim         = Simulator( num_rows=mesh_size, 
                             num_cols=mesh_size, 
                             debug_mode=debug_mode, 
                             max_cycles=1000 )

    graph         = nx.DiGraph()
    proc_range    = ( 2, 8 )
    require_range = ( 2, 10 )

    graph.add_node( 0, processing_time=random.randint( *proc_range ) )
    graph.add_node( 1, processing_time=random.randint( *proc_range ) )
    graph.add_node( 2, processing_time=random.randint( *proc_range ) )
    graph.add_node( 3, processing_time=random.randint( *proc_range ), generate=2 )

    graph.add_edge( 0, 1, weight=random.randint( *require_range ) )  
    graph.add_edge( 0, 2, weight=random.randint( *require_range ) )  
    graph.add_edge( 1, 3, weight=random.randint( *require_range ) )  
    graph.add_edge( 2, 3, weight=random.randint( *require_range ) )  

    visualize_application( graph )

    task_list = sim.graph_to_task( graph )
    
    """ Explicitly mapping the tasks to PEs """ 
    mapping = [ GraphMap( task_id=0, assigned_pe=( 1,1 ) ), 
                GraphMap( task_id=1, assigned_pe=( 2,1 ) ), 
                GraphMap( task_id=2, assigned_pe=( 1,2 ) ), 
                GraphMap( task_id=3, assigned_pe=( 3,3 ) ) ]

    mapping_list = sim.set_assigned_mapping_list( task_list, mapping )
    sim.map( mapping_list )

    """ Randomly mapping the tasks to PEs """
    # mapping_list = sim.get_random_mapping( task_list, do_map=True )

    execution_time = sim.run()
    sim.get_tasks_status( show=True )

    print( f"Execution Time is {execution_time} cycles" )

    output_graph = get_mesh_network( mesh_size, graph, mapping_list )
    visualize_noc_application( output_graph )
