import networkx as nx

from dataclasses import dataclass

from src.router             import Router 
from src.processing_element import ProcessingElement, TaskInfo

@dataclass 
class Map:
    task                : TaskInfo
    assigned_pe         : tuple[int, int] # (x, y)

    def __str__(self) -> str:
        return f"Task: {self.task.task_id} -> PE: {self.assigned_pe}"

class Simulator: 
    def __init__(self, num_rows:int, num_cols:int, debug_mode: bool = False, max_cycles: int = 1000):
        self._debug_mode    = debug_mode   
        self._max_cycles    = max_cycles
        self._num_rows      = num_rows
        self._num_cols      = num_cols
        self._num_pes       = num_rows * num_cols

        self._routers       = self._create_routers()
        self._pes           = self._create_pes()

        self._mapping_list  = []

        self._pe_done_count     = 0    
        self._pe_active_count   = 0

        if self._debug_mode:
            self._visualizer = self._init_visualizer()

    def graph_to_task(self, graph: nx.DiGraph) -> list[TaskInfo]:
        """
        Convert the graph to a list of TaskInfo objects. 
        Have to do shortest job arbitration here
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
                    require_count   = graph[predecessor][node_id]["weight"]
                    require         = RequireInfo(
                                        require_type_id=require_id, 
                                        required_packets=require_count)
                    require_list.append(require)


            is_transmit_node = False
            if len(successors) > 0:
                is_transmit_node = True
                for successor in successors:
                    transmit_id     = successor
                    transmit_count  = graph[node_id][successor]["weight"]
                    transmit        = TransmitInfo(
                                        id=transmit_id, 
                                        require=transmit_count)
                    transmit_list.append(transmit)

                # Sorting based on shortest transmit first
                transmit_list.sort(key=lambda transmit_info: transmit_info.require,)


            task = TaskInfo(
                task_id                     = node_id, 
                processing_cycles           = node["processing_time"], 
                expected_generated_packets  = node["generate"], 
                require_list                = require_list, 
                is_transmit_task            = is_transmit_node, 
                transmit_list               = transmit_list, 
            )

            task_list.append(task)

        return task_list

    def get_random_mapping(self, tasks: list[TaskInfo]) -> list[Map]:
        """
        One-to-one mapping of tasks to PE. 
        As of now, only one task per PE is supported. 
        """
        import random

        print(f"\nRandomly mapping {len(tasks)} tasks to PEs")

        mapping_list    = []
        list_of_pes     = list(self._pes.keys())

        for task in tasks:
            random_pe = random.choice(list_of_pes)
            list_of_pes.remove(random_pe)
            map = Map(task=task, assigned_pe=random_pe)
            mapping_list.append(map)
            print(f"Mapping {map}")

        print()
        return mapping_list

    def map(self, mapping_list: list[Map]) -> None:
        """
        Assign tasks to PEs based on the mapping list. 
        """

        self._mapping_list      = mapping_list

        for router in self._routers.values():
            router.set_mapping_list(mapping_list)

        active_pes = set() # Set will handle duplicates
        for map in mapping_list:
            pe = self._pes[map.assigned_pe]
            pe.assign_task([map.task])
            active_pes.add(map.assigned_pe)

        self._pe_active_count = len(active_pes)

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
        Check the flit_list and return the flits that are required by the router.
        Also updates the flit_list by removing the required flits.
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

    def _draw_active_routers(self, cycle_count: int, color_map: dict, axes) -> None:
        from matplotlib import pyplot as plt
        from src.utils              import draw_router_status
        
        num_rows = len(axes)  # Assuming axes is a 2D array-like structure
    
        for router in self._routers.values():
            router_xy = router.get_pos()
            transformed_x = num_rows - 1 - router_xy[1]  # Transform y-coordinate
            transformed_y = router_xy[0]  # x-coordinate remains the same
            
            ax = axes[transformed_x, transformed_y]
            color_map = draw_router_status(router, color_map=color_map, ax=ax)
    
        plt.suptitle(f"Cycle: {cycle_count}")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # Remove spacing


    def run(self) -> int:
        assert self._mapping_list, "Tasks have not been assigned to PEs"
        print(f"\nRunning simulation with {self._num_rows}x{self._num_cols} mesh PEs")

        cycle_count         = 0
        current_flit_list   = []
        
        while True: 
            print(f"\n>{cycle_count}")
            cycle_count += 1
            status_list = [] # To check if simulation is done

            for pe in self._pes.values():
                is_done = pe.process(None)
                status_list.append(is_done)

            flits_for_next_cycle = []

            for router in self._routers.values():
                required_flit           = self._get_required_flit( current_flit_list, router )  
                flits_from_out_buffers  = router.process( required_flit, self._routers, self._pes)
                flits_for_next_cycle.extend(flits_from_out_buffers)

            current_flit_list = flits_for_next_cycle

            if self._debug_mode:
                self._visualizer(cycle_count)

            if self.is_stop_condition_met(status_list, cycle_count):
                return cycle_count - 1

    def is_stop_condition_met(self, status_list: list[bool], cycle_count: int) -> bool:
        assert cycle_count < self._max_cycles, f"Simulation did not finish in {self._max_cycles} cycles"

        for status in status_list:
            if status == True:
                self._pe_done_count += 1

        if self._pe_done_count == self._pe_active_count:
            print(f"\nSimulation finished in {cycle_count - 1} cycles")
            return True

        return False

    def _init_visualizer(self) -> None:
        from src.visualizer import Visualizer

        visualizer = Visualizer(
                        num_rows=self._num_rows, 
                        num_cols=self._num_cols, 
                        routers=self._routers)

        return visualizer


if __name__ == "__main__":

    import random

    from src.processing_element import TaskInfo, RequireInfo, TransmitInfo
    from data.utils import load_graph_from_json, visualize_graph

    random.seed(0)

    sim = Simulator(num_rows=3, num_cols=3, debug_mode=True)

    graph_path  = "data/test_task_graph.json"
    graph       = load_graph_from_json(graph_path)
    task_list   = sim.graph_to_task(graph)
    # visualize_graph(graph)
    # print(graph)
    exit()



    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [], 
                is_transmit_task            = True, 

                transmit_list               = [ TransmitInfo(
                                                    id = 1, 
                                                    require=1) ] )

    task_1  = TaskInfo(
                task_id                     = 1, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
                require_list                = [RequireInfo(
                                                require_type_id=0,
                                                required_packets=1)], 
                is_transmit_task            = False, 
            )

    task_list = [task_0, task_1]

    mapping_list = sim.get_random_mapping(task_list)
    sim.map(mapping_list)
    sim.run()