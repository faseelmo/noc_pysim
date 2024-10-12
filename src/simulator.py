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
    def __init__(self, num_rows:int, num_cols:int, debug_mode: bool = False):
        self._debug_mode    = debug_mode   
        self._num_rows      = num_rows
        self._num_cols      = num_cols
        self._num_pes       = num_rows * num_cols

        self._routers       = self._create_routers()
        self._pes           = self._create_pes()

        self._mapping_list  = []

        self._pe_done_count     = 0    
        self._pe_active_count   = 0

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
        for x in range(self._num_rows):
            for y in range(self._num_cols):
                router = Router( pos=(x, y), debug_mode=self._debug_mode )
                router_lookup[(x, y)] = router
        return router_lookup

    def _create_pes(self) -> dict[tuple[int, int], ProcessingElement]:
        pe_lookup = {}
        for x in range(self._num_rows):
            for y in range(self._num_cols):
                pe = ProcessingElement( xy=(x, y), debug_mode=self._debug_mode, router_lookup=self._routers )
                pe_lookup[(x, y)] = pe
        return pe_lookup

    def run(self) -> None:
        print(f"\nRunning simulation with {self._num_rows}x{self._num_cols} mesh PEs")

        assert self._mapping_list, "Tasks have not been assigned to PEs"

        cycle_count     = 0
        flit_list       = []

        while True: 
            print(f"\n>{cycle_count}")
            cycle_count += 1
            status_list = []

            for pe in self._pes.values():
                # PEs gets require flits directly from the router
                is_done = pe.process(None)
                status_list.append(is_done)

            for router in self._routers.values():
                flit_list = router.process( flit_list, self._routers, self._pes)

            if self.is_stop_condition_met(status_list, cycle_count):
                break

    def is_stop_condition_met(self, status_list: list[bool], cycle_count: int) -> bool:
        assert cycle_count < 1000, "Simulation did not finish in 1000 cycles"

        for status in status_list:
            if status == True:
                self._pe_done_count += 1

        if self._pe_done_count == self._pe_active_count:
            print(f"\nSimulation finished in {cycle_count} cycles")
            return True

        return False


if __name__ == "__main__":

    import random

    from src.processing_element import TaskInfo, RequireInfo

    random.seed(0)

    sim = Simulator(num_rows=3, num_cols=3, debug_mode=True)

    task_0  = TaskInfo(
                task_id                     = 0, 
                processing_cycles           = 4, 
                expected_generated_packets  = 1, 
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
                                                required_packets=1)], 
                is_transmit_task            = False, 
            )

    task_list = [task_0, task_1]

    mapping_list = sim.get_random_mapping(task_list)
    sim.map(mapping_list)
    sim.run()