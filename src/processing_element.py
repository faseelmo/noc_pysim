from enum           import Enum
from dataclasses    import dataclass 
from typing         import Optional

from .packet        import Packet

class TaskStatus(Enum):
    IDLE        =   "idle"         # idle (waiting for packets)
    PROCESSING  =   "processing"   # computing
    DONE        =   "done"         # done processing


@dataclass
class TaskDependency:
    require_id:     int # Task id of the task that requires the packets
    generate_id:    int # Task id of the task that generates the packets


@dataclass
class RequireInfo:
    require_type_id:        int
    required_packets:       int
    received_packet_count:  int = 0


@dataclass
class TaskInfo:
    task_id:                    int 
    processing_cycles:          int
    expected_generated_packets: int         # number of packets to generate (constant)
    require_list :              list[RequireInfo]   
    current_processing_cycle:   int = 0     
    generated_packet_count:     int = 0     # number of packets generated (incremented)
    sent_generated_packets:     int = 0     # for packets required by other task in the same PE
    status:                     TaskStatus = TaskStatus.IDLE
    start_cycle:                int = None
    end_cycle:                  int = None


class ProcessingElement:
    def __init__(
            self, 
            xy                  : tuple [int, int], 
            computing_list      : list  [TaskInfo], 
            debug_mode          : bool  =  False, 
            shortest_job_first  : bool  = False
        ):

        self.xy                         = xy 
        self.compute_list               = computing_list
        self.compute_is_busy            = False
        self.shortest_job_first         = shortest_job_first    
        self.debug_mode                 = debug_mode
        self.current_processing_cycle   = 0   # Might have to move this to instantiation later

        
        self.required_packet_types  = self._get_unique_required_packet_type()
        self.dependency_list        = self._check_inter_task_dependency()

        self._debug_print(f"Dependency list: {self.dependency_list}")

    def _debug_print(self, string: str) -> None: 

        if self.debug_mode:
            print(string)

    def _increment_processing_cycle(self) -> None:
        """Increments the processing cycle for the PE"""

        self.current_processing_cycle += 1
        print(f"Cycle: {self.current_processing_cycle}")

    def _get_unique_required_packet_type(self) -> list[int]:

        packet_type_list = []

        for compute_task in self.compute_list: 
            for require in compute_task.require_list:
                if require.require_type_id not in packet_type_list:
                    packet_type_list.append(require.require_type_id)

        self._debug_print(f"Unique packet types required in this PE: {packet_type_list}")

        return packet_type_list


    def _check_inter_task_dependency(self) -> list[TaskDependency]:
        """
        Checking if any of the tasks require packets from the same PE
        """
        task_list = [] # creating a list of task ids
        for compute_task in self.compute_list:
            task_list.append(compute_task.task_id)

        dependency_list = [] 
        for compute_task in self.compute_list:
            for require in compute_task.require_list:
                # check if the required packet type is in the task list
                if require.require_type_id in task_list:

                    task_depend = TaskDependency(
                                    generate_id=require.require_type_id,    # task that generates the packets
                                    require_id=compute_task.task_id)        # task that requires the packets
            
                    dependency_list.append(task_depend)

        return dependency_list

    def _update_TaskInfo(self, task_id: int) -> None:
        """ 
        - Increment received_packet_count for all the tasks that require the packet
        - If this behavior is not desired, the function can be modified by returning 
          after the first increment.
        """

        for compute_task in self.compute_list:
            for require in compute_task.require_list:

                if require.required_packets == require.received_packet_count:
                    # skipping if required packets have been received
                    continue

                if require.require_type_id == task_id:
                    require.received_packet_count += 1
                    # return

    def _recieve_packets(self, packet: Packet) -> None:
        """
        Checks if the packet received is required by the PE
        Updates the packet status and location  
        """

        packet_source_task_id = packet.get_source_task_id()

        if packet_source_task_id not in self.required_packet_types:
            raise ValueError(f"Packet type {packet_source_task_id} not required in this PE")

        packet.increment_flits()
        self._debug_print(f"{self} Recieving flits (type: {packet_source_task_id}) {packet.get_flits_transmitted()}/{packet.get_size()}")
        is_transmitted, recieved_packet_task_id = packet.check_transmission_status()
        
        if is_transmitted:
            packet.update_location(self)
            self._update_TaskInfo(recieved_packet_task_id)

    def _reset_received_packet_task(self, compute_task: TaskInfo) -> None:
        """
        Resets the current processing cycle to 0
        """
        for require in compute_task.require_list:
            require.received_packet_count = 0

    def _can_start_new_processing(self) -> None:
        """
        Checks if all the required packets for a task have been received
            Processing can only start if all required packets (w/ task_id) have been received
            > Room for optimization here
        Also does scheduling based on the number of required packets. 
        Priority is given to the task that requires the least number of packets. 
        """

        tasks_ready_to_execute = []

        for compute_task in self.compute_list:
            readiness_check     = []

            require_list_len    = len(compute_task.require_list)

            if compute_task.expected_generated_packets == compute_task.sent_generated_packets:
                # if task has generated all the packets required for 
                # other tasks in the same PE skip that particular 
                # compute task  
                continue

            total_require_count = 0  # for scheduling
            for require in compute_task.require_list:
                total_require_count += require.required_packets
                if compute_task.status is TaskStatus.IDLE:
                    if require.received_packet_count == require.required_packets:
                        readiness_check.append(True)

            if len(readiness_check) == require_list_len:

                if self.shortest_job_first:
                    # For Shortest Job First Scheduling
                    tasks_ready_to_execute.append( (total_require_count, compute_task) ) 

                else:
                    # Randomly scheduling the task for processing
                    compute_task.status         = TaskStatus.PROCESSING
                    compute_task.start_cycle    = self.current_processing_cycle

                    self.compute_is_busy = True
                    self._reset_received_packet_task(compute_task)
                    self._debug_print(f"Scheduling (random) task {compute_task.task_id} for processing")

                    return 


        # Shortest Job First Scheduling 
        if self.shortest_job_first and  tasks_ready_to_execute:

            execute_task = min(tasks_ready_to_execute, key=lambda x: x[0])[1]
            execute_task.status = TaskStatus.PROCESSING 
            execute_task.start_cycle = self.current_processing_cycle

            if self.debug_mode:
                debug_tasks_ready_to_execute = [(task_info.task_id, count) for count, task_info in tasks_ready_to_execute]
                self._debug_print(f"Tasks ready to execute (id, require count): {debug_tasks_ready_to_execute}")

            self.compute_is_busy = True
            self._reset_received_packet_task(execute_task)
            self._debug_print(f"Scheduling (SJF) task {execute_task.task_id} for processing")


    def _update_task_as_complete(self, compute_task: TaskInfo) -> None:

        if compute_task.generated_packet_count == compute_task.expected_generated_packets:

            compute_task.status = TaskStatus.DONE
            compute_task.end_cycle = self.current_processing_cycle
            self.compute_is_busy = False

    
    def _check_generate_for_inter_task_dependency(self, current_task: TaskInfo) -> None:  
        """
        Check if the generated packets are required by other tasks in the same PE 
        """
        for dependency in self.dependency_list:
            # Incrementing the count of sent generated packets
            # if the generated packets are required by the same PE
            if dependency.generate_id == current_task.task_id:

                self._debug_print(f"Incrementing sent generated packets for task {current_task.task_id}")
                current_task.sent_generated_packets += 1

                break
    
        for task_in_compute_list in self.compute_list:
            # Incrementing the count of received packets 
            #   if the generated packets are required by a 
            #   different Task in the same PE
            for require in task_in_compute_list.require_list:

                if require.required_packets == require.received_packet_count:
                    continue

                if require.require_type_id == current_task.task_id:

                    require.received_packet_count += 1
                    self._debug_print(
                        f"Task {task_in_compute_list.task_id} has received {require.received_packet_count}/{require.required_packets} "
                        f"packets of type {require.require_type_id}")

                    return 

    def _process_compute_task(self, compute_task: TaskInfo) -> None:
        """
        Tasks generate packets here
        To do: There should be a way to check if the packets generated 
        by the processing element will be sent outside. 
        """
        if compute_task.status is TaskStatus.PROCESSING:
            compute_task.current_processing_cycle += 1  

            if compute_task.current_processing_cycle == compute_task.processing_cycles:

                self._debug_print(
                    f"Task {compute_task.task_id} is done processing "
                    f"{compute_task.current_processing_cycle}/{compute_task.processing_cycles}"
                )

                compute_task.generated_packet_count     += 1  
                compute_task.current_processing_cycle   = 0 

                self._update_task_as_complete(compute_task)

                self._debug_print(
                    f" -> Generated {compute_task.generated_packet_count}/{compute_task.expected_generated_packets} " 
                    f"packets in task {compute_task.task_id}"
                )

                self._check_generate_for_inter_task_dependency(compute_task) 

            else :
                self._debug_print(
                    f"Task {compute_task.task_id} is processing at cycle "
                    f"{compute_task.current_processing_cycle}/{compute_task.processing_cycles}"
                )

    def _process_tasks(self) -> None:
        """
        Checks if multiple tasks are processing at the same time
        Increment the processing cycle for the task (in compute_list) that is processing
        """

        processing_tasks = [task for task in self.compute_list if task.status == TaskStatus.PROCESSING]
        if len(processing_tasks) > 1:
            raise ValueError("Multiple tasks are processing at the same time")

        for compute_task in self.compute_list:
            self._process_compute_task(compute_task)

    def _get_packet_count(self) -> None:
        print(f"Require List:")
        for compute_task in self.compute_list:
            print(f" - Task {compute_task.task_id}")
            for require in compute_task.require_list:

                self._debug_print(
                    f"   • type {require.require_type_id} "
                    f"({require.received_packet_count}/{require.required_packets})"
                )

    def _check_task_requirements_met(self) -> bool:
        """
        Checks if all the tasks have generated the expected number of packets
        Also checks if the status of the task is done 
        This is useful in the simulate function to get the total cycle count required
        """

        for compute_task in self.compute_list:
            if (compute_task.expected_generated_packets != compute_task.generated_packet_count or

                compute_task.status is not TaskStatus.DONE):
                return False
        
        return True

    def process(self, packet: Optional[Packet]) -> bool:

        self._increment_processing_cycle()

        if packet is not None: 
            self._recieve_packets(packet)

        if not self.compute_is_busy:
            self._can_start_new_processing()    # status: IDLE     -> PROCESSING
            return 

        if self.compute_is_busy:
            self._process_tasks()               # status: PROCESSING  -> DONE

        self._get_packet_count() 

        if self._check_task_requirements_met(): # stops the simulation now 
            return True

    def __repr__(self) -> str:
        return f"PE({self.xy})"

if __name__ == "__main__":

    """
    Arguments to pass to ProcessingElement:    
    1. xy: tuple[int, int] - x, y coordinates of the PE
    2. Computing List 
        list[TaskInfo] - List of tasks assigned to the PE

    Receive Conditions 
    1. Single PE recieves packets from one PE                           [x]
        - If receives more than the required packets
            if computing is not busy, start processing
    2. Single PE recieves packets from multiple PEs                     [x]
        - Check for require type [task_id in packet] 
            and require count 
    3. Two tasks assigned to one PE in seuquential order                [x]
    4. Two tasks assigned to one PE in parallel order                   [x]
        - Both tasks require packets from the same PE
        - Both tasks require packets from different PEs
            > Check which task gets all the required packets first. 
            > Start processing the task that gets all the required 
                packets first. 
            > If the other task gets all the required packets 
            before the first 
            > task is done processing, wait for the first task to finish processing
            
    Generate Conditions 
    1. Check 
        if 
            the output port for the PE is busy, 
        else 
            send the packets to the Network Interface
    """

    from .packet import PacketStatus
    import copy 

    # Graph in Thesis Notes Pg. 13
    task_1 = TaskInfo(
        task_id=1, 
        processing_cycles=5, 
        expected_generated_packets=2, 
        require_list=[
            RequireInfo(
                require_type_id=0, 
                required_packets=3), 
            RequireInfo(
                require_type_id=2, 
                required_packets=2)
        ]
    )
    task_3 = TaskInfo(
        task_id=3,
        processing_cycles=4, 
        expected_generated_packets=3,
        require_list=[
            RequireInfo(
                require_type_id=1, 
                required_packets=2), 
        ]
    )
    computing_list = [task_1, task_3]

    pe_1 = ProcessingElement((0, 0), computing_list, debug_mode=True, )

    # The Packet (src and dest does not matter for now)
    packet_0 = Packet(
        source_xy=(0, 0),
        dest_xy=(1, 1),
        source_task_id=0
    )

    packet_2 = Packet(
        source_xy=(0, 0),
        dest_xy=(1, 1),
        source_task_id=2
    )

    packet_0_1 = copy.deepcopy(packet_0)
    packet_0_2 = copy.deepcopy(packet_0)
    packet_2_1 = copy.deepcopy(packet_2)

    packet_list = [packet_0, packet_0_1,  packet_2, packet_2_1, packet_0_2,]
    current_packet = packet_list.pop(0)  

    max_cycle = 50

    for cycle in range(max_cycle):
        print(f"\n> {cycle}")
        pe_1.process(current_packet)
        if not current_packet is None and current_packet.status is PacketStatus.IDLE:
            # IDLE means that the packet has been transmitted
            if len(packet_list) > 0:
                current_packet = packet_list.pop(0)
            else: 
                # if all packets in the packet list have been processed
                #  set the current packet to None to signify that there are no more packets
                current_packet = None
        if pe_1.compute_is_busy:
            status = "Computing"
        else: 
            status = "Free"
        print(f"POST: {pe_1}\t{status}")

