from enum import Enum
from packet import Packet


class TaskStatus(Enum):
    PENDING     =   "pending"      # require pending
    PROCESSING  =   "processing"   # computing


from dataclasses import dataclass 
from typing import Optional


@dataclass
class TASKDependency:
    require_id: int     # Task id of the task that requires the packets
    generate_id: int    # Task id of the task that generates the packets


@dataclass
class RequireInfo:
    require_type_id: int
    required_packets: int
    received_packet_count: int = 0


@dataclass
class TaskInfo:
    task_id: int 
    processing_cycles: int
    expected_generated_packets: int
    require_list : list[RequireInfo]
    current_processing_cycle: int = 0   
    generated_packet_count: int = 0
    sent_generated_packets: int = 0
    status: TaskStatus = TaskStatus.PENDING


class ProcessingElement:
    def __init__(
            self, 
            xy: tuple[int, int], 
            computing_list: list[TaskInfo]
        ):

        self.xy = xy 
        self.compute_list = computing_list
        self.compute_is_busy = False
        
        self.required_packet_types = self.get_unique_required_packet_type()
        self.dependency_list = self.check_inter_task_dependency()
        print(f"Dependency list: {self.dependency_list}")

    def get_unique_required_packet_type(self) -> list[int]:
        packet_type_list = []
        for compute_task in self.compute_list: 
            for require in compute_task.require_list:
                if require.require_type_id not in packet_type_list:
                    packet_type_list.append(require.require_type_id)
        print(f"Unique packet types required in this PE: {packet_type_list}")
        return packet_type_list

    def check_inter_task_dependency(self) -> list[TASKDependency]:
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
                    task_depend = TASKDependency(
                        generate_id=require.require_type_id,    # task that generates the packets
                        require_id=compute_task.task_id         # task that requires the packets
                    )
                    dependency_list.append(task_depend)

        return dependency_list

    def update_task_info(self, task_id: int):
        print(f"Updating task info for task {task_id}")
        for compute_task in self.compute_list:
            for require in compute_task.require_list:
                if require.require_type_id == task_id:
                    require.received_packet_count += 1
                    print(f"Task {compute_task.task_id} has received {require.received_packet_count}/{require.required_packets} packets of type {require.require_type_id}")
                    break

    def recieve_packets(self, packet: Packet):
        """
        Checks if the packet received is required by the PE
        Updates the packet status and location  
        """
        if packet.source_task_id not in self.required_packet_types:
            raise ValueError(f"Packet type {packet.source_task_id} not required in this PE")

        packet.increment_flits()
        print(f"{self} Incrementing flits to {packet.flits_transmitted}/{packet.size}")
        is_transmitted, task_type = packet.check_transmission_status()
        
        if is_transmitted:
            packet.update_location(self)
            self.update_task_info(task_type)

    def can_start_new_processing(self):
        """
        Checks if required packets for a task have been received
            Processing can only start if all required packets (w/ task_id) have been received
            > Room for optimization here
        """
        if self.compute_is_busy:
            return

        for compute_task in self.compute_list:
            readiness_check = []
            require_list_len = len(compute_task.require_list)
            for require in compute_task.require_list:
                if compute_task.status is TaskStatus.PENDING:
                    if require.received_packet_count == require.required_packets:
                        readiness_check.append(True)
                    print(
                        f" - Task {compute_task.task_id} type {require.require_type_id} packets "
                        f"({require.received_packet_count}/{require.required_packets})"
                    )

            if len(readiness_check) == require_list_len:
                compute_task.status = TaskStatus.PROCESSING
                self.compute_is_busy = True
                print(f"Received all required packets for task {compute_task.task_id} to start computing")

    def reset_task(self, compute_task: TaskInfo):
        compute_task.current_processing_cycle = 0
        compute_task.generated_packet_count += 1
        for require in compute_task.require_list:
            require.received_packet_count = 0

    def update_task_status(self, compute_task: TaskInfo):
        if compute_task.generated_packet_count == compute_task.expected_generated_packets:
            compute_task.status = TaskStatus.PENDING
            self.compute_is_busy = False

    
    def check_generate_for_inter_task_dependency(self, compute_task: TaskInfo): 
        """
        Check if the generated packets are required by other tasks in the same PE 
        """
        print(f"Checking generate for inter task dependency")
        for dependency in self.dependency_list:
            if dependency.generate_id == compute_task.task_id:
                print(f" ~ 1.Generate used by same PE")
                compute_task.sent_generated_packets += 1
    
        # append require count 
        for compute_task in self.compute_list:
            for require in compute_task.require_list:
                for dependency in self.dependency_list:
                    if dependency.generate_id == require.require_type_id:
                        require.received_packet_count += 1
                        print(f" ~ 2.Generate used by same PE")
                        print(
                            f"Task {compute_task.task_id} has received {require.received_packet_count}/{require.required_packets} "
                            f"packets of type {require.require_type_id}")

    def process_compute_task(self, compute_task: TaskInfo):
        if compute_task.status is TaskStatus.PROCESSING:
            compute_task.current_processing_cycle += 1  
            if compute_task.current_processing_cycle == compute_task.processing_cycles:
                self.reset_task(compute_task)
                self.update_task_status(compute_task)
                self.check_generate_for_inter_task_dependency(compute_task) # there's an issue here
                print(
                    f"Generated {compute_task.generated_packet_count}/{compute_task.expected_generated_packets} " 
                    f"packets for task {compute_task.task_id}"
                )
            else :
                print(
                    f"Task {compute_task.task_id} is processing at cycle "
                    f"{compute_task.current_processing_cycle}/{compute_task.processing_cycles}"
                )

    def increment_processing_cycle(self):
        """
        Checks if multiple tasks are processing at the same time
        Increment the processing cycle for the task (in compute_list) that is processing
        """

        processing_tasks = [task for task in self.compute_list if task.status == TaskStatus.PROCESSING]
        if len(processing_tasks) > 1:
            raise ValueError("Multiple tasks are processing at the same time")

        for compute_task in self.compute_list:
            self.process_compute_task(compute_task)

    def process(self, packet: Optional[Packet]):
        if packet is not None: 
            print(f"{self} is recieving packet {packet}")
            self.recieve_packets(packet)

        if not self.compute_is_busy:
            self.can_start_new_processing()   # status: pending     -> processing
        if self.compute_is_busy:
            self.increment_processing_cycle() # status: processing  -> done

    def __repr__(self):
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
    3. Two tasks assigned to one PE in seuquential order                
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

    # The Task 
    task_1 = TaskInfo(
        task_id=1, 
        processing_cycles=5, 
        expected_generated_packets=1, 
        require_list=[
            RequireInfo(
                require_type_id=0, 
                required_packets=1), 
            RequireInfo(
                require_type_id=2, 
                required_packets=1)
        ]
    )
    task_3 = TaskInfo(
        task_id=3,
        processing_cycles=4, 
        expected_generated_packets=3,
        require_list=[
            RequireInfo(
                require_type_id=1, 
                required_packets=1), 
            # RequireInfo(
            #     require_type_id=3, 
            #     required_packets=1), 
        ]
    )
    computing_list = [task_1, task_3]

    # The Processing Element
    pe_1 = ProcessingElement((0, 0), computing_list)

    # The Packet (src and dest does not matter for now)
    from packet import PacketStatus
    import copy 

    packet_1 = Packet(
        source_xy=(0, 0),
        dest_xy=(1, 1),
        source_task_id=0
    )

    packet_2 = Packet(
        source_xy=(0, 0),
        dest_xy=(1, 1),
        source_task_id=2
    )

    # packet_2 = copy.deepcopy(packet_1)
    # packet_4 = copy.deepcopy(packet_3)
    # packet_5 = copy.deepcopy(packet_3)

    packet_list = [packet_1, packet_2]
    
    current_packet = packet_list.pop(0)

    max_cycle = 50
    for cycle in range(max_cycle):
        print(f"\n> {cycle}")
        pe_1.process(current_packet)
        if not current_packet is None and current_packet.status is PacketStatus.IDLE:
            if len(packet_list) > 0:
                current_packet = packet_list.pop(0)
            else: 
                current_packet = None
        if pe_1.compute_is_busy:
            status = "Computing"
        else: 
            status = "Free"
        print(f"POST: {pe_1}\t{status}")

