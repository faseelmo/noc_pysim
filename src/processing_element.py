from enum import Enum
from packet import Packet


class TaskStatus(Enum):
    PENDING = "pending"         # require pending
    PROCESSING = "processing"   # computing
    READY_TO_PROCESS = "ready"  # require satisfied



from dataclasses import dataclass 
from typing import Optional


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


    def get_unique_required_packet_type(self) -> list[int]:
        packet_type_list = []
        for compute_task in self.compute_list: 
            for require in compute_task.require_list:
                if require.require_type_id not in packet_type_list:
                    packet_type_list.append(require.require_type_id)
        print(f"Unique packet types required in this PE: {packet_type_list}")
        return packet_type_list


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
        print(f"Incrementing flits to {packet.flits_transmitted}/{packet.size}")
        is_transmitted, task_type = packet.check_transmission_status()

        
        if is_transmitted:
            packet.update_location(self)
            self.update_task_info(task_type)

    def can_start_new_processing(self):
        """
        Checks if required packets for a task have been received
            Processing can only start if all required packets (w/ task_id) have been received
            * Room for optimization here

        """
        for compute_task in self.compute_list:
            readiness_check = []
            require_list_len = len(compute_task.require_list)
            for require in compute_task.require_list:
                if compute_task.status is TaskStatus.PENDING:
                    if require.received_packet_count == require.required_packets:
                        readiness_check.append(True)
                        print(f" - Type {require.require_type_id} packets for task {compute_task.task_id} satisfied")

            if len(readiness_check) == require_list_len:
                compute_task.status = TaskStatus.PROCESSING
                self.compute_is_busy = True
                print(f"Received all required packets for task {compute_task.task_id}")



    def reset_task(self, task: TaskInfo):
        task.current_processing_cycle = 0
        task.generated_packet_count += 1
        for require in task.require_list:
            require.received_packet_count = 0

    def update_task_status(self, task: TaskInfo):
        if task.generated_packet_count == task.expected_generated_packets:
            task.status = TaskStatus.PENDING
            self.compute_is_busy = False


    def increment_processing_cycle(self):
        """
        Increment the processing cycle for the task (in compute_list) that is processing
        if processing cycle condition is met
            for task 
                receive packets = 0
                generated packets += 1
                current processing cycle = 0

            if all packets have been generated
                status = done
        """
        # Checking if multiple tasks are processing at the same time
        processing_tasks = [task for task in self.compute_list if task.status == TaskStatus.PROCESSING]
        if len(processing_tasks) > 1:
            raise ValueError("Multiple tasks are processing at the same time")

        for compute_task in self.compute_list:
            if compute_task.status is TaskStatus.PROCESSING:
                compute_task.current_processing_cycle += 1  
                if compute_task.current_processing_cycle == compute_task.processing_cycles:
                    self.reset_task(compute_task)
                    self.update_task_status(compute_task)
                    print(
                        f"Generated {compute_task.generated_packet_count}/{compute_task.expected_generated_packets} " 
                        f"packets for task {compute_task.task_id}"
                    )
                else :
                    print(
                        f"Task {compute_task.task_id} is processing at cycle "
                        f"{compute_task.current_processing_cycle}/{compute_task.processing_cycles}"
                    )


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
    1. Single PE recieves packets from one PE [x]
        - If receives more than the required packets
            if computing is not busy, start processing
    2. Single PE recieves packets from multiple PEs 
        - Check for require type [task_id in packet] and require count
    3. Two tasks assigned to one PE in seuquential order 
    3. Two tasks assigned to one PE in parallel order
        - Both tasks require packets from the same PE
        - Both tasks require packets from different PEs
            > Check which task gets all the required packets first. 
            > Start processing the task that gets all the required packets first. 
            > If the other task gets all the required packets before the first 
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
        expected_generated_packets=2, 
        require_list=[
            RequireInfo(
                require_type_id=0, 
                required_packets=1), 
            RequireInfo(
                require_type_id=2, 
                required_packets=1), 

        ]
    )
    computing_list = [task_1]

    # The Processing Element
    pe_1 = ProcessingElement((0, 0), computing_list)

    # The Packet (src and dest does not matter for now)
    from packet import PacketStatus

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

    packet_list = [packet_1, packet_2]
    
    current_packet = packet_list.pop(0)

    max_cycle = 20
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

