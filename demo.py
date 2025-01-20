import networkx as nx
from src.simulator import Simulator

graph = nx.DiGraph()
graph.add_node( 0, processing_time=5 )
graph.add_node( 1, processing_time=4, generate=2 )
graph.add_edge( 0, 1, weight=2 )

sim = Simulator( num_rows=3, num_cols=3, debug_mode=False )
sim.graph_to_task( graph )
sim.get_random_mapping( do_map=True )

sim.run()
sim.get_tasks_status( show=True )


