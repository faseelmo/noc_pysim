from src.simulator import Simulator
from data.utils import load_graph_from_json
from src.utils import get_graph_report

sim = Simulator(num_rows=3, num_cols=3)
graph = load_graph_from_json("data/test_sim_task.json")

sim.graph_to_task(graph)
sim.get_random_mapping(do_map=True)
sim.run()

get_graph_report(graph, sim.get_mapping_list(), show=True)