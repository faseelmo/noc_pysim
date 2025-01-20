### 0. Prerequisites
#### Setting up the environment 
If your python version is earlier than 3.9, consider updating a newer one. Follow tutorial [here](https://docs.python-guide.org/starting/install3/linux/#install3-linux). 


Run the following script 
```bash 
source venv_script.sh 
```

#### Run all tests using
```bash
python3 -m pytest tests
```

### 1. Usage 

```python 
# from demo.py
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
```


#### Output: 
```bash 
python3 demo.py 
---------Final Report---------
Task 	 PE 	 Start 	 End
 0	(1, 1) 	 1 	 18
 1	(0, 1) 	 34 	 42
```

For a detailed usage example, refer to the main function in [src/simulator.py](https://github.com/faseelmo/noc_pysim/blob/main/src/simulator.py).






