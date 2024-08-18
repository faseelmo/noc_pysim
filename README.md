#### Note:
When a packet is received (from outside the PE) or a packet is generated (inside the PE) and multiple tasks
inside the Processing Element is dependent on this Packet, then the packet is copied to all the dependent
tasks. Assumption here is that the cache is big enough to store the packet.  

The *require* and *generate* of nodes/tasks in the graph should match. Otherwise, there could be weird behavior observed from the simulator. 

### 0. Prerequisites
#### Setting up the environment 
```bash 
unset PYTHONPATH  # to avoid conflits with system Python packages
python3 -m venv "venv"
source venv/bin/activate
pip install -r requirements.txt
```


#### Run all tests using
```bash
python3 -m pytest tests
```

### 1. Data Generation
#### Automated Approach
To create random graphs, to run those graphs through a simulator and to split training and test dataset
```bash
./data/create_training_data.sh GEN_COUNT NUM_NODES
```
> *GEN_COUNT* is the total number of graphs that'll be generated.  
> *NUM_NODES* is the max number of node that'll be possible for the generated graphs. 

Alternatively, the above data generation pipeline can also be executed individually as follows 
##### &emsp;1. Generating Graphs
```bash
python3 -m data.create_graph_tasks --help # To see the arg list
# refer data/create_training_data.sh for usage
```

##### &emsp;2. Finding Latency on generated graphs
```bash
python3 -m data.simulate_latency_on_graphs --help # To see the arg list
```

##### &emsp;3. Splitting train and test data
```bash
python3 -m data.create_test_data
```

#### Inspect Generated Data
```bash
python3 -m data.inspect_data
python3 -m data.histogram_data # To see the frequency of number
                               # of nodes in test and training data
```

### 2. Training
Parameters for training can be modified in training/params.yaml
Start the training using
```bash
python3 -m training.train directory_name
```
> Note: Results will be saved in training/results/directory_name 


