#### Note:
When a packet is received (from outside the PE) or a packet is generated (inside the PE) and multiple tasks
inside the Processing Element is dependent on this Packet, then the packet is copied to all the dependent
tasks. Assumption here is that the cache is big enough to store the packet.

### 0. Prerequisites
#### Setting up the environment 
```bash 
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
source data/create_training_data.sh
```
> Note: Max number of nodes of the generated graph can be changed in the bash script above

Alternatively, the above data generation pipeline can also be done individually
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
python3 -m training.train name_of_results_save_directory
```


