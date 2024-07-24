#### Note:  
When a packet is received (from outside the PE) or a packet is generated (inside the PE) and multiple tasks 
inside the Processing Element is dependent on this Packet, then the packet is copied to all the dependent 
tasks. Assumption here is that the cache is big enough to store the packet.  


#### Run all tests using  
```
python3 -m pytest tests
```  

### Training Data Generation 
#### Automated Approach
To create random graphs, to run those graphs through a simulator and to split training and test dataset
```
source data/create_training_data.sh 
```
> Note: Max number of nodes of the generated graph can be changed in the bash script above
  
Alternatively, the above data generation pipeline can also be done individually
##### &emsp;1. Generating Graphs  
```
python3 -m data.create_graph_tasks --help # To see the arg list
# refer data/create_training_data.sh for usage
```

##### &emsp;2. Finding Latency on generated graphs
```
python3 -m data.simulate_latency_on_graphs --help # To see the arg list 
```

##### &emsp;3. Splitting train and test data
```
python3 -m data.create_test_data
```

#### Inspect Generated Data 
```
python3 -m data.inspect_data
```

#### Training 
Training parameters can be modified in training/params.yaml  
Start the training using  
```
python3 -m training.train name_of_results_save_directory
```







