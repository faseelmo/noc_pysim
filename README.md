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
To create random graphs, to run those graphs through a simulator and to split training and test dataset. 
```
source data/create_training_data.sh 
```
  
Alternatively, the above data generation pipeline can also be done individually
#### Graph Generation 
```
python3 -m data.create_graph_tasks --help # To see the arg list
```

#### Find Latency on generated graphs
```
python3 -m data.simulate_latency_on_graphs --help # To see the arg list 
```

#### Splitting train and test data
```
python3 -m data.create_test_data
```





