#### Note:  
When a packet is received (from outside the PE) or a packet is generated (inside the PE) and multiple tasks 
inside the Processing Element is dependent on this Packet, then the packet is copied to all the dependent 
tasks. Assumption here is that the cache is big enough to store the packet.  


#### Run all tests using  
```
python3 -m pytest tests
```  

  
#### Generate Graphs using   
```
python3 -m data.create_data --help # To see the arg list
python3 -m data.create_data   
```



