## Code for "Epistemic Side Effects & Avoiding Them (Sometimes)"

This code is based on the code of the paper "Be Considerate Avoiding Negative Side Effects in Reinforcement Learning". You can find the code [here](https://github.com/praal/beconsiderate). 

## Installation instructions

The code has the following requirements: 

- Python3
- numpy
- matplotlib


## How to run the code

To run the code, first move to the *src* folder. There are multiple scripts to run in *tests* folder, and each one reproduces one of the experiments in the paper. 

- **Results of Table 1**
To run these experiments you need to execute the lines below (each line corresponds to a column of a table). 
```
python3 -m tests.epkitchen A  
```
```
python3 -m tests.epkitchen B  
```
```
python3 -m tests.epkitchen C  
```
```
python3 -m tests.wetkitchen
```
```
python3 -m tests.spoiledkitchen 