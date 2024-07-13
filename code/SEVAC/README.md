## Intro
This repo includes the implementation of single agent navigation in uncertain topological networks, and it is still under refinement.

## Dataset
The dataset includes 5 networks slots. The data includes the starting and ending nodes for each edge, as well as its average value. The 'P' column represents the transit probability for probabilistic edges. For detailed adjustments, refer to the variations in the data sets under `./Networks/Chicago`.

## Dependencies(in `imports.py`)
- Python 3.9
- NumPy
- SciPy
- Pandas
- NetworkX
- random
- matplotlib
- time

## Description
The source code includes the following files:
- `main.py`, to modify the data file, learning rate, number of runs, and to run the complete code.
- `imports.py`, all the libraries in need.
- `load_data.py`, load the data file.
- `network_operations.py`, create the network and get the travel time by sampling.
- `simulated_annealing.py`, use simulated annealing algorithm to get the shortest time from Origin to Destination.
- `functions.py`, 包括VPG，VAC，off-policy VPG，SEVAC算法，以及将这些算法进行比较的函数compare
- `theta_operations.py`, update the theta for the experienced edges.
- `run_fai.py`,  collect trajectories according to theta
- `initialize_theta.py`,  如果路网较大，调用其中的函数来初始化θ来进行诱导
- `travel_simulate.py`,  通过模拟走过之前生成的路径来完成数据复用，实现'off-policy'
- `generate_net.py`,  根据原始路网数据生成含方差sigma以及通行概率P的数据完整的路网