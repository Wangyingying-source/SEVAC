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
- `functions.py`, including VPG, VAC, off-policy VPG, SEVAC algorithms, and compare functions to compare these algorithms
- `theta_operations.py`, update the theta for the experienced edges.
- `run_fai.py`, collect trajectories according to theta
- `initialize_theta.py`, if the network is large, call the function to initialize theta for induction
- `travel_simulate.py`, complete data reuse by simulating the previously generated path to achieve 'off-policy'
- `generate_net.py`, generate a complete network with variance sigma and pass probability P according to the original network data