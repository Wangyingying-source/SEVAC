# BENCHMARK
Modify the map path, map_id, and test_name and run directly
Need to apply for a gurobi license

yang.py corresponds to the OSMIP algorithm
prakash.py corresponds to the DOT algorithm
fma.py corresponds to the FMA algorithm
PQL.py corresponds to the PQL algorithm

## experiment_benchmark
Includes experiments of GP3, PQL, CTD, and AC

## experiment_benchmark2
Includes experiments of FMA, DOT, OS-MIP, and ILP
map_id: Set 0~7 to represent Sioux Falls, Anaheim, Winnipeg, Chicago-Sketch, Chengdu-Weekend Off-peak Hour, Chengdu-Weekend Peak Hour, Chengdu-Weekday Off-peak Hour, and Chengdu-Weekday Peak Hour
OD_pairs: Save in the Networks directory
T_factor: coefficient of T_{let} before multiplication (0.95, 1, 1.05)
kappa: coefficient of generating variance in the paper (0.15, 0.25, 0.5)
K: maximum number of iterations of FMA
S: number of samples of PLM, ILP, OS-MIP
MaxIter: maximum number of iterations of PLM, ILP, OS-MIP
DOT_delta: window size of DOT

## calculate_sota
Used to convert txt data into csv data and calculate sota value

## generate_pass_prob
Used to generate pass probability

## handle_drl_data
Used to complete let in drl data



