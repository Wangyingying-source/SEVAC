import math
import random
import numpy as np
from network_operations import get_travel_time
from simulated_annealing import simulated_annealing

def initialize_theta(G, theta, shortest_path):
    all_edges = list(G.edges())
    for i in range(len(shortest_path)-1):
        specified_edge = (shortest_path[i], shortest_path[i+1])
        index = all_edges.index(specified_edge)
        theta[index]=1.5
    return theta


def initialize_theta2(G, origin, destination, theta, time_limit, shortest_path,alpha):
    shortest_path, shortest_time = simulated_annealing(G, origin, destination, 100, 0.95, 1000)
    time_limit = 1.1 * shortest_time  # 设置时间限制为最短时间

    time_budget=time_limit
    all_edges = list(G.edges())
    Tau_one=[]
    for i,node in enumerate(shortest_path):   
        if node== shortest_path[len(shortest_path)-1]:
            break
        current_node=node 
        if time_budget<0:
            break
        available_edges = []
        for u, v, data in G.out_edges(current_node, data=True):
            available_edges.append(v)

        # 如果没有可用的边，处理这个场景
        if not available_edges:
            break
        numerators=[]
        denominator=0
        fais=[]
        for executable_node in available_edges:
            specified_edge = (current_node, executable_node)
            edge_vector = np.zeros(len(all_edges))
            index = all_edges.index(specified_edge)
            edge_vector[index] = 1
            #print(edge_vector)
            fai=np.concatenate((time_budget*edge_vector,edge_vector),axis=None)
                
            fais.append(fai)
            numerator=np.exp(np.dot(theta,fai))
            numerators.append(numerator)
            denominator+=numerator
        probabilities = np.array(numerators) / denominator
        node_list=list(available_edges)
        chosen_index = node_list.index(shortest_path[i+1])
        next_node=shortest_path[i+1]
        chosen_fai=fais[chosen_index]
        travel_time=get_travel_time(G[current_node][next_node]['weight'], G[current_node][next_node]['sigma'])
        Tau_one.append([[current_node,time_budget],G[current_node][next_node],travel_time,available_edges,fais,chosen_fai,probabilities])
        

        time_budget-=travel_time
        current_node=next_node
    Tau_one.append([[current_node,time_budget],0,0,0,0,0,0])
    gradient=np.zeros(2*len(G.edges()))
    for state in Tau_one:
        if state==Tau_one[len(Tau_one)-1]:
            break
        tmp=state[5]
        index=0
        while index<len(state[3]):
            fai=state[4][index]
            prob=state[6][index]
            index+=1
            tmp-=prob*fai
        gradient+=tmp
    theta+=gradient*alpha

    return theta