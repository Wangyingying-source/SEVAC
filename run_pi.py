import math
import random
import numpy as np
from network_operations import get_travel_time
import time
import concurrent.futures
from joblib import Parallel, delayed


def run_pi_one(G, origin, destination, theta, time_limit, trajectory_number, current_iteration):
        current_node=origin
        Tau_one=[]
        time_budget=time_limit
        all_edges = list(G.edges())
        
        while(current_node!=destination):
            if time_budget<0:
                break
            available_edges = []
            for u, v, data in G.out_edges(current_node, data=True):
                #
                #random.seed(int(time.time() * 100000))
                if data['edge_type'] == 'edge1' or (data['edge_type'] == 'edge2' and random.random() < data['p']):
                    available_edges.append(v)

            # 如果没有可用的边，处理这个场景
            if not available_edges:
                break
            numerators=[]
            denominator=0

            phis=[]
            for executable_node in available_edges:
                #将该边映射到m维向量中
                specified_edge = (current_node, executable_node)
                edge_vector = np.zeros(len(all_edges))
                index = all_edges.index(specified_edge)
                edge_vector[index] = 1
                
                phi=np.concatenate((time_budget*edge_vector,edge_vector),axis=None)
                
                phis.append(phi)
                numerator=np.exp(np.dot(theta,phi))
                #if current_node==3:
                #    print(available_edges)
                if current_iteration>10:
                    numerator=np.exp(np.dot(theta,phi))
                numerators.append(numerator)
                denominator+=numerator

            probabilities = np.array(numerators) / np.sum(numerators)


            chosen_index = np.random.choice(range(len(available_edges)), p=probabilities)#随机选择边
            next_node= available_edges[chosen_index]
            chosen_fai=phis[chosen_index]
            travel_time=get_travel_time(G[current_node][next_node]['weight'], G[current_node][next_node]['sigma'])# 若要加入行程时间的不确定性，则使用该行
            #travel_time=G[current_node][next_node]['weight']
            #0:state;1:action;2:reward;3:当前状态的可用边;4:可用边对应的Φ;5:选择的边的Φ;6:可用边对应的选择概率
            Tau_one.append([[current_node,time_budget],G[current_node][next_node],travel_time,available_edges,phis,chosen_fai,probabilities,chosen_index])

            time_budget-=travel_time
            current_node=next_node
        #最后一个节点
        Tau_one.append([[current_node,time_budget],0,0,0,0,0,0,0])

        return Tau_one



def run_pi(G, origin, destination, theta, time_limit, trajectory_number, current_iteration):
    Tau=[]
    nums_core=8
    Tau=Parallel(n_jobs=nums_core)(delayed(run_pi_one)(G, origin, destination, theta, time_limit, trajectory_number, current_iteration) for _ in range(trajectory_number))
    
    return Tau