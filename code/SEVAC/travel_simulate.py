import numpy as np
def travel_simulate(G,taus,theta,time_limit,origin,destination):
    #通过模拟走过之前的路径，得到结果，实现数据复用
    tmp=0
    all_edges = list(G.edges())
    target=0
    for tau in taus:
        if tau[len(tau)-1][0][1]>=0 and tau[len(tau)-1][0][0]==destination:
            path=[]
            probability_of_mu=1
            for element in tau:
                path.append(element[0][0])
                if element!=tau[len(tau)-1]:
                    probability_of_mu*=element[6][element[7]]
            #print(probability_of_mu)
            probability_of_pi=1
            time_budget=time_limit
            for i,node in enumerate(path):   
                if node== path[len(path)-1]:
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

                    fai=np.concatenate((time_budget*edge_vector,edge_vector),axis=None)
                    fais.append(fai)
                    numerator=np.exp(np.dot(theta,fai))
                    numerators.append(numerator)
                    denominator+=numerator
                probabilities = np.array(numerators) / denominator
                node_list=list(available_edges)
                chosen_index = node_list.index(path[i+1])
                probability_of_pi*=probabilities[chosen_index]
                next_node=path[i+1]
                chosen_fai=fais[chosen_index]
                travel_time=G[current_node][next_node]['weight']
                time_budget-=travel_time
                current_node=next_node
                one=chosen_fai
                for i in range(len(available_edges)):
                    one-=probabilities[i]*fais[i]
                tmp+=one
            bz=probability_of_pi/probability_of_mu#Π与μ选择此路径的概率的比值
            target+=bz*tmp


    return target

def travel_simulate_VAC(G,taus,theta,time_limit,origin,destination):
    sum_gradient=0
    
    all_edges = list(G.edges())

    target=0
    for tau in taus:
        
            path=[]
            probability_of_mu=1
            for element in tau:
                path.append(element[0][0])
                if element!=tau[len(tau)-1]:
                    probability_of_mu*=element[6][element[7]]
            #print(probability_of_mu)
            probability_of_pi=1
            time_budget=time_limit
            tmp=np.zeros(2*len(all_edges))
            for i,node in enumerate(path):   
                if node== path[len(path)-1]:
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
                    fai=np.concatenate((time_budget*edge_vector,edge_vector),axis=None)
                        
                    fais.append(fai)
                    numerator=np.exp(np.dot(theta,fai))
                    numerators.append(numerator)
                    denominator+=numerator
                probabilities = np.array(numerators) / denominator
                node_list=list(available_edges)
                chosen_index = node_list.index(path[i+1])
                probability_of_pi*=probabilities[chosen_index]
                next_node=path[i+1]
                chosen_fai=fais[chosen_index]
                travel_time=G[current_node][next_node]['weight']
                time_budget-=travel_time
                current_node=next_node
                tmp=chosen_fai
                for i in range(len(available_edges)):
                    tmp-=probabilities[i]*fais[i]
                sum_gradient+=tmp
            bz=probability_of_pi/probability_of_mu
            
            if tau[len(tau)-1][0][1]>=0 and tau[len(tau)-1][0][0]==destination:
                target+=0.5*bz*tmp
            else:
                target-=0.5*bz*tmp

    return target
