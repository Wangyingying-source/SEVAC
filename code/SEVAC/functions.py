from imports import *
from load_data import load_data
from network_operations import create_network, get_travel_time
from simulated_annealing import simulated_annealing
from theta_operations import update_theta
from run_pi import run_pi
from initialize_theta import initialize_theta
from travel_simulate import travel_simulate,travel_simulate_VAC
import json
from scipy.stats import norm

def VPG(G, iterations, alpha,start, end, time_budget,shortest_path):
    if time_budget!=0:
        time_limit=time_budget
    
    
    M=100 #假设每次遍历生成的轨迹数为10
    #time_limit=20 
    # 初始化 theta 值
    thetas=np.zeros(2*len(G.edges()))
    # 运行模拟
    success_rates = []
    success_count=0
    for episode in range(iterations):
        #对于较大的路网，需要使用最短路径来诱导
        if success_count==0:
            thetas=initialize_theta(G,thetas,shortest_path)
        
        taus=run_pi(G,start,end,thetas,time_limit,M,episode)#采集样本

        gradient=np.zeros(2*len(G.edges()))
        for tau in taus:
            #按时到达
            if tau[len(tau)-1][0][1]>=0 :
                if tau[len(tau)-1][0][0]==end:
                    success_count+=1
                tmp2=np.zeros(2*len(G.edges()))
                for state in tau:
                    if state==tau[len(tau)-1]:
                        break
                    tmp=state[5]
                    index=0
                    while index<len(state[3]):
                        fai=state[4][index]
                        prob=state[6][index]
                        index+=1
                        tmp-=prob*fai
                    tmp2+=tmp
                gradient+=tmp2
        gradient=gradient/M
        thetas+=alpha*gradient

        

        # 记录成功率
        episode+=1
        success_rate = success_count / (M*episode)
        print(f"Episode: {episode}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)
    #plt.plot(success_rates,label='VPG')
    #plt.show()
    return success_rates



def VAC(G, iterations, alpha, start, end, time_budget,shortest_path):
    if time_budget!=0:
        time_limit=time_budget
    
    
    M=100 #假设每次遍历生成的轨迹数
    
    #time_limit=15   #设置时间
    # 初始化 theta 值
    thetas=np.zeros(2*len(G.edges()))
    # 运行模拟
    success_rates = []
    success_count=0
    for episode in range(iterations):
        #对于较大的路网，需要使用最短路径来诱导
        if success_count==0:
            thetas=initialize_theta(G,thetas,shortest_path)
        
        taus=run_pi(G,start,end,thetas,time_limit,M,episode)#采集样本

        gradient=np.zeros(2*len(G.edges()))
        #对VAC算法进行了简化，若按时到达，系数为0.5；若没有系数为-0.5，这样来使得向量不稀疏
        for tau in taus:
            #未能按时到达
            if tau[len(tau)-1][0][1]<0 or tau[len(tau)-1][0][0]!=end:
                tmp2=np.zeros(2*len(G.edges()))
                for state in tau:
                    if state==tau[len(tau)-1]:
                        break
                    tmp=state[5]
                    index=0
                    while index<len(state[3]):
                        fai=state[4][index]
                        prob=state[6][index]
                        index+=1
                        tmp-=prob*fai
                    tmp2+=tmp
                gradient-=0.5*tmp2
            #按时到达
            if tau[len(tau)-1][0][1]>=0 and tau[len(tau)-1][0][0]==end:
                success_count+=1
                path=[]
                tmp2=np.zeros(2*len(G.edges()))
                for state in tau:
                    path.append(state[0][0])
                    if state==tau[len(tau)-1]:
                        break
                    tmp=state[5]
                    index=0
                    while index<len(state[3]):
                        fai=state[4][index]
                        prob=state[6][index]
                        index+=1
                        tmp-=prob*fai
                    tmp2+=tmp
                gradient+=0.5*tmp2
                #print(path)
        #print(gradient)
        gradient=gradient/M
        thetas+=2*alpha*gradient
        # 记录成功率
        episode+=1
        success_rate = success_count / (M*episode)
        print(f"Episode: {episode}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)
    
    return success_rates

def off_policy_VPG(G, iterations, alpha, start, end, time_budget,shortest_path):
    if time_budget!=0:
        time_limit=time_budget
    
    
    M=100 #假设每次遍历生成的轨迹数
    #time_limit=15   #设置时间
    # 初始化 theta 值
    thetas=np.zeros(2*len(G.edges()))
    # 运行模拟
    success_rates = []
    success_count=0
    taus_episode=[]
    taus_last=0
    for episode in range(iterations):
        #对于较大的路网，需要使用最短路径来诱导
        if success_count==0:
            thetas=initialize_theta(G,thetas,shortest_path)
        
        taus=run_pi(G,start,end,thetas,time_limit,M,episode)#采集样本
        taus_episode.append(taus)
        gradient=np.zeros(2*len(G.edges()))
        
        for tau in taus:
            #按时到达
            tmp2=np.zeros(2*len(G.edges()))
            if tau[len(tau)-1][0][1]>=0 and tau[len(tau)-1][0][0]==end:
                success_count+=1
                path=[]
                for state in tau:
                    path.append(state[0][0])
                    if state==tau[len(tau)-1]:
                        break
                    tmp=state[5]
                    index=0
                    while index<len(state[3]):
                        fai=state[4][index]
                        prob=state[6][index]
                        index+=1
                        tmp-=prob*fai
                    tmp2+=tmp
                gradient+=tmp2
        #数据复用
        if episode>=10:
            gradient+=travel_simulate(G,taus_last,thetas,time_limit,start,end)
        taus_last=taus
        gradient=gradient/M
        thetas+=alpha*gradient
        # 记录成功率
        episode+=1
        success_rate = success_count / (M*episode)
        print(f"Episode: {episode}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)
    
    return success_rates

def SEVAC(G, iterations, alpha, start, end, time_budget,shortest_path):
    if time_budget!=0:
        time_limit=time_budget
    
    
    M=100 #假设每次遍历生成的轨迹数为10
    # 初始化 theta 值
    thetas=np.zeros(2*len(G.edges()))
    # 运行模拟
    success_rates = []
    success_count=0
    taus_episode=[]
    taus_last=0
    for episode in range(iterations):
        #对于较大的路网，需要使用最短路径来诱导
        if success_count==0:
            thetas=initialize_theta(G,thetas,shortest_path)
        
        taus=run_pi(G,start,end,thetas,time_limit,M,episode)#采集样本
        taus_episode.append(taus)
        gradient=np.zeros(2*len(G.edges()))
        
        for tau in taus:
            #按时到达
            tmp2=np.zeros(2*len(G.edges()))
            if tau[len(tau)-1][0][1]>=0 and tau[len(tau)-1][0][0]==end:
                success_count+=1
                path=[]
                
                for state in tau:
                    path.append(state[0][0])
                    if state==tau[len(tau)-1]:
                        break
                    tmp=state[5]
                    index=0
                    while index<len(state[3]):
                        fai=state[4][index]
                        prob=state[6][index]
                        index+=1
                        tmp-=prob*fai
                    tmp2+=tmp
                gradient+=0.5*tmp2
            if tau[len(tau)-1][0][1]<0 or tau[len(tau)-1][0][0]!=end:
                tmp2=np.zeros(2*len(G.edges()))
                for state in tau:
                    if state==tau[len(tau)-1]:
                        break
                    tmp=state[5]
                    index=0
                    while index<len(state[3]):
                        fai=state[4][index]
                        prob=state[6][index]
                        index+=1
                        tmp-=prob*fai
                    tmp2+=tmp
                gradient-=0.5*tmp2

        
        thetas+=2*alpha*gradient

        if episode>=10:
            gradient+=travel_simulate_VAC(G,taus_last,thetas,time_limit,start,end)
        taus_last=taus
        gradient=gradient/M
        thetas+=2*alpha*gradient
        # 记录成功率
        episode+=1
        success_rate = success_count / (M*episode)
        print(f"Episode: {episode}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)
    return success_rates

def compare(network, iteration, alpha):
    file_path='Networks//'+network+'//'+network+'_network.csv'
    od_path='Networks//'+network+'//'+network+'_OD.csv'
    data = load_data(file_path)
    G = create_network(data)
    ods = pd.read_csv(od_path)
    probs_VPG=[]
    probs_VAC=[]
    probs_off_policy_VPG=[]
    probs_SEVAC=[]
    output_VPG=[]
    output_VAC=[]
    output_off_policy_VPG=[]
    output_SEVAC=[]
    p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=0
    times=0
    for _, od in ods.iterrows():
        flag=True
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12=pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series()
        start=od['O']
        end=od['D']
        try:
            shortest_path=nx.shortest_path(G,start,end,weight='weight')
            shortest_time=sum(G[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
            shortest_path_sigma=sum(G[u][v]['sigma'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
            time_k=1
            time_budget=time_k*shortest_time
            success_rates1=VPG(G,iteration,alpha,start, end, time_budget,shortest_path)
            success_rates2=VAC(G,iteration,alpha,start, end, time_budget,shortest_path)
            success_rates3=off_policy_VPG(G,iteration,alpha,start,end,time_budget,shortest_path)
            success_rates4=SEVAC(G,iteration,alpha, start, end, time_budget,shortest_path)
            success_rates5=VPG(G,iteration,alpha,start, end, time_budget*0.95,shortest_path)
            success_rates6=VAC(G,iteration,alpha,start, end, time_budget*0.95,shortest_path)
            success_rates7=off_policy_VPG(G,iteration,alpha,start,end,time_budget*0.95,shortest_path)
            success_rates8=SEVAC(G,iteration,alpha, start, end, time_budget*0.95,shortest_path)
            success_rates9=VPG(G,iteration,alpha,start, end, time_budget*1.05,shortest_path)
            success_rates10=VAC(G,iteration,alpha,start, end, time_budget*1.05,shortest_path)
            success_rates11=off_policy_VPG(G,iteration,alpha,start,end,time_budget*1.05,shortest_path)
            success_rates12=SEVAC(G,iteration,alpha, start, end, time_budget*1.05,shortest_path)
        except:
            flag=False

        if flag:
            times+=1
            o1['od']=o2['od']=o3['od']=o4['od']=o5['od']=o6['od']=o7['od']=o8['od']=o9['od']=o10['od']=o11['od']=o12['od']=[start,end]
            o1['tf']=o2['tf']=o3['tf']=o4['tf']=1
            o5['tf']=o6['tf']=o7['tf']=o8['tf']=0.95
            o9['tf']=o10['tf']=o11['tf']=o12['tf']=1.05
            o1['let']=o2['let']=o3['let']=o4['let']=o5['let']=o6['let']=o7['let']=o8['let']=o9['let']=o10['let']=o11['let']=o12['let']=shortest_time
            o1['prob']=o2['prob']=o3['prob']=o4['prob']=norm.cdf(shortest_time,shortest_time,shortest_path_sigma)
            o5['prob']=o6['prob']=o7['prob']=o8['prob']=norm.cdf(shortest_time*0.95,shortest_time,shortest_path_sigma)
            o9['prob']=o10['prob']=o11['prob']=o12['prob']=norm.cdf(shortest_time*1.05,shortest_time,shortest_path_sigma)
            o1['sota']=success_rates1[len(success_rates1)-1]
            o2['sota']=success_rates2[len(success_rates2)-1]
            o3['sota']=success_rates3[len(success_rates3)-1]
            o4['sota']=success_rates4[len(success_rates4)-1]
            o5['sota']=success_rates5[len(success_rates5)-1]
            o6['sota']=success_rates6[len(success_rates6)-1]
            o7['sota']=success_rates7[len(success_rates7)-1]
            o8['sota']=success_rates8[len(success_rates8)-1]
            o9['sota']=success_rates9[len(success_rates9)-1]
            o10['sota']=success_rates10[len(success_rates10)-1]
            o11['sota']=success_rates11[len(success_rates11)-1]
            o12['sota']=success_rates12[len(success_rates12)-1]

            output_VPG.append(o1)
            output_VPG.append(o5)
            output_VPG.append(o9)
            output_VAC.append(o2)
            output_VAC.append(o6)
            output_VAC.append(o10)
            output_off_policy_VPG.append(o3)
            output_off_policy_VPG.append(o7)
            output_off_policy_VPG.append(o11)
            output_SEVAC.append(o4)
            output_SEVAC.append(o8)
            output_SEVAC.append(o12)
            
            p1+=success_rates1[len(success_rates1)-1]
            p2+=success_rates2[len(success_rates2)-1]
            p3+=success_rates3[len(success_rates3)-1]
            p4+=success_rates4[len(success_rates4)-1]
            p5+=success_rates5[len(success_rates5)-1]
            p6+=success_rates6[len(success_rates6)-1]
            p7+=success_rates7[len(success_rates7)-1]
            p8+=success_rates8[len(success_rates8)-1]
            p9+=success_rates9[len(success_rates9)-1]
            p10+=success_rates10[len(success_rates10)-1]
            p11+=success_rates11[len(success_rates11)-1]
            p12+=success_rates12[len(success_rates12)-1]
    p1=p1/times
    p2=p2/times
    p3=p3/times
    p4=p4/times
    p5=p5/times
    p6=p6/times
    p7=p7/times
    p8=p8/times
    p9=p9/times
    p10=p10/times
    p11=p11/times
    p12=p12/times
    
    pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10,pp11,pp12=pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series(),pd.Series()
    pp1['tf']=1.0
    pp2['tf']=1.0
    pp3['tf']=1.0
    pp4['tf']=1.0
    pp5['tf']=0.95
    pp6['tf']=0.95
    pp7['tf']=0.95
    pp8['tf']=0.95
    pp9['tf']=1.05
    pp10['tf']=1.05
    pp11['tf']=1.05
    pp12['tf']=1.05
    pp1['prob']=p1
    pp2['prob']=p2
    pp3['prob']=p3
    pp4['prob']=p4
    pp5['prob']=p5
    pp6['prob']=p6
    pp7['prob']=p7
    pp8['prob']=p8
    pp9['prob']=p9
    pp10['prob']=p10
    pp11['prob']=p11
    pp12['prob']=p12

    probs_VPG.append(pp1)
    probs_VPG.append(pp5)
    probs_VPG.append(pp9)
    probs_VAC.append(pp2)
    probs_VAC.append(pp6)
    probs_VAC.append(pp10)
    probs_off_policy_VPG.append(pp3)
    probs_off_policy_VPG.append(pp7)
    probs_off_policy_VPG.append(pp11)
    probs_SEVAC.append(pp4)
    probs_SEVAC.append(pp8)
    probs_SEVAC.append(pp12)

    output1=pd.DataFrame(output_VPG)
    output2=pd.DataFrame(output_VAC)
    output3=pd.DataFrame(output_off_policy_VPG)
    output4=pd.DataFrame(output_SEVAC)

    probs1=pd.DataFrame(probs_VPG)
    probs2=pd.DataFrame(probs_VAC)
    probs3=pd.DataFrame(probs_off_policy_VPG)
    probs4=pd.DataFrame(probs_SEVAC)

    output1.to_csv('Networks//'+network+'//csv//'+network+'_VPG.csv',index=False)
    output2.to_csv('Networks//'+network+'//csv//'+network+'_VAC.csv',index=False)
    output3.to_csv('Networks//'+network+'//csv//'+network+'_off_policy_VPG.csv',index=False)
    output4.to_csv('Networks//'+network+'//csv//'+network+'_SEVAC.csv',index=False)

    probs1.to_csv('Networks//'+network+'//csv//'+network+'_VPG_mean_prob.csv',index=False)
    probs2.to_csv('Networks//'+network+'//csv//'+network+'_VAC_mean_prob.csv',index=False)
    probs3.to_csv('Networks//'+network+'//csv//'+network+'_off-policy_VPG_mean_prob.csv',index=False)
    probs4.to_csv('Networks//'+network+'//csv//'+network+'_SEVAC_mean_prob.csv',index=False)

    #data={"VPG":success_rates1,'VAC':success_rates2,'off-policy VPG':success_rates3,'SEVAC':success_rates4}
    #plt.plot(success_rates1,label='VPG')
    #plt.plot(success_rates2,label='VAC')
    #plt.plot(success_rates3,label='off-policy VPG')
    #plt.plot(success_rates4,label='SEVAC')
    #plt.legend()
    #plt.show()