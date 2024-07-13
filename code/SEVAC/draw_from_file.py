import json
import matplotlib.pyplot as plt

def draw(file_path):
    with open(file_path) as f:
        data = json.load(f)

    # 读取数据并写入数组
    success_rates1=[]
    for item in data['VPG']:
        success_rates1.append(item)
    success_rates2=[]
    for item in data['VAC']:
        success_rates2.append(item)
    success_rates3=[]
    for item in data['off-policy VPG']:
        success_rates3.append(item)
    success_rates4=[]
    for item in data['SEVAC']:
        success_rates4.append(item)

    plt.plot(success_rates1,label='VPG')
    plt.plot(success_rates2,label='VAC')
    plt.plot(success_rates3,label='off-policy VPG')
    plt.plot(success_rates4,label='SEVAC')
    plt.show()