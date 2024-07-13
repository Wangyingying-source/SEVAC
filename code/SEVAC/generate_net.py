import math
from imports import *
from load_data import load_data
def generate_all(source_file_path,target_file_path,disconnected_rate):
    data = load_data(source_file_path)
    uncertain_roads_number=math.floor(disconnected_rate*len(data))
    uncertain_roads=random.sample(range(0,len(data)),uncertain_roads_number)
    new_data=[]
    for i, edge in data.iterrows():
        sigma=random.uniform(0,0.4)*edge['Cost']
        edge['sigma']=sigma
        if i in uncertain_roads:
            edge['P']=random.random()
        else:
            edge['P']=1
        new_data.append(edge)
        print(f"{i}/{len(data)}")
    df = pd.DataFrame(new_data)
        
    print(df)
    df.to_csv(target_file_path, index=False)
