from imports import *

def create_network(edges_data):
    G = nx.DiGraph()
    
    for _, edge in edges_data.iterrows():
        #print(edge)
        if edge['prob']>=1:
            G.add_edge(edge['From'], edge['To'], weight=edge['Cost'], sigma=edge['sigma'],
                    p=edge['prob'], edge_type='edge1')
            
        if edge['prob'] < 1:
            G.add_edge(edge['From'], edge['To'], weight=edge['Cost'], sigma=edge['sigma'],
                       p=edge['prob'], edge_type='edge2')
    
        
    return G

def get_travel_time(mu, sigma):
    lower, upper = 0, np.inf
    X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs()
