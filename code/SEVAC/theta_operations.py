def update_theta(G, selected_edges_probs, reward, alpha):
    for u, v, prob in selected_edges_probs:
        G[u][v]['theta'] += alpha * reward * (1 - prob)
