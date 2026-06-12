# NOT IN USE - TOO SLOW!
# NOT IN USE - TOO SLOW!
# NOT IN USE - TOO SLOW!
# NOT IN USE - TOO SLOW!
# NOT IN USE - TOO SLOW!

import numpy as np

def prepare_group_cost(group_cost, cost):
    """
    Prepare group cost by calculating cumulative costs.
    Args:
        group_cost (numpy.ndarray): A 3D array for group costs.
        cost (numpy.ndarray): A 2D array of base costs.
    """
    L, L2 = cost.shape
    for i in range(L):
        for j in range(i + 1, L + 1):
            for k in range(L2 - (j - i - 1)):
                group_cost[i, j, k] = group_cost[i, j - 1, k] + cost[j - 1, k + j - i - 1]

def nn_dp(G, E, F, Cost, tmin, L, Nt):
    """
    Nearest Neighbor Dynamic Programming (nn_dp).
    Args:
        G (numpy.ndarray): 1D array for dynamic programming results.
        E (numpy.ndarray): 1D array to store indices.
        F (numpy.ndarray): 1D array to store previous indices.
        Cost (numpy.ndarray): 3D cost array.
        tmin (int): Minimum time.
        L (int): Length.
        Nt (int): Number of time steps.
    """
    G[0] = 0
    for i in range(tmin, L + 1):
        for k in range(Nt):
            for l in range(i - tmin + 1):
                new_val = G[l] + Cost[l, i, k]
                if new_val < G[i]:
                    G[i] = new_val
                    E[i] = k
                    F[i] = l
