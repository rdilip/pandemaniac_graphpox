#cython: language_level=3
# TOD
# Port to graph-tool
# Markov chain monte carlo approach
# Come up with a separate algorithm involving centrality
# let it choose whatever 

# For large fights let's use chokepoints
# for small fights let's use some notion of centrality + communities?? 
# we still need a strategy for one v one against TAs.
# And I think it needs to be somewhat random...
import numpy as np
import sim
import networkx as nx
import sys
import json
import os
from itertools import cycle
from graph_io import write_seed_node_file, load_graph
import time


def clique_control(G,
                   N,
                   num_teams,
                   centrality="degree",
                   num_high_deg_nodes=3,
                   threshold_ix=None):
    """ The idea behind clique control is to 
    Args:
        G: networkx Graph
        N: number of seed nodes
        num_teams: int, number of teams playing
    Returns:
        infect: list of nodes
    """
    if type(G) is dict:
        G = nx.Graph(G)
    if threshold_ix is None:
        threshold_ix = int(0.3*N*num_teams)
    if centrality == "degree":
        cent = dict(nx.algorithms.degree_centrality(G))
    else:
        raise ValueError
    
    nodes, cs = np.array(list(cent.keys())), list(cent.values())
    deg = [d for _, d in nx.degree(G)]
    high_nodes = nodes[np.argsort(cs)[::-1]][:num_high_deg_nodes]
    # Ignore the first nodes that we think other teams are likely to choose
    threshold = np.sort(deg)[::-1][threshold_ix]
    
    infect = []
    for high_node in cycle(high_nodes):
        # We originally wanted high_node to be disconnected from any of the infected nodes... but this ends up
        # being actually pretty annoying to check and it turns out some nodes are mostly connected to every node?
        infect.append(high_node)
        cliques = nx.algorithms.clique.cliques_containing_node(G, nodes=high_node)

        u = _get_node_in_most_cliques(nodes, G.degree(), cliques, threshold, exclude=infect)
        infect.append(u)

        cliques_with_u = [c for c in cliques if u in c]
        v = _get_node_in_most_cliques(nodes, G.degree(), cliques_with_u, threshold, exclude=infect)
        infect.append(v)
        if len(infect) >= N:
            return infect[:N]


def _get_node_in_most_cliques(nodes, deg, cliques, threshold, exclude=None):
    counts = [u for clique in cliques for u in clique]
    freqs = np.bincount(counts)
    nodes_by_freqs = nodes[:len(freqs)][np.argsort(freqs)[::-1]] # less than G.nodes()
    for u in nodes_by_freqs[1:]:
        if deg[u] <= threshold and u not in exclude:
            return u
    return -1

def simulated_annealing(g, N, strategies, infect0=None, num_teams=None, ix=None):
    """ Implements annealing method.
    Args:
        G: Dictionary graph
        N: Number of seed nodes
        strategies: Each element of strategies should be a Callable that returns a set
            of seed nodes for the other players.
        infect0: initial guess
    Returns:
        infect: list of nodes to infect
    """
    start = time.time()
    if infect0 is None:
        infect0 = centrality_seeds(g, N, algorithm="closeness", keep=None,\
                ignore=None, weighted=False)

    nodes = {"infect": infect0.copy()}
    all_strats = [s() for s in strategies]

    number_of_nodes = len(g)

    for i in range(len(all_strats)):
        nodes[f"strategy{i}"] = all_strats[i]

    E_prev = np.inf
    Nruns, NT = 200, 100
    Ts = np.exp(-np.linspace(0, 10, NT)) 
    print(f"Initial energy: {_energy(g, nodes)}")
    random_numbers = np.random.rand(NT, Nruns)
    random_choices = np.random.randint(N, size=(NT, Nruns))

    for Tix in range(NT):
        for run in range(Nruns):
            all_strats = [s() for s in strategies]
            replace_ix = random_choices[Tix, run]
            replace = nodes["infect"][replace_ix]
            # replace_with = np.random.choice(g[nodes["infect"][replace_ix]]) 
            replace_with = np.random.choice(number_of_nodes)
            nodes["infect"][replace_ix] = replace_with

            for i in range(len(all_strats)):
                nodes[f"strategy{i}"] = all_strats[i]
            E = _energy(g, nodes)
            dE = E - E_prev
            P = np.exp(-dE / Ts[Tix])

            if random_numbers[Tix, run] < min(P, 1): # accept w probability p
                E_prev = E
            else:
                nodes["infect"][replace_ix] = replace

            print(f"Energy at T={Ts[Tix]}, iter={run}: {E_prev}", end='\r')
        if num_teams is not None and ix is not None:
            write_seed_node_file(nodes["infect"], num_teams, N, ix)
    print()
    print("Elapsed time: " + str(time.time() - start))
    print()
    return nodes["infect"], E_prev

def _energy(G, nodes):
    performance = sim.run(G, nodes)
    return 1. - performance["infect"] / np.sum(list(performance.values()))

def centrality(G, algorithm="degree", **kwargs):
    centrality_algorithms = {
                                "degree": nx.degree_centrality,
                                "closeness": nx.closeness_centrality,
                                "betweenness": nx.betweenness_centrality,
                                "harmonic": nx.harmonic_centrality,
                                "pagerank": nx.pagerank,
                                "percolation": nx.percolation_centrality,
                                "second_order": nx.second_order_centrality,
                            }
    return centrality_algorithms[algorithm](G, **kwargs)

def centrality_seeds(g, N, 
                    algorithm="degree",
                    keep=0.1,
                    ignore=None,
                    weight=None,
                    return_max=False):
    """ Returns a sampling of nodes based on a specified centrality measure
    Args:
        g: Graph as a dictionary of edges
        N: Number of seed nodes
        algorithm: str, one of the options in centrality()
        keep: float, only samples from the top fraction of centralities.
        ignore: float > 0. Will ignore nodes with a centrality higher than (1 - ignore) *
            the maximal centrality of the graph.
        weighted: str, if not None then uses a weighed probability measure. Either 'centrality' or `linear'
    """
    G = nx.Graph(g)
    cdict = centrality(G, algorithm=algorithm)
    clist = np.array([v for _, v in cdict.items()])
    max_centrality = np.max(clist)
    ordering = np.argsort(clist)[::-1]
    
    nodes_to_ignore, nodes_to_keep = 0, N
    if keep is not None:
        nodes_to_keep = int(keep * G.number_of_nodes())
    if ignore is not None:
        nodes_to_ignore = int(ignore * G.number_of_nodes())

    all_nodes_by_centrality = np.array(list(G.nodes()))[ordering]
    nodes_by_centrality = all_nodes_by_centrality[nodes_to_ignore:nodes_to_keep]
    centralities = clist[ordering][nodes_to_ignore:nodes_to_keep]

    top_node = all_nodes_by_centrality[0]
    if return_max:
        return all_nodes_by_centrality[:N]


    p = np.ones(len(centralities))
    # Need to change this weightig mechanism...
    if weight == "centrality":
        p = centralities 
    elif weight == "linear":
        p = np.arange(len(centralities), dtype=float)[::-1]
    p /= np.sum(p)

    infect = [top_node] + list(np.random.choice(nodes_by_centrality, N-1, replace=False, p=p))
    return infect

def chokepoint(g, N, algorithm="degree"):
    G = nx.Graph(g)
    points = np.array([u for u in nx.articulation_points(G)])
    centralities = centrality(G, algorithm=algorithm)
    measures = [centralities[u] for u in points]
    nodes = list(points[np.argsort(measures)[::-1]][:N])
    return nodes

def community_control(g, N, alpha=1):
    G = nx.Graph(g)
    comms = nx.community.greedy_modularity_communities(G)
    max_comm = list(comms[0])
    H = G.subgraph(max_comm)
    deg_centralities = centrality(H, algorithm="degree")
    close_centralities = centrality(H, algorithm="closeness")
    measures = [close_centralities[u] for u in H.nodes()]
    top_N_nodes = np.array(max_comm)[np.argsort(measures)[::-1]][:int(alpha*N)]
    if alpha > 1:
        return list(top_N_nodes[:2]) + list(np.random.choice(top_N_nodes, N-2, replace=False))
    else:
        return top_N_nodes

def bridge_control(g, N):
    G = nx.Graph(g)
    bridges = np.array([i for i in nx.bridges(G)])
    deg =  centrality(G, algorithm="closeness")
    summed_degrees = [deg[u]+deg[v] for u, v in bridges]
    order = np.argsort(summed_degrees)[::-1]
    return bridges[order].ravel()[:N]

   
def random(g, N):
    G = nx.Graph(g)
    deg = [d for (_, d) in G.degree()]
    nodes = np.array(list(G.nodes()))[np.argsort(deg)]
    return np.random.choice([i for i in nodes[:2*N]], size=N)

def run_strategy(n, N, ix):
    g = load_graph(n, N, ix)

