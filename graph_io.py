""" I/O file for the Pandemaniac project """

import json
import numpy as np
import os
import pickle

def make_dir_structure():
    for folder in ["graphs", "seed_nodes"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

def load_graph(num_teams, Nseed_nodes, ix):
    with open(f"graphs/{num_teams}.{Nseed_nodes}.{ix}.json", "rb") as f:
        graph = json.load(f)
    return graph


def write_seed_node_file(seed_nodes, num_teams, Nseed_nodes, ix):
    """ Given a function that returns the seed nodes, writes the function to a file.
    Args:
        seed_node_fn: Either a callable or an iterable. If a callable, then
            seed_node() should return an Iterable of length Nseed_nodes
        num_teams: int, number of teams.
        Nseed_nodes: int, number of seed nodes
        ix: int, index
    Returns:
        None
    """
    all_seed_nodes = [seed_nodes for i in range(50)]
    if hasattr(seed_nodes, '__call__'):
        all_seed_nodes = [seed_nodes() for i in range(50)]

    with open(f"seed_nodes/{num_teams}.{Nseed_nodes}.{ix}.txt", "w+") as f:
        for node_set in all_seed_nodes:
            for node in node_set:
                f.write(node + "\n")


