from collections import defaultdict
from qtensor.tools.lazy_import import pynauty
import networkx as nx
from multiprocessing import Pool
from functools import partial
import psutil

from qtensor.utils import get_edge_subgraph

def get_adjacency_dict(G):
    """Returns adjacency dictionary for G
    G must be a networkx graph
    Return format: { n : [n1,n2,...], ... }
    where [n1,n2,...] is a list of neighbors of n
    """
    adjacency_dict = {}
    for n, neigh_dict in G.adjacency():
        neigh_list = []
        for neigh, attr_dict in neigh_dict.items():
            assert(len(attr_dict) == 0)
            neigh_list.append(neigh)
        adjacency_dict[n] = neigh_list
    return adjacency_dict



def relabel_edge_first(G, e):
    """Takes graph G and returns a relabelled graph 
    such that the vertices in edge e are labelled 0,1
    and all other vertices are labelled using 
    consecutive integers
    """
    assert(len(e) == 2)
    mapping = {e[0]: 0, e[1]: 1}
    i = 2
    for n in G.nodes():
        if n not in e:
            mapping[n] = i
            i += 1
    assert(len(mapping) == G.number_of_nodes())
    return nx.relabel_nodes(G, mapping)


def graph_cert(G):
    g = pynauty.Graph(number_of_vertices=G.number_of_nodes(), directed=nx.is_directed(G),
                adjacency_dict = get_adjacency_dict(G),
                vertex_coloring = [set([0,1]), set(range(2, G.number_of_nodes()))])
    cert = pynauty.certificate(g)
    return cert

def get_cert_e_tuples(e,G=None,p=None):
    """A helper function for multiprocessing
    """
    subgraph = relabel_edge_first(get_edge_subgraph(G, e, p), e)
    cert = graph_cert(subgraph)
    return e,cert


def get_edge_orbits_lightcones(G,p):
    """Takes graph G and number of QAOA steps p
    returns unique subgraphs that QAOA sees
    dict: {orbit_id : [list of edges in orbit]} 
    and maximum number of nodes in a lightcone subgraph
    """

    eorbits = defaultdict(list)
    # for each edge construct the light cone subgraph and compute certificate  
    if G.number_of_edges() > 1000:
        # accelerate with multiprocessing if computing for a large graph
        with Pool(psutil.cpu_count()) as pool:
            certs_e_tuples = pool.map(partial(get_cert_e_tuples, G=G,p=p), G.edges())
    else:
        certs_e_tuples = [get_cert_e_tuples(e,G=G,p=p) for e in G.edges()]

    for e,cert in certs_e_tuples:
        eorbits[cert].append(e)

    eorbits_integer_keys = {}
    for i, (cert, edges) in enumerate(eorbits.items()):
        eorbits_integer_keys[i] = edges

    assert(len(eorbits_integer_keys) == len(eorbits))
    return eorbits_integer_keys, None 
