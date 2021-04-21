import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed import sample_neighbors, find_edges
from dgl.distributed import partition_graph, load_partition, load_partition_book
import sys
import multiprocessing as mp
import numpy as np
import time
from utils import get_local_usable_addr
from pathlib import Path
import pytest
from scipy import sparse as spsp
from dgl.distributed import DistGraphServer, DistGraph

# TODO: On slave node, change this to be 1.
SERVER_ID = 0  # 0 or 1. 

def start_server(rank, tmpdir, disable_shared_mem, graph_name):
    g = DistGraphServer(rank, "ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=disable_shared_mem)
    g.start()

def start_sample_client(rank, tmpdir, disable_shared_mem):
    gpb = None
    #if disable_shared_mem:
    _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'cora.json', rank)
    dgl.distributed.initialize("ip_config.txt")

    # Phase #1.
    dist_graph = DistGraph("cora", gpb=gpb)
    try:
        sampled_graph = sample_neighbors(dist_graph, [0, 2000], 3)
    except Exception as e:
        print(e)
        sampled_graph = None
    print("Phase 1:")
    print(sampled_graph)

    # Phase #2.
    print("Phase 2:")
    batch = []
    r = []
    print("# of nodes: ", dist_graph.number_of_nodes())
    for i in range(dist_graph.number_of_nodes()):
        batch.append(i)
        if len(batch) == 50:
            sampled_graph = sample_neighbors(dist_graph, batch, 5)
            r.append(sampled_graph.num_edges())
            batch.clear()
    print('DONE: batch count:', len(r))
    print('Number of edges in each batch:', r)

    dgl.distributed.exit_client()

def check_rpc_sampling(tmpdir, num_server):
    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        if i != SERVER_ID: continue
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'cora'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    print("Start sampling")
    sampled_graph = start_sample_client(0, tmpdir, num_server > 1)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    #src, dst = sampled_graph.edges()
    #assert sampled_graph.number_of_nodes() == g.number_of_nodes()

    #assert np.all(F.asnumpy(g.has_edges_between(src, dst)))
    #eids = g.edge_ids(src, dst)
    #print(eids)    
    #assert np.array_equal(
    #    F.asnumpy(sampled_graph.edata[dgl.EID]), F.asnumpy(eids))

if __name__ == "__main__":
    g = CitationGraphDataset("cora")[0]
    g.readonly()
    print(g.idtype)
    partition_graph(g, 'cora', 2, "./data/", num_hops=1, part_method='metis', reshuffle=False)

    import tempfile

    # TODO: Update folder name.
    tmpdirname = "/home/centos/github/dgl/poc/data"
    
    os.environ['DGL_DIST_MODE'] = 'distributed'
    check_rpc_sampling(Path(tmpdirname), 2)
