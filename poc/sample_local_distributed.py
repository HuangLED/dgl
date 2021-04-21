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

from dgl.backend import *
from dgl.nn import *

def arange(start, stop, dtype=int64):
    return copy_to(_arange(start, stop, dtype), _default_context)

def start_server(rank, tmpdir, disable_shared_mem, graph_name):
    g = DistGraphServer(rank, "local_ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=disable_shared_mem)
    g.start()

def start_sample_client(rank, tmpdir, disable_shared_mem):
    gpb = None
    if disable_shared_mem:
        _, _, _, gpb, _, _, _ = load_partition(tmpdir / 'test_sampling.json', rank)
    dgl.distributed.initialize("local_ip_config.txt")
    dist_graph = DistGraph("test_sampling", gpb=gpb)

    # Phase #1.
    dist_graph = DistGraph("cora", gpb=gpb)
    try:
        sampled_graph = sample_neighbors(dist_graph, [0, 2000], 3)
    except Exception as e:
        print(e)
        sampled_graph = None
    print("Phase 1:")
    print(sampled_graph)

    print("\nPhase 2:")
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
    return sampled_graph

def check_rpc_sampling(tmpdir, num_server):
    g = CitationGraphDataset("cora")[0]
    g.readonly()
    print(g.idtype)
    partition_graph(g, 'test_sampling', 2, tmpdir, num_hops=1, part_method='metis', reshuffle=False)

    pserver_list = []
    ctx = mp.get_context('spawn')
    for i in range(num_server):
        p = ctx.Process(target=start_server, args=(i, tmpdir, num_server > 1, 'test_sampling'))
        p.start()
        time.sleep(1)
        pserver_list.append(p)

    time.sleep(3)
    sampled_graph = start_sample_client(0, tmpdir, num_server > 1)
    print("Done sampling")
    for p in pserver_list:
        p.join()

    src, dst = sampled_graph.edges()
    assert sampled_graph.number_of_nodes() == g.number_of_nodes()
    eids = g.edge_ids(src, dst)

if __name__ == "__main__":
    num_server = 2
    ip_config = open("local_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    os.environ['DGL_DIST_MODE'] = 'distributed'
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
      check_rpc_sampling(Path(tmpdirname), 2)