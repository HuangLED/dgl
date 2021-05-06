import dgl
import unittest
import os
from dgl.data import CitationGraphDataset
from dgl.distributed import sample_neighbors, find_edges
from dgl.distributed import partition_graph, load_partition, load_partition_book
from dgl.distributed import sampling_service_pb2, sampling_service_pb2_grpc
import sys
import multiprocessing as mp
import numpy as np
import time
from utils import get_local_usable_addr
from pathlib import Path
import pytest
from scipy import sparse as spsp
from dgl.distributed import DistGraphServer, DistGraph
from dgl.distributed.dist_context import get_sampler_pool
from dgl.backend import *
from dgl.nn import *
import traceback

from dgl.distributed.rpc_client import connect_to_server, shutdown_servers
from dgl.distributed.role import init_role
from dgl.distributed.constants import MAX_QUEUE_SIZE
from dgl.distributed.kvstore import init_kvstore, close_kvstore
from dgl.distributed.dist_context import _init_rpc

FOLDER = "/home/centos/tmp"

def start_server(rank, tmpdir, disable_shared_mem, graph_name):
    g = DistGraphServer(rank, "local_ip_config.txt", 1, 1,
                        tmpdir / (graph_name + '.json'), disable_shared_mem=True)
    g.start()

def check_rpc_sampling(tmpdir, num_server):
    gpb = None

    # create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1,
                                                    initializer=_init_rpc,
                                                    initargs=("local_ip_config.txt", 1, MAX_QUEUE_SIZE, 'socket', 'default', 2)))
    sampling_service_pb2_grpc.add_SampleServiceServicer_to_server(
            SampleServicer(gpb), server)

    print('Starting server on port 50027.')
    server.add_insecure_port('[::]:50027')
    server.start()

    # Now, use test_client.py.
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

    #for p in pserver_list:
    #    p.join()

import grpc
from concurrent import futures
import time
from dgl.distributed import sampling_service_pb2
from dgl.distributed import sampling_service_pb2_grpc
import os
import threading

# create a class to define the server functions, derived from
# calculator_pb2_grpc.CalculatorServicer
class SampleServicer(sampling_service_pb2_grpc.SampleService):
    def __init__(self, gpb):
        # 拉起图server。纯为测试用。正确做法显然不应该在这里启动Graph Server
        pserver_list = []
        ctx = mp.get_context('spawn')
        for i in range(num_server):
            p = ctx.Process(target=start_server, args=(i, Path(FOLDER), num_server > 1, 'test_sampling'))
            p.start()
            time.sleep(1)
            pserver_list.append(p)
        time.sleep(10)

        self.gpb = gpb
        print('Sample Service constructor PID: ', os.getpid())
        print("current thread:", threading.current_thread().name)

        # Doing sampling here is trivial.
        """
        dgl.distributed.initialize("local_ip_config.txt", num_workers=10)
        _, _, _, gpb_local, _, _, _ = load_partition(FOLDER+'/test_sampling.json', 0)
        self.dist_graph = DistGraph("test_sampling", gpb=gpb_local)
        sampled_graph = sample_neighbors(self.dist_graph, range(1500,2000), 3)
        #sampled_graph = sample_neighbors(self.dist_graph, [1000], 3)
        print("Sampling done inside GRPC service constructor", os.getpid(), sampled_graph)
        """

    # TODO: 需要改成用异步的方式采样查询。
    def SampleNeighbor(self, request, context):
        print('Processing one request with pid: ', os.getpid())
        print("current thread:", threading.current_thread().name)
        
        response = sampling_service_pb2.Result()
        if request.count > 0:
            # TODO: need to save partition book as a context/session for each thread. Has to be thread safe.
            _, _, _, gpb_local, _, _, _ = load_partition(FOLDER+'/test_sampling.json', 0)
            dist_graph = DistGraph("test_sampling", gpb=gpb_local)
            
            # TODO: pass in batch of nodes.
            sampled_graph = sample_neighbors(dist_graph, [request.id], request.count)
            print("Sampling done inside GRPC request processing: ", sampled_graph)
            # TODO: inspect into Graph's data structure.
            response.neighbor = sampled_graph.num_edges()

        return response

if __name__ == "__main__":
    g = CitationGraphDataset("cora")[0]
    g.readonly()
    partition_graph(g, 'test_sampling', 2, Path(FOLDER), num_hops=3, part_method='metis', reshuffle=False)
    
    num_server = 2
    ip_config = open("local_ip_config.txt", "w")
    for _ in range(num_server):
        ip_config.write('{}\n'.format(get_local_usable_addr()))
    ip_config.close()

    os.environ['DGL_DIST_MODE'] = 'distributed'
    check_rpc_sampling(Path(FOLDER), 2)
