
import numpy as np
from pysat.solvers import Glucose3
from collections import defaultdict
import time
import os
import torch
import argparse

def get_hamiltonian_path(n,assignments):
    path = [None]*n
    for i in range(n):
        for j in range(n):
            if assignments[i*n+j] > 0: # True
                path[i] = j+1
    return path
     
def reduce_Hamiltonian_Path_to_SAT_and_solve(n,edges, graph_type=1):
    # print(n)
    def index(i, j):
        return n*i + j + 1
         
    m = len(edges)
    # n_clauses = 2*n + (2*n*n-n-m)*(n-1)
    n_vars = n*n
    clauses = []
    
    # 1. Each vertex $j$ must appear at least once.
    for j in range(n):
        clause = []
        for i in range(n):
            clause.append(index(i,j))
        clauses.append(clause)

    # 3.  Each position $i$ must be occupied by at least one vertex.
    for i in range(n):
        clause = []
        for j in range(n):
            clause.append(index(i,j)) 
        clauses.append(clause)

    # 2.  Each vertex $j$ must appear at most once.
    for j in range(n):
        for i in range(n):
            for k in range(i+1, n):
                clauses.append((-index(i,j), -index(k,j)))

    # 4.  Each position $i$ must be occupied by at most one vertex.
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                clauses.append((-index(i,j), -index(i,k)))

    # 5. Nonadjacent vertex $i$ and $j$ cannot be adjacent in the cycle.
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if not [i+1, j+1] in edges:
                    clauses.append((-index(k%n,i), -index((k+1)%n,j)))


    # print_clauses(clauses)
    g = Glucose3()
    for c in clauses:
        # print(c)
        g.add_clause(c)
    num_clauses = g.nof_clauses()
    status = g.solve()
    assignments = g.get_model()
    # print(status, graph_type)
    assert int(status) == graph_type, "status = {}, graph_type = {}".format(status, graph_type)
    # print_SAT_solution(assignments)
    if status==1:
        path = get_hamiltonian_path(n,assignments)
        # print(path)
    return num_clauses

def read(n):
    path='node/'+str(n)+'_H.txt'
    # path='node/'+str(n)+'_NH.txt'
    elst=[]
    with open(path, "r") as f:
        data = f.read().split('\n')
    edges=[]
    for link in data:
        link=link.strip().split()
        # print(link)
        if len(link)!=0:
            v,u=int(link[0]),int(link[1])
            edges.append([v,u])
            edges.append([u,v])

        else:
            if len(edges)!=0:
                elst.append(edges)
            # print(edges)
            edges=[]
        # print(data)
    # print(len(elst))
    for i in range(len(elst)):
        print(i)
        reduce_Hamiltonian_Path_to_SAT_and_solve(n,elst[i])
        print()
        # break

def process_one_graph(filename):
    """
    Args: 
        filename (str): path to the graph, should be one txt file
    Returns:
        dt (float): time used to solve this graph
        num_node (int): the number of nodes
        num_edge (int): the number of the edges
        num_dofd (int): the (max) degree of freedom of this graph

    """
    print(filename)
    elst=[]
    with open(filename, "r") as f:
        data = f.read().split('\n')
    edges=[]
    graph_type = 0 if filename[-6] == 'N' else 1
    nodes_set = defaultdict(set)    # used to store all the adjacent nodes info
    edge_set = set()                # used to store all the edges
    graphs = []
    for link in data:
        link=link.strip().split()
        # print(link)
        if len(link)!=0:
            v,u=int(link[0]),int(link[1])
            edges.append([v,u])
            edges.append([u,v])
            one_edge = "{}_{}".format(u, v) if u < v else "{}_{}".format(v, u)
            nodes_set[v].add(u)
            nodes_set[u].add(v)
            edge_set.add(one_edge)
        else:
            if len(edges) == 0:
                continue
            # if len(edges)!=0:
            elst.append(edges)
                # num_edge += 1
            # print(edges)
            
            num_node = len(nodes_set.keys())
            num_edge = len(edge_set)
            num_dof_max = max([len(ele) for ele in nodes_set.values()])
            num_dof_ave = sum([len(ele) for ele in nodes_set.values()])/len(nodes_set.keys())
            # if graph_type == 1:
            #     print("num_dof_max = ", num_dof_max)
            #     print("num_dof_ave = ", num_dof_ave)
            #     print(nodes_set)

            gt_num_node = int(filename.split("/")[-1].split("_")[0])
            assert num_node == gt_num_node, "num_node = {}, gt_num_node = {}".format(num_node, gt_num_node)
            one_graph = [num_node, num_edge, num_dof_max, num_dof_ave, graph_type, filename, edges]
            graphs.append(one_graph)

            edges=[]
            nodes_set = defaultdict(set)    # used to store all the adjacent nodes info
            edge_set = set()  
            # break
    
    for i in range(len(elst)):
        start_t = time.time()
        num_clauses = reduce_Hamiltonian_Path_to_SAT_and_solve(num_node,elst[i], graph_type=graph_type)
        end_t = time.time()
        dt = end_t - start_t
        graphs[i].append(num_clauses)
        graphs[i].append(dt)
    
    # for ele in graphs:
    #     ele.append(filename)
    # graphs = dict(( +str(i), ele) for i, ele in enumerate(graphs))
    # print(graphs)
    return graphs

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--node_size', default=-1, type=int)
    parser.add_argument('--root_path', default="node", type=str)
    parser.add_argument('--output_folder', default="outputs20220531", type=str)
    parser.add_argument('--graph_type', type=str)
    args = parser.parse_args()
    # read(4)
    # exp_path = 'node/6_H.txt'
    # process_one_graph(exp_path)
    output_folder = args.output_folder
    check_mkdir(output_folder)
    res = []
    root_path = args.root_path
    files = os.listdir(root_path)
    for filename in files:
        if not filename.endswith("H.txt"):
            continue
        if args.node_size != -1:
            node_size = int(filename.split("_")[0])
            if node_size != args.node_size:
                continue
        if args.graph_type is not None:
            assert args.graph_type in ["H", "NH"]
            if args.graph_type == "H":
                if filename.endswith("NH.txt"):
                    continue
            else:
                if filename.endswith("_H.txt"):
                    continue
        graphfilename = os.path.join(root_path, filename)
        print("Begin with file {}!".format(graphfilename))
        one_res = process_one_graph(graphfilename)
        res.extend(one_res)
        print("Done with file {}!".format(graphfilename))
        torch.save(one_res, output_folder+"/{}.pt".format(filename[:-4]))
    if args.node_size == -1:
        torch.save(res, output_folder+"/original_graphs20220531.pth")

    
    