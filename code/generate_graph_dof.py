"""
This script is used to generate new graphs with given degree of freedom

"""
import numpy as np
import random
from pysat.solvers import Glucose3
from copy import copy, deepcopy
import random
from collections import defaultdict
import os

def get_hamiltonian_path(n,assignments):
    path = [None]*n
    for i in range(n):
        for j in range(n):
            if assignments[i*n+j] > 0: # True
                path[i] = j+1
    return path
     
def reduce_Hamiltonian_Path_to_SAT_and_solve(n,edges):
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
    status = g.solve()
    assignments = g.get_model()
    print(status)
    # print_SAT_solution(assignments)
    if status==1:
        path = get_hamiltonian_path(n,assignments)
        print(path)

def generate_H(N,elst1,elst2,n1,n2,flag):
    elst=[]
    # with open('data.txt','w') as f:
    # f.write(str)  
    arr = np.arange(n1+n2)
    arr1= np.arange(n1)
    arr2= np.arange(n2)+n1
    if n1+n2>24:
        len_elst1=min(len(elst1),8)
        len_elst2=min(len(elst2),10)
    else:
        len_elst1=len(elst1)
        len_elst2=len(elst2)

    for i in range(len_elst1):
        for j in range(len_elst2):
            for k in range(N):
                edges1=elst1[i].copy()
                edges2=np.array(elst2[j])+n1
                edges3=edges2.tolist()
                # print(edges3)
                edges1.extend(edges3)
                if flag==1:
                    np.random.shuffle(arr)
                    for k in range(n1+n2):
                        a=arr[k]+1
                        b=arr[(k+1)%(n1+n2)]+1
                        edges1.append([a,b])
                else:
                    np.random.shuffle(arr1)
                    np.random.shuffle(arr2)
                    n=n1
                    for k in range(n-1):
                        a=arr1[k]+1
                        b=arr1[k+1]+1
                        edges1.append([a,b])
                    n=n2
                    for k in range(n-1):
                        a=arr2[k]+1
                        b=arr2[k+1]+1
                        edges1.append([a,b])
                    
                    edges1.append([arr1[0]+1,arr2[0]+1])
                    edges1.append([arr1[0]+1,arr2[1]+1])
            
                # print(edges1)
                elst.append(edges1)
    #         return 
    print(len(elst))
    return elst
                    # print(edges1)
        #             break
        #         break
        #     break
        # break




def read(n):
    # path='node/'+str(n)+'_H.txt'
    path='node/'+str(n)+'_NH.txt'
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
            # edges.append([u,v])

        else:
            if len(edges)!=0:
                elst.append(edges)
            # print(edges)
            edges=[]
        # print(data)
    print(len(elst))
    return elst
    # for i in range(len(elst)):
    #     print(i)
    #     reduce_Hamiltonian_Path_to_SAT_and_solve(n,elst[i])
    #     print()
        # break
def read_graph(path):
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
            # edges.append([u,v])
        else:
            if len(edges)!=0:
                elst.append(edges)
            # print(edges)
            edges=[]
        # print(data)
    print(len(elst))
    return elst

    
def test(n1,n2,N):
    print()
    print("----------")
    print(n1,n2,N)
    # n1,n2=8,8
    elst1,elst2=read(n1),read(n2)
    # N=30
    NH=generate_H(N,elst1,elst2,n1,n2,0)
    write_H(n1+n2,NH,'_NH.txt')

    YH=generate_H(N,elst1,elst2,n1,n2,1)
    write_H(n1+n2,YH,'_H.txt')
    # NH1=generate_H(elst1,elst2,n1,n2,0)

# test(8,6,30)
# test(12,10,1) # 22
# test(12,12,1) # 24
# test(12,14,1)  # 26
# test(14,14,1) # 28
# test(14,16,1) # 30
# test(16,16,1) # 32
# test(16,18,1) # 34
def add_dof(graph, dof):
    graph = deepcopy(graph)
    nodes_set = defaultdict(set)    # used to store all the adjacent nodes info
    edge_set = set()                # used to store all the edges
    for [u, v] in graph:
        one_edge = "{}_{}".format(u, v) if u < v else "{}_{}".format(v, u)
        edge_set.add(one_edge)
        nodes_set[v].add(u)
        nodes_set[u].add(v)
    new_edges = []
    neighbors = deepcopy(nodes_set)
    for node in nodes_set.keys():
        node_neighbors = nodes_set[node]
        if len(node_neighbors) < dof:
            # print("--"*10)
            # print("add edge for node: ", node)
            # random pick nodes in the list
            # find the available nodes that can add edge to it
            # needs to be: not connected to node, its dof is less than dof
            local_neighbors = deepcopy(neighbors)
            if node in local_neighbors:
                del local_neighbors[node]
            neighbor_list = list(local_neighbors.keys())
            # print("neighbor_list  = ", neighbor_list)
            # print("node_neighbors = ", node_neighbors)
            if len(neighbor_list) == 0:
                break
            random.shuffle(neighbor_list)
            for neighbor in neighbor_list:
                if neighbor in node_neighbors: # if it is already linked to node
                    continue
                if len(neighbors[neighbor]) >= dof: # this case should not happen
                    continue
                new_edge = [node, neighbor]
                new_edges.append(new_edge)
                one_edge = "{}_{}".format(node, neighbor) if node < neighbor else "{}_{}".format(neighbor, node)
                edge_set.add(one_edge)
                nodes_set[node].add(neighbor)
                nodes_set[neighbor].add(node)
                neighbors[node].add(neighbor)
                neighbors[neighbor].add(node)
                # print("node {} added with {}".format(node, neighbor))
                if len(nodes_set[node]) >= dof:
                    break
    # print("=="*10)
    # print("new_edges    = ", new_edges)
    # print("before graph = ", graph)
    graph.extend(new_edges)
    # print("after  graph = ", graph)
    nodes_count = len(nodes_set)
    # print("--"*10)
    # add hc to graph
    nodes_list = list(nodes_set.keys())
    random.shuffle(nodes_list)
    round_nodes_list = nodes_list+[nodes_list[0]]
    # print(round_nodes_list)
    for idx, node in enumerate(round_nodes_list):
        next_node_id = idx + 1
        if next_node_id == len(round_nodes_list):
            break
        
        next_node = round_nodes_list[next_node_id]
        one_edge = "{}_{}".format(node, next_node) if node < next_node else "{}_{}".format(next_node, node)
        # print(idx, one_edge)
        if one_edge in edge_set:
            continue
        edge_set.add(one_edge)
        nodes_set[node].add(next_node)
        nodes_set[next_node].add(node)
        graph.append([node, next_node])
    #     print("added {} and {}".format(node, next_node))
    # print(graph)
    # num_dof_max = max([len(ele) for ele in nodes_set.values()])
    # print("for H graph, the max dof is : ", num_dof_max)
    # remove edges for one node, so that the graph is not hc 
    # print("=="*10)
    random.shuffle(nodes_list)
    sp_node = nodes_list[0]
    graph_nh = [ele for ele in graph if sp_node not in ele ]
    graph_with_s = [ele for ele in graph if sp_node  in ele ]
    for ele in nodes_set[sp_node]: # remove it from all its neighbors's set
        nodes_set[ele].remove(sp_node)
    del nodes_set[sp_node]

    one_sp = graph_with_s[0]
    second_node_pool = [i+1 for i in range(nodes_count) if i != sp_node]
    random.shuffle(second_node_pool)
    second_node = second_node_pool[0]

    one_more_edge = [sp_node, second_node]

    graph_nh.append(one_more_edge)
    nodes_set[one_more_edge[0]].add(one_more_edge[1])
    nodes_set[one_more_edge[1]].add(one_more_edge[0])
    # print(graph_nh)
    # num_dof_max = max([len(ele) for ele in nodes_set.values()])
    # print("for NH graph, the max dof is : ", num_dof_max)
    return graph, graph_nh, nodes_count


def generate_hc_nhc(graphs, output_folder, dofs, threshold=5000):
    """
    add a hc to the graph
    """
    graphs = deepcopy(graphs)
    graphs_hc, graphs_nhc = [], []
    nodes_count0 = 0
    # print(dofs, len(graphs))
    for dof in dofs:
        for idx, graph in enumerate(graphs):
            if idx > threshold:
                break
            graph_hc, graph_nhc, nodes_count = add_dof(graph, dof)
            # print("----")
            graphs_hc.append(graph_hc)
            graphs_nhc.append(graph_nhc)
            if idx != 0:
                # print(" nodes_count = {},  nodes_count0 = {}".format(nodes_count, nodes_count0))
                assert nodes_count == nodes_count0
            nodes_count0 = nodes_count
        
    nhc_output_path = os.path.join(output_folder, "{}_NH.txt".format(nodes_count))
    hc_output_path  = os.path.join(output_folder, "{}_H.txt".format(nodes_count))
    save_H(graphs_hc,  hc_output_path)
    save_H(graphs_nhc, nhc_output_path)

    

def save_H(graph, file_name):
    with open(file_name,'w') as f:
        for i in range(len(graph)):
            for (a,b) in graph[i]:
                s=' {} {} '.format(a,b)+'\n'
                f.write(s)
            f.write('\n')
        f.close()
    return 0



def write_H(n,elst,h):
    path='node3/'+str(n)+h

    with open(path,'w') as f:
        for i in range(len(elst)):
            for (a,b) in elst[i]:
                s=' {} {} '.format(a,b)+'\n'
                f.write(s)
            f.write('\n')
                # print(' {} {} '.format(a,b))
            # print()
        # break


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':


    graphs_path = 'node/20_NH.txt'   # we use the graphs in this file
    graphs_path = 'node/18_NH.txt'   # we use the graphs in this file
    base_graphs = read_graph(graphs_path)
    # print(base_graphs[0])
    # add_dof(base_graphs[0], 4)

    dof=4
    dofs = [ele for ele in range(4, 11)]
    # for dof in dofs:
    output_folder = 'nodes_{}'.format(7)
    check_mkdir(output_folder)
    generate_hc_nhc(base_graphs[-6000:], output_folder, dofs)

