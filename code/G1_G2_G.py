
import numpy as np
import random
from pysat.solvers import Glucose3

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
test(12,10,1) # 22
test(12,12,1) # 24
test(12,14,1)  # 26
test(14,14,1) # 28
test(14,16,1) # 30
test(16,16,1) # 32
test(16,18,1) # 34

