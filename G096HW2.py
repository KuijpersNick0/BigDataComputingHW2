# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:11:23 2023

@author: Umut, Chel, Nick
"""

from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import time
import statistics 

def CountTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)  
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge[0]
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count

def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        # if isinstance(edge, str):
        #     u, v = edge.split(',')
        # else: 
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if int(v) > int(u):
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if int(w) > int(v) and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def MR_ApproxTCwithNodeColors(edges, C):
    # define hash equation variables in this scope to have the same vars for every calculations.
    p = 8191
    a = rand.randint(1, p-1)
    b = rand.randint(0, p-1)

    def hash_color(pair):
        def run_equation(vertex):
            return ((a*int(vertex) + b) % p) % int(C)

        u, v = pair
       
        hcu = run_equation(u)
        hcv = run_equation(v)

        # Check if endpoints have the same hash score
        if hcu == hcv:
            return (hcu, (u, v))
        else:
            return None

    t_count = edges.map(lambda x: hash_color(
        x)).filter(lambda x: x is not None).groupByKey().map(lambda x: (0, CountTriangles(x[1]))).reduceByKey(lambda x, y: x+y)

    t_final = C*C*t_count.collect()[0][1]

    return t_final

def splitPair(pair):
    u, v = map(int, pair.split(','))
    return (u, v)


def MR_ExactTC(edges, C):    
    # define hash equation variables in this scope to have the same vars for every calculations.
    p = 8191
    a = rand.randint(1, p-1)
    b = rand.randint(0, p-1)

    def create_pairs(pair, C):
        def run_equation(vertex):
            return ((a*int(vertex) + b) % p) % int(C)
        
        u, v = pair 
        # u, v = map(int, pair.split(',')) 

        hcu = run_equation(u)
        hcv = run_equation(v)
        
        pairs_dict = {}
        for i in range(C):
            key = tuple(sorted([hcu, hcv, i]))
            pairs_dict.setdefault(key, []).append((u,v))
        return pairs_dict

    pairs = edges.flatMap(lambda x: create_pairs(x, C).items())
    L_k = pairs.groupByKey().map(lambda x: (x[0], list(x[1])))
    t_k = L_k.map(lambda x: (x[0], CountTriangles2(x[0], x[1], a, b, p, C)))
    t_final = t_k.map(lambda x: x[1]).sum()
    
    return t_final


def main():

    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: python hw2.py <C> <R> <F> <file_name>"
    
    # SPARK SETUP
    conf = SparkConf().setAppName('G096HW2')
    conf.set("spark.locality.wait", "0s")
    sc = SparkContext(conf=conf).getOrCreate()
    #spark.sparkContext.setLogLevel("WARN")

    # INPUT READING
    # 1. Read number of colours
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)

    # 2. Read number of runs
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)

     # 3. Read number of runs
    F = sys.argv[3]
    assert F.isdigit(), "F must be an integer"
    F = int(F)

    # 4. Read input graph
    data_path = sys.argv[4]
    #assert os.path.isfile(data_path), "File or folder not found"

    rawData = sc.textFile(data_path)
    
    num_executors = sc.getConf().get("spark.executor.instances")
    
    # Transform the RDD of strings into an RDD of edges, partition and cache them.
    edges = rawData.map(lambda x: tuple(map(int, x.split(',')))).repartition(numPartitions=32).cache()

    # SETTING GLOBAL VARIABLES
    print("OUTPUT with parameters:  with " +str(num_executors) +" executors, " +str(C)
          + " colors, " +str(R) + " repetitions, flag " +str(F) + ", file " +
          str(os.path.basename(data_path)))
    print("Dataset = ", os.path.basename(data_path))
    print("Number of Edges = ", edges.count())
    print("Number of Colors = ", C)
    print("Number of Repetitions = ", R) 
    
    if (F == 0):
        print("Approximation algorithm with node coloring")
        
        t_estimates_partitions = []
        start_time = time.time()
        for i in range(1,R):            
            #create RDD here to circumvent lazy execution
            t_estimates_partitions.append(MR_ApproxTCwithNodeColors(edges, C))
            
        print("-Number of triangles (median over " +str(R) +" runs) =", statistics.median(t_estimates_partitions))
        
        print("Running time = ", ((time.time() - start_time)*1000)/R, " ms")

    
    else :
        print("Exact algorithm with node coloring")        
        # Count the triangles

        t_estimates_partitions = []
        start_time = time.time()
        
        for i in range(1,R):           
            #create RDD here to circumvent lazy execution
            t_estimates_partitions.append(MR_ExactTC(edges, C))
            
        print("- Number of triangles" , statistics.median(t_estimates_partitions))
        
        print("- Running time (average over " +str(R) + " runs) = ", ((time.time() - start_time)*1000)/R, " ms")

    sc.stop()


if __name__ == "__main__":
    main()