#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools

FILEPATH = "./data/data_full.csv"
NUMA_NODES = list(range(8))
PREFIX = "G_"
NUMA_DISTS = [
    [10,  16,  16,  22,  16,  22,  16,  22],
    [16,  10,  22,  16,  22,  16,  22,  16],
    [16,  22,  10,  16,  16,  22,  16,  22],
    [22,  16,  16,  10,  22,  16,  22,  16],
    [16,  22,  16,  22,  10,  16,  16,  22],
    [22,  16,  22,  16,  16,  10,  22,  16],
    [16,  22,  16,  22,  16,  22,  10,  16],
    [22,  16,  22,  16,  22,  16,  16,  10],
]

def pfx(ev_str):
    return PREFIX + ev_str

def mem_acc_str(from_node, to_node):
    return "mem_acc_%d_to_%d" % (from_node, to_node)

def get_mem_acc(df, node_from, node_to, use_weight=True):
    weight = (NUMA_DISTS[node_from][node_to]/10) if use_weight else 1
    return weight*df[pfx(mem_acc_str(node_from, node_to))]

def contention(df):
    mem_acc = mem_acc_sum(df, False)
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for f, t in itertools.product(NUMA_NODES, NUMA_NODES):
        values += get_mem_acc(df, f, t, True)/mem_acc
    return values

def mem_acc_sum(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for f, t in itertools.product(NUMA_NODES, NUMA_NODES):
        values += get_mem_acc(df, f, t, use_weight)
    return values

def mem_acc_mean(df, use_weight=False):
    return mem_acc_sum(df, use_weight)/len(NUMA_NODES)**2

def mem_acc_var(df, use_weight=False):
    u = mem_acc_mean(df, use_weight)
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for f, t in itertools.product(NUMA_NODES, NUMA_NODES):
        values += (get_mem_acc(df, f, t, use_weight) - u)**2
    return values/len(NUMA_NODES)**2

def mem_acc_std(df, use_weight=False):
    return np.sqrt(mem_acc_var(df, use_weight))


def mem_acc_to(df, node, use_weight=False):
    return loc_mem_acc_to(df, node, use_weight) +\
        rem_mem_acc_to(df, node, use_weight)

def mem_acc_to_max(df, use_weight=False):
    maxx = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        acc_to = mem_acc_to(df, n, use_weight)
        maxx = np.maximum(maxx, acc_to)
    return maxx

def mem_acc_to_sum(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += mem_acc_to(df, n, use_weight)
    return values

def mem_acc_to_mean(df, use_weight=False):
    return mem_acc_to_sum(df, use_weight)/len(NUMA_NODES)

def mem_acc_to_var(df, use_weight=False):
    u = mem_acc_to_mean(df, use_weight)
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += (mem_acc_to(df, n, use_weight) - u)**2
    return values/len(NUMA_NODES)

def mem_acc_to_std(df, use_weight=False):
    return np.sqrt(mem_acc_var(df, use_weight))

def mem_acc_to_sqrtsumsq(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += mem_acc_to(df, n, use_weight)**2
    return np.sqrt(values)


def loc_mem_acc_to(df, node, use_weight=False):
    return get_mem_acc(df, node, node, use_weight)

def loc_mem_acc_to_max(df, use_weight=False):
    maxx = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        acc_to = loc_mem_acc_to(df, n, use_weight)
        maxx = np.maximum(maxx, acc_to)
    return maxx

def loc_mem_acc_to_sum(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += loc_mem_acc_to(df, n, use_weight)
    return values

def loc_mem_acc_to_mean(df, use_weight=False):
    return loc_mem_acc_to_sum(df, use_weight)/len(NUMA_NODES)

def loc_mem_acc_to_var(df, use_weight=False):
    u = loc_mem_acc_to_mean(df, use_weight)
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += (loc_mem_acc_to(df, n, use_weight) - u)**2
    return values/len(NUMA_NODES)

def loc_mem_acc_to_std(df, use_weight=False):
    return np.sqrt(loc_mem_acc_to_var(df, use_weight))

def loc_mem_acc_to_sqrtsumsq(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += loc_mem_acc_to(df, n, use_weight)**2
    return np.sqrt(values)


def rem_mem_acc_to(df, node, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        if n != node:
            values += get_mem_acc(df, n, node, use_weight)
    return values

def rem_mem_acc_to_max(df, use_weight=False):
    maxx = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        acc_to = rem_mem_acc_to(df, n, use_weight)
        maxx = np.maximum(maxx, acc_to)
    return maxx

def rem_mem_acc_to_sum(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += rem_mem_acc_to(df, n, use_weight)
    return values

def rem_mem_acc_to_mean(df, use_weight=False):
    return rem_mem_acc_to_sum(df, use_weight)/len(NUMA_NODES)

def rem_mem_acc_to_var(df, use_weight=False):
    u = rem_mem_acc_to_mean(df, use_weight)
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += (rem_mem_acc_to(df, n, use_weight) - u)**2
    return values/len(NUMA_NODES)

def rem_mem_acc_to_std(df, use_weight=False):
    return np.sqrt(rem_mem_acc_to_var(df, use_weight))

def rem_mem_acc_to_sqrtsumsq(df, use_weight=False):
    values = np.array(df.shape[0]*[0], dtype=np.float64)
    for n in NUMA_NODES:
        values += rem_mem_acc_to(df, n, use_weight)**2
    return np.sqrt(values)


def show(df, cols):
    print(df[["name:suite"] + [pfx(c) for c in cols]])

def main():
    df = pd.read_csv(FILEPATH)

    use_w = True

    df[pfx("mem_acc_sum")] = mem_acc_sum(df, use_w)
    df[pfx("mem_acc_std")] = mem_acc_std(df, use_w)

    df[pfx("mem_acc_to_max")] = mem_acc_to_max(df, use_w)
    df[pfx("mem_acc_to_std")] = mem_acc_to_std(df, use_w)
    df[pfx("mem_acc_to_sqrtsumsq")] = mem_acc_to_sqrtsumsq(df, use_w)

    #df[pfx("loc_mem_acc_to_max")] = loc_mem_acc_to_max(df, use_w)
    #df[pfx("loc_mem_acc_to_std")] = loc_mem_acc_to_std(df, use_w)
    #df[pfx("loc_mem_acc_to_sqrtsumsq")] = loc_mem_acc_to_sqrtsumsq(df, use_w)

    #df[pfx("rem_mem_acc_to_max")] = rem_mem_acc_to_max(df, use_w)
    #df[pfx("rem_mem_acc_to_std")] = rem_mem_acc_to_std(df, use_w)
    #df[pfx("rem_mem_acc_to_sqrtsumsq")] = rem_mem_acc_to_sqrtsumsq(df, use_w)

    df = df.drop(\
        ["G_mem_acc_%d_to_%d" % (i, j) for i in range(8) for j in range(8)],
        axis=1)
    show(df,["mem_acc_sum"])

    print(df.columns)

    df.to_csv("data_derived.csv", index=False)

if __name__ == "__main__":
    main()
