#!/usr/bin/env python3

import os

CPUS = {i: "CPU%d" % (i*8) for i in range(8)}
UMASKS = {i: hex(2**i) for i in range(8)}
EVENT = "amd_nb/event=0x1e0"

def mean(lst):
    return sum(lst)/max(len(lst), 1)

def get_ev(lines, n_node_from, n_node_to, col=1):
    cpu_str = CPUS[n_node_from]
    event = "%s,umask=%s/" % (EVENT, UMASKS[n_node_to])
    lines = [l for l in lines if cpu_str in l and event in l]
    if not lines:
        raise Exception("error in get_ev")
    return int(lines[0].split(",")[col])

def metric_str(n_node_from, n_node_to):
    return "mem_acc_%d_to_%d" % (n_node_from, n_node_to)

def get_ev_line(dir_path):
    with open(os.path.join(dir_path, "results", "name")) as f:
        name = f.read().strip().lower()
    with open(os.path.join(dir_path, "results", "suite")) as f:
        suite = f.read().strip().lower()

    evs = {metric_str(i, j): [] for i in range(8) for j in range(8)}

    for n in range(1, 17):
        with open(os.path.join(dir_path, "perf-stat-out_%d" % n)) as f:
            lines = [l.strip() for l in f]

        for i in range(8):
            for j in range(8):
                evs[metric_str(i, j)].append(get_ev(lines, i, j))

    evs_lst = [mean(evs[metric_str(i, j)]) for i in range(8) for j in range(8)]
    return name + ":" + suite + "," + ",".join(map(str, evs_lst))

def main():
    header = "name:suite," + ",".join("mem_acc_%d_to_%d" % (i, j) for i in \
        range(8) for j in range(8))
    print(header)

    for s in range(25):
        dir_path = "./baut_run_20-02-2016_16-19-52/state_%d" % s
        print(get_ev_line(dir_path))

if __name__ == "__main__":
    main()
