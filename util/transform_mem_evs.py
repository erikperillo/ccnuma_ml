#!/usr/bin/env python3

import pandas as pd
import numpy as np

FILEPATH = "./../data/mem_evs.csv"

def r_sq(est_y, y):
    ss_tot = ((y - y.mean())**2).sum()
    #ss_reg = ((est_y - u_y)**2).sum()
    ss_res = ((y - est_y)**2).sum()

    return 1 - ss_res/ss_tot

def main():
    df = pd.read_csv(FILEPATH)

    df2 = pd.DataFrame()
    df2["name:suite"] = df["name:suite"]
    for i in range(8):
        df2["loc_mem_acc_%d" % i] = df["mem_acc_%d_to_%d" % (i, i)]
        df2["rem_mem_acc_%d" % i] = 0
        for j in range(8):
            if i != j:
                df2["rem_mem_acc_%d" % i] += df["mem_acc_%d_to_%d" % (j, i)]
    print(df2)

    df3 = pd.DataFrame()
    df3["name:suite"] = df2["name:suite"]
    df3["loc_mem_acc_sum"] = 0
    df3["loc_mem_acc_sqrtsumsq"] = 0
    df3["rem_mem_acc_sum"] = 0
    df3["rem_mem_acc_sqrtsumsq"] = 0
    for i in range(8):
        df3["loc_mem_acc_sum"] += df2["loc_mem_acc_%d" % i]
        df3["loc_mem_acc_sqrtsumsq"] += df2["loc_mem_acc_%d" % i]**2
        df3["rem_mem_acc_sum"] += df2["rem_mem_acc_%d" % i]
        df3["rem_mem_acc_sqrtsumsq"] += df2["rem_mem_acc_%d" % i]**2
    df3["rem_mem_acc_sqrtsumsq"] = np.sqrt(df3["rem_mem_acc_sqrtsumsq"])
    df3["loc_mem_acc_sqrtsumsq"] = np.sqrt(df3["loc_mem_acc_sqrtsumsq"])
    df2_loc = df2.drop(["name:suite"] + ["rem_mem_acc_%d"%i for i in range(8)],
        axis=1)
    df3["loc_mem_acc_max"] = df2_loc.max(axis=1)
    df2_rem = df2.drop(["name:suite"] + ["loc_mem_acc_%d"%i for i in range(8)],
        axis=1)
    df3["rem_mem_acc_max"] = df2_rem.max(axis=1)
    print(df3)

    df3.drop("name:suite", axis=1).to_csv("mem_evs_rebels_derived.csv",
        index=False)

    df4 = pd.DataFrame()
    df4["name:suite"] = df3["name:suite"]
    df4["loc_mem_acc_sqrtsumsq"] = df3["loc_mem_acc_sqrtsumsq"]
    df4.to_csv("lel.csv", index=False)

if __name__ == "__main__":
    main()
