#!/usr/bin/env python3

import pandas as pd
import numpy as np

def hist(lst):
    return {k: lst.count(k) for k in set(lst)}

def get_clases(il_speedups, an_speedups):
    def clase(speedups):
        print(speedups)
        il_s, an_s = speedups
        if il_s < 1 and an_s < 1:
            return "node_local"
        return "interleave" if (il_s > an_s) else "autonuma"
    return list(map(clase, zip(il_speedups, an_speedups)))

def main():
    old = pd.read_csv("./data_clf.csv")
    df = pd.read_csv("./def.csv")
    an = pd.read_csv("./an.csv")
    il = pd.read_csv("./il.csv")

    #print(old["name:suite"] == df["name:suite"])
    #print(old["name:suite"] == an["name:suite"])
    #print(old["name:suite"] == il["name:suite"])
    print(df["time_elapsed_s"]/old["time_elapsed_s"])

    df2 = pd.DataFrame(old).drop(["time_elapsed_s", "speedup_il", "speedup_an",
        "best_policy"], axis=1)
    df2["time_elapsed_s"] = df["time_elapsed_s"]
    df2["speedup_il"] = df["time_elapsed_s"]/il["time_elapsed_s"]
    df2["speedup_an"] = df["time_elapsed_s"]/an["time_elapsed_s"]
    clases = get_clases(df2["speedup_il"].values, df2["speedup_an"].values)
    df2["best_policy"] = clases
    print(df2[:4])
    print(clases)
    print(hist(clases))
    df2.to_csv("new_data.csv", index=False)

    #print(old["speedup_il"]/df2["speedup_il"])
    #print(df2["speedup_an"])
    #print(old["speedup_an"]/df2["speedup_an"])
    #print(df2["speedup_an"])

if __name__ == "__main__":
    main()
