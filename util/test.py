#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def r_sq(est_y, y):
    ss_tot = ((y - y.mean())**2).sum()
    ss_res = ((y - est_y)**2).sum()
    return 1 - ss_res/ss_tot

def f(x, coeffs):
    a, b, c = coeffs
    return a*x**2 + b*x + c

def main():
    df = pd.read_csv("./data/data.csv")

    #print(df["name:suite"][3])
    #df = df.drop([3], axis=0)

    labels = df["name:suite"].values
    frac_stalled_cycles_backend = (df["T_stalled_cycles_backend"]\
        /df["T_cycles"]).values
    mem_acc = df["G_mem_acc_sum"].values
    mem_acc_to_std = df["G_mem_acc_to_std"].values
    mem_acc_to_sqrtsumsq = df["G_mem_acc_to_sqrtsumsq"].values

    x = (mem_acc_to_sqrtsumsq/mem_acc)*frac_stalled_cycles_backend
    y = df["speedup_an"]

    npb_ids = [i for i, v in enumerate(labels) if v.split(":")[1] == "npb"]
    easy_ids = [i for i, v in enumerate(labels) if v.split(":")[1] == "easy"]
    splash2x_ids = [i for i, v in enumerate(labels)\
        if v.split(":")[1] == "splash2x"]

    fit = np.polyfit(x, y, 2)
    print("fit:", fit)
    _x = np.arange(0.9*x.min(), 1.1*x.max(), 0.01)
    _y = f(_x, fit)

    r_squared = r_sq(f(x, fit), y)

    plt.scatter(x[npb_ids], y[npb_ids], color="blue", label="NPB")
    plt.scatter(x[splash2x_ids], y[splash2x_ids], color="green",
        label="Splash2X")
    plt.scatter(x[easy_ids], y[easy_ids], color="magenta", label="easy")
    #plt.plot([], [], "*", color="y", label="R² = %f" % r_squared)
    plt.plot(_x, _y, color="red", label="regressão (R² = %f)" % r_squared)
    plt.xlabel("(mem_acc_to_sqrtsumsq/mem_acc_sum)*frac_stalled_cycles_backend")
    plt.ylabel("speedup")
    plt.legend(loc=1)
    plt.title("Autonuma speedup")
    plt.show()

    print("R² = %f" % r_squared)

if __name__ == "__main__":
    main()
