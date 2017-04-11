#!/usr/bin/env python3

import numpy as np
import pandas as pd
import itertools
import operator
import random
import pickle
from sklearn.svm import SVC

SVC_GRID_SEARCH_PARAMS = {
    "C": [2**n for n in range(-5, 5)],
    "gamma": [2**n for n in range(-5, 5)],
    "kernel": ["rbf"]
}
DATA_FILEPATH = "./data_clf.csv"
MODEL_FILEPATH = "./classifier_svm.pkl"

def grid_search(X_tr, X_te, y_tr, y_te, clf, grid, verbose=True):
    #statistics for grid search in format (params, tr_score, te_score)
    stats = []
    #print function. if verbose is false, does not print anything
    vprint = print if verbose else (lambda *args, **kwargs: None)

    #performing grid search
    keys = grid.keys()
    values = grid.values()
    for combination in itertools.product(*values):
        params = {k: v for k, v in zip(keys, combination)}
        vprint("params:", params)
        clf.set_params(**params)

        vprint("\ttraining...", end="")
        clf.fit(X_tr, y_tr)
        vprint(" done.")

        tr_score = clf.score(X_tr, y_tr)
        te_score = clf.score(X_te, y_te)
        vprint("\ttrain score: %.4f | test score: %.4f" % (tr_score, te_score))

        stats.append((params, tr_score, te_score))

    #sorting by te_score and tr_score, in this order
    stats.sort(key=operator.itemgetter(2, 1), reverse=True)
    return stats

def _r_squared(est_y, y):
    ss_tot = ((y - y.mean())**2).sum()
    ss_res = ((y - est_y)**2).sum()
    return 1 - ss_res/ss_tot

def r_squared(reg, x, y):
    est_y = reg.predict(x)
    return _r_squared(est_y, y)

def shuf(lst):
    lst = list(lst)
    random.shuffle(lst)
    return lst

def indexes(df, field, values):
    ids = []
    for v in values:
        ids.append(df[field][df[field] == v].index.tolist()[0])
    return ids

def error(msg, code=1):
    print("error:", msg)
    exit(1)

transform = {
    "time_elapsed_mean": ("time_elapsed_s", 0),
    "instructions_mean": ("T_instructions", 12),
    "cycles_mean": ("T_cycles", 12),
    "stalled-cycles-backend_mean": ("T_stalled_cycles_backend", 12),
    "lsq_load_mean": ("T_lsq_load_evs", 12),
    "lsq_store_mean": ("T_lsq_store_evs", 12),
    "speedup_il": ("speedup_il", 0),
    "speedup_an": ("speedup_an", 0),
    "loc_mem_acc_sum": ("G_loc_mem_acc_sum", 9),
    "rem_mem_acc_sum": ("G_rem_mem_acc_sum", 9),
    "loc_mem_acc_max": ("G_loc_mem_acc_max", 9),
    "rem_mem_acc_max": ("G_rem_mem_acc_max", 9),
    "rem_mem_acc_sqrtsumsq": ("G_rem_mem_acc_sqrtsumsq", 9)
}

def get_clases(il_speedups, an_speedups):
    def clase(speedups):
        print(speedups)
        il_s, an_s = speedups
        if il_s < 1.01 and an_s < 1:
            return "node_local"
        return "interleave" if (il_s > an_s) else "autonuma"
    return list(map(clase, zip(il_speedups, an_speedups)))

def transf(df):
    df = df.drop(["mem_acc_evs_mean"], 1)
    df = df.rename(columns={k: transform[k][0] for k in transform.keys()})
    for v in transform.values():
        k, power = v
        df[k] = df[k]/(10**power)
    return df

def div(df):
    cols = [c for c in df.columns.values if c not in ["name:suite"]]
    #df = df.drop(["mem_acc_evs_mean"], axis=1)
    for c in cols:
        print("%s: %.3g" % (c, df[c].values.mean()))
        df[c] = df[c]/df[c].values.mean()
    return df

def transform_df(data_filepath):
    #opening csv file and formating input data for regression
    df = pd.read_csv(data_filepath)
    df = transf(df)
    clases = get_clases(df["speedup_il"].values, df["speedup_an"].values)
    df["best_policy"] = clases
    df.to_csv("data_clf.csv", index=False)

TEST_SAMPLES = [
    #autonuma
    "add:easy",
    #node_local
    "cholesky:splash2x",
    "lu_cb:splash2x",
    #interleave
    "prod:easy",
    "barnes:splash2x",
    "water_spatial:splash2x",
    "ocean_cp:splash2x",
    "lu_ncb:splash2x",
    "bt.c:npb",
    "cg.c:npb",
    #"lu.c:npb"
]

def indexes(df, field, values):
    return [df[field][df[field] == v].index.tolist().pop() for v in values]

def clf(data_filepath, as_test=None, weight=False, save_model_to=None):
    """
    Fits SVM multiclass for csv data in <data_filepath>, performing a grid
    search to get the best parameters.
    <as_test> is either None or a list of benchmark names to use in test split.
    If <weight> is true, samples have a different C value according to its
    classes' frequency in dataset.
    <save_model_to> is either None or a string with a filename to save the best
    model found using pickle.
    """
    #reading dataframe from csv file
    df = pd.read_csv(data_filepath)
    df = df.drop(["speedup_il", "speedup_an"], axis=1)

    #formatting data for classification
    X = df.drop(["name:suite", "best_policy"], axis=1).as_matrix()
    y = df["best_policy"].values

    #getting train and test indexes
    if as_test is None:
        tr_te_idxs = None
    elif isinstance(as_test, str) and as_test.lower() in ["rand", "random"]:
        error("'random' not supported yet.")
    elif isinstance(as_test, list):
        te_idxs = indexes(df, "name:suite", as_test)
        tr_idxs = list(set(range(len(df))) - set(te_idxs))
        tr_te_idxs = (tr_idxs, te_idxs)
    else:
        error("as_test must be either None, 'random' or a list.")
    #getting train and test splits
    X_tr, X_te = X[tr_idxs, :], X[te_idxs, :]
    y_tr, y_te = y[tr_idxs], y[te_idxs]

    #setting weights to classes based on their frequencies if required
    if weight:
        _y = list(y)
        weights = {k: (_y.count(k)/len(_y))**-1 for k in set(_y)}
        C_vals = SVC_GRID_SEARCH_PARAMS["C"]
        del SVC_GRID_SEARCH_PARAMS["C"]
        class_weight = [{k: weights[k]*c for k in weights} for c in C_vals]
        SVC_GRID_SEARCH_PARAMS["class_weight"] = class_weight

    print("performing grid search...")
    #instantiating classifier
    clf = SVC()
    #calling grid_search
    stats = grid_search(X_tr, X_te, y_tr, y_te,
        clf, SVC_GRID_SEARCH_PARAMS, verbose=True)
    print("done. 5 best parameters sets:")
    print("\n".join(map(str, stats[:5])))

    #getting best parameters
    best_params = stats[0][0]
    #setting parameters
    clf.set_params(**best_params)
    print("fitting for all data using best parameters...", end="")
    clf.fit(X, y)
    print(" done. score: %.4f" % clf.score(X, y))
    print("y, predicted y:\n%s" % "\n".join(map(str, zip(y, clf.predict(X)))))

    if save_model_to is not None:
        with open(save_model_to, "wb") as f:
            pickle.dump(clf, f)
        print("model saved to '%s'" % save_model_to)

def main():
    clf(DATA_FILEPATH,
        as_test=TEST_SAMPLES,
        weight=False,
        save_model_to=MODEL_FILEPATH)

if __name__ == "__main__":
    main()
