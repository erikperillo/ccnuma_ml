#!/usr/bin/env python3

import numpy as np
import pandas as pd
import itertools
import operator
import random
import pickle
import oarg
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#parameters for SVM classifier grid search
SVM_GRID_SEARCH_PARAMS = {
    "C": [2**n for n in range(-10, 11)],
    "gamma": [2**n for n in range(-10, 11)],
    "kernel": ["rbf"],
    "class_weight": [None, "balanced"]
}

#parameters for random forest classifier grid search
RF_GRID_SEARCH_PARAMS = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [x for x in range(4, 13, 2)],
    #"max_features": [2, "sqrt", "log2", None],
    "max_features": [2, "sqrt"],
    #"max_depth": [2*x for x in range(6, 11)] + [None],
    "max_depth": [2*x for x in range(6, 9)],
    "class_weight": [None],
    "random_state": [55]
}

#benchmarks to use in test split
TEST_SAMPLES = [
    #autonuma class
    "prod:easy",
    "water_nsquared:splash2x",
    "radix:splash2x",
    "fft:splash2x",
    #node_local class
    "cholesky:splash2x",
    "raytrace:splash2x",
    #interleave class
    "water_spatial:splash2x",
    "ocean_cp:splash2x",
    "lu_ncb:splash2x",
    "cg.c:npb",
    "lu.c:npb"
]

#path of csv file with data
DATA_FILEPATH = "./data/data.csv"

#directory to store models
MODELS_DIR = "./models"

def error(msg, code=1):
    """Prints error message <msg> and exits with code <code>."""
    print("error:", msg)
    exit(code)

def indexes(df, field, values):
    """Gets row indexes in <df> where column <field> has one of <values>."""
    return [df[field][df[field] == v].index.tolist().pop() for v in values]

def grid_search(X_tr, X_te, y_tr, y_te, clf, grid, verbose=True):
    """Performs grid search to find best parameter for classifier <clf>.
    <X_tr, X_te, y_tr, y_tr> are X and y train and test sets.
    <grid> is a dict in format {'parameter': [list, of, values]}.
    If <verbose> is True, prints some information."""
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

def train(data_filepath, clf="svm", as_test=None, save_model_to=None,
    rand_state=42):
    """Fits SVM multiclass for csv data in <data_filepath>, performing a grid
    search to get the best parameters.
    <as_test> is either None or a list of benchmark names to use in test split.
    <save_model_to> is either None or a string with a filename to save the best
    model found using pickle."""
    #cheking arguments
    if clf != "rf" and clf != "svm":
        error("regression must be 'rf' (random forest) or 'svm' (SVM)")

    #reading dataframe from csv file
    df = pd.read_csv(data_filepath)
    df = df.drop(["speedup_il", "speedup_an"], axis=1)

    #formatting data for classification
    X = df.drop(["name:suite", "best_policy"], axis=1).as_matrix()
    y = df["best_policy"].values

    #selection of indexes for train and test
    if isinstance(as_test, list):
        te_idxs = indexes(df, "name:suite", as_test)
        tr_idxs = list(set(range(len(df))) - set(te_idxs))
        cv_gen = zip([tr_idxs], [te_idxs])
    else:
        kfold = StratifiedKFold(n_splits=as_test, shuffle=True,
            random_state=rand_state)
        cv_gen = kfold.split(X, y)

    #estimator
    est = RandomForestClassifier() if clf == "rf" else SVC()
    #grid parameters
    gs_params = RF_GRID_SEARCH_PARAMS if clf == "rf" else SVM_GRID_SEARCH_PARAMS
    #grid search
    gs = GridSearchCV(est, gs_params, cv=cv_gen)

    #training
    print("training %s..." % clf, end="", flush=True)
    gs.fit(X, y)
    print(" done.")

    #getting best estimator found in grid search
    est = gs.best_estimator_

    #showing results
    #print("cv results:\n", gs.cv_results_)
    #print("mean test scores:\n", gs.cv_results_["mean_test_score"]),
    print("best parameters:", gs.best_params_)
    #print("mean of mean test scores:",
    #    np.array(gs.cv_results_["mean_test_score"]).mean())
    print("best score:", gs.best_score_)
    print("total score:", est.score(X, y))
    preds = pd.DataFrame({"name:suite": df["name:suite"], "y": y,
        "predicted_y": est.predict(X)})
    print(preds)

    #saving model if required
    save_model_to = None
    if save_model_to is not None:
        with open(save_model_to, "wb") as f:
            pickle.dump(est, f)
        print("model saved to '%s'" % save_model_to)

def main():
    clf = oarg.Oarg("-c --clf", "rf", "regression model (rf/svm)", 0)
    hlp = oarg.Oarg("-h --help", False, "this help message")
    data_filepath = oarg.Oarg("-f --data-filepath", DATA_FILEPATH,
        "data filepath", 2)
    rand_state = oarg.Oarg("-s --rand-state", 2531, "random state for kfold")
    oarg.parse()

    if hlp.val:
        oarg.describe_args("options:")
        exit()

    train(data_filepath.val,
        clf=clf.val,
        as_test=3,
        save_model_to="%s/classifier_%s.pkl" % (MODELS_DIR, clf.val),
        rand_state=rand_state.val)

if __name__ == "__main__":
    main()

#good rs: 45, 24
