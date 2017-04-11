#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import random
import pickle
import oarg

#parameters for SVM regressor grid search
SVM_GRID_SEARCH_PARAMS = {
    "C": [2**n for n in range(-11, 11)],
    "gamma": [2**n for n in range(-11, 11)],
    "kernel": ["rbf"]
}

#parameters for random forest regressor grid search
RF_GRID_SEARCH_PARAMS = {
    #"criterion": ["mse", "mae"],
    "criterion": ["mse"],
    #"n_estimators": [x for x in range(2, 10)],
    "n_estimators": [x for x in range(2, 6)],
    #"max_features": [2, "auto", "sqrt", "log2", None],
    "max_features": [2, "sqrt"],
    #"max_depth": [2*x for x in range(2, 10)] + [None],
    "max_depth": [2*x for x in range(2, 5)], # + [None],
    #"random_state": [21, 34, 55]
    "random_state": [34, 55]
}

#benchmarks to use in test split
TEST_SAMPLES = [
    "add:easy",
    "prod:easy",
    "ep.c:npb",
    "ft.c:npb",
    "mg.c:npb",
    "ocean_ncp:splash2x",
    "water_spatial:splash2x",
    "lu_ncb:splash2x",
    "barnes:splash2x"
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

def get_suites(df):
    return list(map(lambda ns: ns.split(":")[1], df["name:suite"].values))

def train(data_filepath, mode="il", reg="rf", as_test=3, save_model_to=None,
    rand_state=42):
    """Grid search for regression on speedup for mem policy specified by <mode>.
    <reg> specifies the regressor (Random Forest or svm).
    <as_test> is either int (giving k in k-fold) or a list of benchmark names
    to be used as test samples.
    <save_model_to> is either a string specifying a path to save model or None.
    Saves best estimator using pickle."""
    #cheking arguments
    if mode != "il" and mode != "an":
        error("mode must be 'il' (interleave) or 'an' (autonuma)")
    if reg != "rf" and reg != "svm":
        error("regression must be 'rf' (random forest) or 'svm' (svm)")

    #opening csv file and formating input data for regression
    df = pd.read_csv(data_filepath)

    #formatting data for regression
    X = df.drop(["name:suite", "speedup_il", "speedup_an", "best_policy"],
        axis=1).as_matrix()
    y = df["speedup_" + mode].values

    #selection of indexes for train and test
    if isinstance(as_test, list):
        te_idxs = indexes(df, "name:suite", as_test)
        tr_idxs = list(set(range(len(df))) - set(te_idxs))
        cv_gen = zip([tr_idxs], [te_idxs])
    else:
        #good ones: 805(svm-il), 1982(svm-an), 805(rf-il), 1350(rf-an)
        suites = get_suites(df)
        kfold = StratifiedKFold(n_splits=as_test, shuffle=True,
            random_state=rand_state)
        cv_gen = kfold.split(X, suites)

    #estimator
    est = RandomForestRegressor() if reg == "rf" else SVR()
    #grid parameters
    gs_params = RF_GRID_SEARCH_PARAMS if reg == "rf" else SVM_GRID_SEARCH_PARAMS
    #grid search
    gs = GridSearchCV(est, gs_params, cv=cv_gen)

    #training
    print("mode: %s | training %s ..." % (mode, reg), end="", flush=True)
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
    #preds = pd.DataFrame({"name:suite": df["name:suite"], "y": y,
    #    "predicted_y": est.predict(X)})
    #print(preds)

    #saving best estimator if required
    if save_model_to is not None:
        with open(save_model_to, "wb") as f:
            pickle.dump(est, f)
        print("model saved to '%s'" % save_model_to)

def main():
    mode = oarg.Oarg("-m --mode", "il", "memory policy mode (il/an)", 0)
    reg = oarg.Oarg("-r --reg", "rf", "regression model (rf/svm)", 1)
    data_filepath = oarg.Oarg("-f --data-filepath", DATA_FILEPATH,
        "data filepath", 2)
    rand_state = oarg.Oarg("-s --rand-state", 2531, "random state for kfold")
    do_not_save = oarg.Oarg("-S", False, "do not save model")
    hlp = oarg.Oarg("-h --help", False, "this help message")
    oarg.parse()

    if hlp.val:
        oarg.describe_args("options:")
        exit()

    train(data_filepath.val,
        mode=mode.val,
        reg=reg.val,
        as_test=3,
        save_model_to=None if do_not_save.val else
            ("%s/estimator_%s_%s.pkl" % (MODELS_DIR, mode.val, reg.val)),
        rand_state=rand_state.val)

if __name__ == "__main__":
    main()
