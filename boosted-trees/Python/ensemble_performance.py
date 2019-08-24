from parse_json import parse_json
from DecisionTree import DecisionTree
from bagged_trees import BaggedTrees
from boosted_trees import BoostedTrees
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":
    max_trees = range(1,15)
    max_depth = [2, 4, 10]
    # max_trees = range(1,3)
    # max_depth = [2,4]
    train = "datasets/digits_train.json"
    test = "datasets/digits_test.json"
    
    # load in data
    X_train, y_train, meta_train = parse_json(train, method="Numpy")
    X_test, y_test, meta_test = parse_json(test, method="Numpy")

    for mt in max_trees:
        for md in max_depth:
            bag_dt = BaggedTrees(mt, md)
            bag_dt.fit(X_train, y_train, meta_train, verbose=False)
            y_pred = bag_dt.predict(X_test, None, verbose=False)
            bag_acc = (y_pred == y_test).mean()
            
            boost_dt = BoostedTrees(mt, md)
            boost_dt.fit(X_train, y_train, meta_train, verbose=False)
            y_pred = boost_dt.predict(X_test, None, verbose=False)
            boost_acc = (y_pred == y_test).mean()

            print("{},{},{},{}".format(mt, md, "boost", boost_acc))
            print("{},{},{},{}".format(mt, md, "bag", bag_acc))
            
    
