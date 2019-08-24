from parse_json import parse_json
from DecisionTree import DecisionTree
from bagged_trees import BaggedTrees
from boosted_trees import BoostedTrees
import sys
import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred, metadata):
    """Confusion matrix.

    Parameters
    ----------
    y_true : array, shape = [n_instances]
        Actual target values.
    y_pred : array, shape = [n_instances]
        Predicted target values.

    Returns
    -------
    cm : DataFrame
        Returns the confusion matrix.
    """
    classes = metadata['features'][-1][1]
    y_pred = pd.Categorical(y_pred, categories=classes)
    y_true = pd.Categorical(y_true, categories=classes)

    return pd.crosstab(y_pred, y_true,
                       rownames=['Predicted'], colnames=['Actual'], dropna=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        method = "bag"
        max_trees = 5
        max_depth = 2
        train = "datasets/digits_train.json"
        test = "datasets/digits_test.json"
    else:
        tmp, method, max_trees, max_depth, train, test = sys.argv
        max_trees = int(max_trees)
        max_depth = int(max_depth)
    
    # load in data
    X_train, y_train, meta_train = parse_json(train, method="Numpy")
    X_test, y_test, meta_test = parse_json(test, method="Numpy")

    if method == "bag":
        bdt = BaggedTrees(max_trees, max_depth)
    elif method == "boost":
        bdt = BoostedTrees(max_trees, max_depth)

    bdt.fit(X_train, y_train, meta_train, verbose=False)
    y_pred = bdt.predict(X_test, None, verbose=False)

    # print(confusion_matrix(y_test, y_pred, meta_test))

    # foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    # bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
    # print(pd.crosstab(foo, bar, dropna=False))

    # print out the confusion matrix
    tmp = np.array(confusion_matrix(y_test, y_pred, meta_test))
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if j != tmp.shape[1]-1:
                print("{},".format(tmp[i,j]), end="")
            else:
                print("{}".format(tmp[i,j]))
    



    