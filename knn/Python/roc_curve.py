from knn_classifier import parse_json, KNNClassifier
from accuracy import accuracy_score
import numpy as np
import pandas as pd
import sys

def roc_curve(y_true, y_conf, metadata, verbose=True):
    
    classes_ = metadata["features"][-1][1]
    # assign 'positive' to first class
    pos = classes_[0]
    m = len(y_true)
    num_pos = sum(y_true == pos)
    num_neg = m - num_pos
    
    TP = 0
    FP = 0
    last_TP = 0
    order = np.argsort(-1*y_conf, kind='stable')
    for i in order:
        # find threshold where prev y is pos and current y is negative
        if (i !=  order[0]) and y_conf[i] != y_conf[i-1] and y_true[i] != pos and TP > last_TP:
            FPR = FP / num_neg
            TPR = TP / num_pos
            if verbose: print("{},{}".format(FPR,TPR))
            last_TP = TP
        if y_true[i] == pos:
            TP += 1
        else: 
            FP += 1
    FPR = FP / num_neg
    TPR = TP / num_pos
    if verbose: print("{},{}".format(FPR,TPR))


if __name__ == "__main__":

    if len(sys.argv) == 1:
        k = 30
        train = "datasets/votes_train.json"
        test = "datasets/votes_test.json"
    else:
        k = int(sys.argv[1])
        train = str(sys.argv[2])
        test = str(sys.argv[3])

    # parse the json files for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)

    # fit KNN and predict confidence
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train, meta_train)
    y_conf = knn.predict(X_test, verbose=False, confidence=True)

    roc_curve(y_test, y_conf, meta_test, verbose=True)
    
