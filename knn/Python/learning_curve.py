from knn_classifier import parse_json, KNNClassifier
from accuracy import accuracy_score
import math
import sys
import pandas as pd

if __name__ == "__main__":

    if len(sys.argv) == 1:
        k = 10
        train = "datasets/votes_train.json"
        test = "datasets/votes_test.json"
    else:
        k = int(sys.argv[1])
        train = str(sys.argv[2])
        test = str(sys.argv[3])

    # parse the json files for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)

    for i in range(10):
        N = X_train.shape[0]
        ind = math.floor((i+1)*N/10 - 1) # subtract 1 since indexing starts at 0

        knn = KNNClassifier(k=k)
        knn.fit(X_train.ix[0:ind,:], y_train.ix[0:ind], meta_train)
        y_pred = knn.predict(X_test, verbose=False)

        acc = accuracy_score(y_test, y_pred)

        print(X_train.ix[0:ind,:].shape[0], end="")
        print(",{}".format(acc))
        
