from knn_classifier import parse_json, KNNClassifier
from accuracy import accuracy_score
import sys
import pandas as pd

if __name__ == "__main__":

    if len(sys.argv) == 1:
        max_k = 20
        train = "datasets/digits_train.json"
        val = "datasets/digits_val.json"
        test = "datasets/digits_test.json"
    else:
        max_k = int(sys.argv[1])
        train = str(sys.argv[2])
        val = str(sys.argv[3])
        test = str(sys.argv[4])

    # parse the json files for data
    X_train, y_train, meta_train = parse_json(train)
    X_val, y_val, meta_val = parse_json(val)
    X_test, y_test, meta_test = parse_json(test)
    
    # train classifier on TRAIN, predict on VAL (for k=1,2,...,max_k)
    acc = {}
    for k in range(1,max_k+1):
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train, meta_train)
        y_pred = knn.predict(X_val, verbose=False)

        acc[k] = accuracy_score(y_val, y_pred)
        print("{},{}".format(k,acc[k]))

    best_k = max(acc, key=lambda key: acc[key]) # note that 'max' always returns first value in case of ties
    print(best_k)

    # train on TRAIN + VAL, predict on TEST
    knn_best = KNNClassifier(k=best_k)

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    knn_best.fit(X_train_val, y_train_val, meta_train)
    y_pred = knn_best.predict(X_test, verbose=False)

    test_acc = accuracy_score(y_test, y_pred)
    print(test_acc)

    


    
