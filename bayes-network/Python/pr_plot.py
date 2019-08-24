from bayes import BayesNet
from parse_json import parse_json
# from accuracy import accuracy_score
import pandas as pd
import numpy as np
# import math
# from scipy import stats
import matplotlib.pyplot as plt

if __name__ == "__main__":

    train = "datasets/tic-tac-toe_train.json"
    test = "datasets/tic-tac-toe_test.json"

    # parse the json file for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)
    
    m = len(y_test)
    precision = np.zeros((m,2))
    recall = np.zeros((m,2))
    
    for i,net_type in enumerate(["n","t"]):  

        bn = BayesNet(net_type)
        bn.fit(X_train, y_train, meta_train)
        y_conf = bn.predict(X_test, None, verbose=False, confidence=True)

        thresholds = -1*np.sort(-1*y_conf, kind='stable') 

        pos = meta_test["features"][-1][1][0]
        ActP = np.full(m, sum(y_test == pos))
        TP = np.zeros(m)
        PredP = np.zeros(m)
        for j,thresh in enumerate(thresholds):
            pred_pos = y_conf >= thresh # predict positive if confidence is greater than threshold
            actual_pos = np.array(y_test == pos)
            TP[j] = np.logical_and(pred_pos, actual_pos).sum()
            PredP[j] = sum(pred_pos)

        precision[:,i] = TP / PredP
        recall[:,i] = TP / ActP


    # create the plot
    plt.plot(recall[:,0], precision[:,0], label='Naive Bayes')
    plt.plot(recall[:,1], precision[:,1], label='TAN')
    
    plt.xlim((-0.05,1.05))
    plt.ylim((-0.05,1.05))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR Curve for Bayes Classifier on Tic-Tac-Toe dataset")

    plt.legend(loc="lower left")
    plt.savefig("./images/pr_plot.pdf")


        

