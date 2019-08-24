from bayes import BayesNet
from parse_json import parse_json
from accuracy import accuracy_score
import pandas as pd
import numpy as np
import math
from scipy import stats

if __name__ == "__main__":

    data = "datasets/tic-tac-toe.json"
    k = 10
    X, y, meta = parse_json(data)

    np.random.seed(8)
    random_index = [i for i in range(len(X))]
    np.random.shuffle(random_index)
    
    # split X and y into 10 folds
    X_split = np.array_split(X.ix[random_index,:].reset_index(drop=True), k)
    y_split = np.array_split(y.ix[random_index].reset_index(drop=True), k)
    
    deltas = np.zeros(k)
    for i in range(k):
        acc = []
        for net_type in ["n","t"]:
            X_train = pd.concat(X_split[:i]+X_split[i+1:]).reset_index(drop=True)
            y_train = pd.concat(y_split[:i]+y_split[i+1:]).reset_index(drop=True)
            X_test = X_split[i].reset_index(drop=True)
            y_test = y_split[i].reset_index(drop=True)
            
            bn = BayesNet(net_type)
            bn.fit(X_train, y_train, meta)
            y_pred = bn.predict(X_test,None,verbose=False)

            acc.append(accuracy_score(y_test, y_pred))

        deltas[i] = acc[1]-acc[0] # TAN - NB  

    mean = deltas.mean()
    t_stat = mean / (np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
    df = len(deltas)-1
    p_value = 2*(1-stats.t.cdf(abs(t_stat),df))

    print("Mean: {}".format(mean))
    print("T-statistic: {}".format(t_stat))
    print("P-value: {}".format(p_value))
    