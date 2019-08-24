from parse_json import parse_json
from DecisionTree import DecisionTree
import sys
import numpy as np


class BaggedTrees(object):
    """
    Boostrap Aggregation for Decision Trees.  Fits multiple decision trees and combines the prediction from all trees when making a prediction

    Assumptions:
    - Class (desired prediction) is last attribute in metadata and is named "class"
    
    Parameters
    ----------
    n_trees : int, optional (default=2)
        Numbers of trees to fit
    max_depth : int, optional (default=3)
        Maximum depth for a fitted decision tree
    random_state: int, optional (default=0)
        Chosen random seed

    Attributes
    ----------
    metadata : json object

    X_train : array, shape = [n_instance, n_features]

    y_train : array-like, shape (n_instances,)

    """

    def __init__(self, n_trees=2, max_depth=3, random_state=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state


    def fit(self, X, y, metadata, verbose=False):
        # create n_trees many bagged samples from data
        np.random.seed(self.random_state)
        self.metadata = metadata

        n_instances, n_features = X.shape

        # Randomly subsample X and y with replacement
        ind = [np.random.choice(n_instances, size=(n_instances), replace=True) for i in range(self.n_trees)]
        X_bag = [X[i,:] for i in ind]
        y_bag = [y[i] for i in ind]
        
        # create n_trees many trees from bagged samples and fit them
        self.trees = [DecisionTree() for i in range(self.n_trees)]
        for tree_i, X_i, y_i in zip(self.trees, X_bag, y_bag):
            tree_i.fit(X_i, y_i, metadata['features'], max_depth=self.max_depth)
            
        if verbose:
            # print out indices of bootstrapped samples
            tmp = np.vstack(ind).T
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    if j != tmp.shape[1]-1:
                        print("{},".format(tmp[i,j]), end="")
                    else:
                        print(tmp[i,j])
            print("")
                
                

    def predict(self, X, y_true, verbose=False):

        probs = [tree.predict(X, prob=True) for tree in self.trees]
        probs_avg = np.mean(np.dstack(probs), axis=2)
        
        maxind = np.argmax(probs_avg, axis=1)
        
        y_pred = []
        for ind in maxind:
            y_pred.append(self.metadata['features'][-1][1][ind])
        
        if verbose:
            # print out the predictions
            for i in range(X.shape[0]):
                # predictions from individual trees
                for p in probs:
                    ind = np.argmax(p[i,:])
                    pred = self.metadata['features'][-1][1][ind]
                    print("{},".format(pred), end="")
                # prediction from combined trees
                print("{},".format(y_pred[i]), end="")
                if y_true is not None: 
                    # true label
                    print(y_true[i])

            # print overall accuracy
            print("")
            print( (y_pred == y_test).mean() )


        return np.array(y_pred)
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        n_trees = 2
        max_depth = 3
        train = "datasets/heart_train.json"
        test = "datasets/heart_test.json"
    else:
        tmp, n_trees, max_depth, train, test = sys.argv
        n_trees = int(n_trees)
        max_depth = int(max_depth)
    
    # load in data
    X_train, y_train, meta_train = parse_json(train, method="Numpy")
    X_test, y_test, meta_test = parse_json(test, method="Numpy")

    # Run bagged trees algorithm 
    bdt = BaggedTrees(n_trees, max_depth)
    bdt.fit(X_train, y_train, meta_train, verbose=True)
    bdt.predict(X_test, y_test, verbose=True)

