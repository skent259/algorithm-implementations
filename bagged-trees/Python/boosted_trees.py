from parse_json import parse_json
from DecisionTree import DecisionTree
import sys
import numpy as np


class BoostedTrees(object):
    """
    TODO: 

    Assumptions:
    - Class (desired prediction) is last attribute in metadata and is named "class"
    
    Parameters
    ----------
    max_trees : int, optional (default=2)
        Maximum number of trees to fit
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

    def __init__(self, max_trees=2, max_depth=3, random_state=0):
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_trees = None


    def fit(self, X, y, metadata, verbose=False):

        self.metadata = metadata

        # initialize weights as 1/n
        n_instances, n_features = X.shape
        n_classes = len(self.metadata['features'][-1][1])
        self.weights = np.zeros((n_instances,self.max_trees))
        self.weights[:,0] = 1/n_instances

        self.trees = []
        self.betas = []
        # iterate until max trees is reached (or we break)
        for t in range(self.max_trees):
            weight = self.weights[:,t]

            # fit classifier with weights
            tree = DecisionTree()
            # print(self.weights.shape)
            tree.fit(X, y, self.metadata['features'], max_depth=self.max_depth, instance_weights=weight)
            self.trees.append(tree)

            # compute weighted error
            y_pred = tree.predict(X)

            weighted_error = sum(weight * (y_pred != y)) / sum(weight)
            # if weighted error is too high, break the loop
            if weighted_error > 1 - 1/n_classes:
                self.n_trees = t
                break

            # compute beta
            beta = np.log((1-weighted_error)/weighted_error) + np.log(n_classes - 1)
            self.betas.append(beta)

            # update weights and re-normalize
            if t != self.max_trees - 1:
                self.weights[:,t+1] = weight * np.exp(beta * (y_pred != y)) 
                self.weights[:,t+1] = self.weights[:,t+1] / sum(self.weights[:,t+1])
        
        if self.n_trees is None:  
            self.n_trees = self.max_trees
            
        if verbose:
            # print out weights for training instances on the trees
            tmp = self.weights
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    if j != tmp.shape[1]-1:
                        print("{:.12f},".format(tmp[i,j]), end="")
                    else:
                        print("{:.12f}".format(tmp[i,j]))
            print("")
            # print out tree weights
            print(*self.betas, sep=',')
            print("")
                    
                

    def predict(self, X, y_true, verbose=False):
        
        n_instances, n_features = X.shape
        classes = self.metadata['features'][-1][1]
        n_classes = len(classes)

        predictions = [tree.predict(X) for tree in self.trees]
        score = np.zeros((n_instances, n_classes))
        for i, class_ in enumerate(classes):
            sum_ = np.zeros(n_instances)
            for t in range(self.n_trees):
                sum_ += self.betas[t] * (predictions[t] == class_)
            score[:,i] = sum_

        maxind = np.argmax(score, axis=1)
        
        y_pred = []
        for ind in maxind:
            y_pred.append(self.metadata['features'][-1][1][ind])
                       
        if verbose:
            # print out the predictions
            for i in range(X.shape[0]):
                # predictions from individual trees
                for p in [tree.predict(X) for tree in self.trees]:
                    # ind = np.argmax(p[i,:])
                    # pred = self.metadata['features'][-1][1][ind]
                    print("{},".format(p[i]), end="")
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
        max_trees = 2
        max_depth = 3
        train = "datasets/digits_train.json"
        test = "datasets/digits_test.json"
    else:
        tmp, max_trees, max_depth, train, test = sys.argv
        max_trees = int(max_trees)
        max_depth = int(max_depth)
    
    # load in data
    X_train, y_train, meta_train = parse_json(train, method="Numpy")
    X_test, y_test, meta_test = parse_json(test, method="Numpy")

    # Run bagged trees algorithm 
    bdt = BoostedTrees(max_trees, max_depth)
    bdt.fit(X_train, y_train, meta_train, verbose=True)
    bdt.predict(X_test, y_test, verbose=True)

