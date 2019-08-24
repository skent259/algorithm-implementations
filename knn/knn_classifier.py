"""
usage: knn_classifier <INT k> <TRAINING SET> <TEST SET>
"""
import sys
import json
import numpy as np
import pandas as pd

def parse_json(file_path):
        """
        Takes file path and parses json into data and metadata
        """
        with open(file_path, "r") as read_file:
            dataset = json.load(read_file)
        metadata = dataset["metadata"]
 
        """ Pandas """
        col_names = [i[0] for i in dataset.get("metadata").get("features")]
        data = pd.DataFrame(dataset.get("data"), columns = col_names)   
        X = data.ix[:,:-1] # all but last column
        y = data.ix[:,-1] # just last column

        return X, y, metadata


class KNNClassifier(object):
    
    def __init__(self, k=5):
        self.k = k

    def _set_standardization(self, X):
        """
        Returns 
        ---
        std_constants : dict
            Dictionary with mean and sd vectors for standardization
        """
        n_instances, n_features = X.shape
        numeric_features = [self.metadata["features"][i][1]=="numeric" for i in range(n_features)]
        
        mean = X.ix[:,numeric_features].mean(axis=0)
        sd = X.ix[:,numeric_features].std(axis=0)

        std_constants = {"mean": mean, "sd": sd}
        
        self.std_constants_ = std_constants


    def _standardize(self, X):
        """
        For each continuous feature, subtract mean and divide by standard deviation
        """
        n_instances, n_features = X.shape
        numeric_features = [self.metadata["features"][i][1]=="numeric" for i in range(n_features)]
        X_std = X.copy()

        X_std.ix[:,numeric_features] = np.subtract(X.ix[:,numeric_features], self.std_constants_["mean"]) / self.std_constants_["sd"]
        
        return X_std


    def _compute_distance(self, X, instance):

        n_instances, n_features = X.shape

        numeric_features = [self.metadata["features"][i][1]=="numeric" for i in range(n_features)]
        categorical_features = [not x for x in numeric_features]

        dist_numeric = np.absolute( np.subtract(X.ix[:,numeric_features], instance.ix[numeric_features]) ) # manhattan distance
        dist_categorical = 1- np.equal(X.ix[:,categorical_features],instance.ix[categorical_features]) 
        distance = dist_numeric.sum(axis=1) + dist_categorical.sum(axis=1)

        return distance


    def _find_knn(self, distances):
        """ 
        return boolean vector matching the k-nearest neighbors based on distances vector
        """
        knn_ind = np.argsort(distances, kind='stable')[0:self.k]

        return knn_ind 
        

    def fit(self, X, y, metadata):
        """
        set standardization from training set and store X, y for prediction
        """
        self.metadata = metadata
        self._set_standardization(X)
        self.X_train = self._standardize(X)
        self.y_train = y


    def predict(self, X, verbose=True, confidence=False):
        """
        Predict 
        for each instance in test set, compute distances to training set, find k-nearest neighbors, tally votes from those neighbors
        """
        n_instances, n_features = X.shape

        X_std = self._standardize(X)
        y_pred = []
        y_conf = []
        eps = 1e-5
        classes_ = self.metadata["features"][-1][1]

        for i in range(n_instances):            
            dist = self._compute_distance(self.X_train, X_std.ix[i,:])
            knn_ind = self._find_knn(dist)

            votes = {}
            for class_ in classes_:
                votes[class_] = sum(self.y_train.ix[knn_ind] == class_)
                if verbose: print("{},".format(votes[class_]), end='')

            top_vote = max(votes, key=lambda key: votes[key]) # note that 'max' always returns first value in case of ties
            y_pred.append(top_vote)
            if verbose: print(top_vote)

            # note: confidence values assume binary classification, and first class is positive (=1)
            if confidence:
                w = 1 / (dist[knn_ind]**2 + eps)
                y = self.y_train.ix[knn_ind] == classes_[0]
                conf = sum(w*y) / sum(w)
                y_conf.append(conf)

            if sum(votes.values()) != self.k:
                print("Warning: _find_knn not returning k instances")

        if confidence:  
            return np.array(y_conf)
        else: 
            return np.array(y_pred)


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        k = 10
        train = "datasets/votes_train.json"
        test = "datasets/votes_test.json"
    else:
        k = int(sys.argv[1])
        train = str(sys.argv[2])
        test = str(sys.argv[3])

    # parse the json file for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)
    
    knn = KNNClassifier(k=k)

    knn.fit(X_train, y_train, meta_train)
    knn.predict(X_test, verbose=True)
    
    