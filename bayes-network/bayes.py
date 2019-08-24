from parse_json import parse_json
import sys
import json
import numpy as np
import pandas as pd
import itertools
import math

class BayesNet(object):
    """
    Bayesian Classifier that is compatible with running Naive Bayes and Tree Augmented Network structures.

    Assumptions:
    - Solves binary classification problem
    - All attributes are discrete valued.
    - Laplace estimates are used for probabilities (pseudocounts of 1).
    - Class (desired prediction) is last attribute in metadata and is named "class"
    
    Parameters
    ----------
    net_type : string in "n","t"
        "n" determines Naive Bayes classifier, "t" determines TAN classifier

    Attributes
    ----------
    metadata : json object

    X_train : array, shape = [n_instance, n_features]

    y_train : array-like, shape (n_instances,)

    features_ : list, shape (n_features+1)
        Contains names of all features from metadata (including class as final feature)
    graph_structure_ : array, shape = [n_features + 1,n_features + 1]
        Hold the structure of the graph.  A 1 in the [i,j]th entry denotes an arrow from feature i to feature j (includes class as feature).
     """

    def __init__(self, net_type):
        self.net_type = net_type
    
    def Laplace_estimate(self, data, features, given=None, name="prob"):
        """ 
        Returns Laplace estimates (pseudocounts of 1) for probabilities P(features|given) based on data 
        probabilities are given by the "prob" column, with other columns being features and given
        """
        if given==None:
            df = data.groupby(features).size()+1.0
            levels = [self.feature_levels[f] for f in features]
            mux = pd.MultiIndex.from_product(levels, names=features)
            df = df.reindex(mux, fill_value=1.0).reset_index(name="pcount")
            df["prob"] = df["pcount"] / df["pcount"].sum()
        else: 
            all_ = list(np.append(features,given))
            df = data.groupby(all_).size()+1.0
            levels = [self.feature_levels[f] for f in all_]
            mux = pd.MultiIndex.from_product(levels, names=all_)
            df = df.reindex(mux, fill_value=1.0).reset_index(name="pcount")
            df["prob"] = df["pcount"] / df.groupby(given)["pcount"].transform("sum")
        
        df = df.drop("pcount", axis=1)
        df.rename(columns={'prob': name}, inplace=True)
        return df

        
    def conditional_mutual_information(self, X, y):
        """ 
        Return a matrix of conditional mutual information 
        I(X_i,X_j|Y) = Sum_{x_i} Sum_{x_j} Sum_{y} P(x_i, x_j, y) log_2( P(x_i,x_j|y) / ( P(x_i|y) P(x_j|y) ) )
        """
        data = pd.concat([X, y], axis=1)
        y_ind = [data.columns[-1]]
        
        information = {}
        for features in itertools.combinations(X.columns, 2):
            all_ = list(np.append(features,y_ind))
        
            p_xxy = self.Laplace_estimate(data, all_, name="prob_xxy")  # P(x_i, x_j, y) 
            p_xx_y = self.Laplace_estimate(data, features, given=y_ind, name="prob_xx_y") # P(x_i, x_j | y)
            p_xi_y = self.Laplace_estimate(data, features[0], given=y_ind, name="prob_xi_y") # P(x_i | y) 
            p_xj_y = self.Laplace_estimate(data, features[1], given=y_ind, name="prob_xj_y") # P(x_j | y)

            # create merged dataframes for vectorized calculation of CMI
            df_merged = pd.merge(p_xxy, p_xx_y, how="outer", on=all_)
            join = list(np.append(features[0],y_ind))
            df_merged = pd.merge(df_merged, p_xi_y, how="outer", on=join)
            join = list(np.append(features[1],y_ind))
            df_merged = pd.merge(df_merged, p_xj_y, how="outer", on=join)
            
            # calculate mutual information 
            df_merged["Ixxy"] = df_merged["prob_xxy"] * (df_merged["prob_xx_y"]/(df_merged["prob_xi_y"]*df_merged["prob_xj_y"])).apply(lambda x: math.log2(x))
            information[features] = df_merged["Ixxy"].sum()

        return information

    def make_graph(self):
    
        if self.net_type == "n": # Naive Bayes
            # start with dictionary of feature with empty parent edges
            parent_edges = {feature : [] for feature in self.features_[:-1]}
            
        elif self.net_type == "t": # TAN
            # Conditional mutual information as dictionary
            cond_information = self.conditional_mutual_information(self.X_train, self.y_train)
            
            # Prim's algorithm with CMI as weights
            V = set(self.features_[:-1]) # all features except the class
            E = set(cond_information.keys())
            V_new = {self.features_[0]}
            E_new = set()
            
            while V_new != V:
                # find edges emanating from a vertex in V_new to a vertex not in V_new with max weight
                max_weight = max([cond_information[e] for e in E if (e[1] in V_new) ^ (e[0] in V_new)]) # ^ is exclusive or
                edge_to_add = [e for e in E if cond_information[e]==max_weight][0] #TODO: make sure this handles ties appropriately
                
                E_new.add(edge_to_add)
                for v in edge_to_add: V_new.add(v)
            
            # Add nodes based on MST
            def find_children(root, edge_set):
                children = set()
                for e in edge_set:
                    if root in e:
                        children = children.union(set(e))
                children.remove(root)
                return list(children)

            def find_all_children(grandparent, parent, edge_set):
                """ recursive function to get all children based on edges and an initial root"""
                parent_dict = {parent: find_children(parent, edge_set)}
                if grandparent: parent_dict[parent].remove(grandparent)
                
                child_dict = {}
                for child in parent_dict[parent]:
                    child_dict = {**child_dict, **find_all_children(parent, child, edge_set)}

                return {**parent_dict, **child_dict}
            
            root = self.features_[0]
            child_edges = find_all_children(None, root, E_new)
            parent_edges = {feature : [x for x in child_edges if feature in child_edges[x]] for feature in child_edges.keys()}
            
            
        # add class as a parent for all X features
        for i in parent_edges: parent_edges[i].append("class")
        self.graph_structure_ = parent_edges


    def compute_posterior(self, X):
        p_xy = np.zeros((len(X),2))
        
        for i,c in enumerate(self.classes_):
            data = X
            data["class"] = c 
            p_x_y = np.ones(len(X))
            
            for f in self.features_[:-1]:
                parents = self.graph_structure_[f]
                parents_x = list(np.append(parents, f))
                table = self.Px_cond[f]

                d1 = data.loc[:,parents_x]
                d2 = table.loc[:,parents_x]
                index = d1.apply(lambda x: x==d2, axis=1).apply(lambda x: x.min(axis=1))
                p_x_y = p_x_y*[float(table.loc[index.ix[i],"prob"]) for i in range(len(index))]

            p_xy[:,i] = self.Py[i]*p_x_y

        return np.apply_along_axis(lambda x: x/sum(x), 1, p_xy) 
        

    def print_graph(self):
        """ Print graph structure for program output """
        for f in self.features_[:-1]:
            print(f, end=" ")
            for j in self.graph_structure_[f]: print(j, end=" ")
            print("") 
        print("")


    def print_predictions(self, y_pred, y_true, probs):
        """ Print predicted class, actual class, and probability for program output """
        for i in range(len(y_pred)):
            print("{0} {1} {2:.12f}".format(y_pred[i], y_true.ix[i,:], probs[i]))
        print("")


    def fit(self, X, y, metadata):
        """Fit Naive Bayes or TAN.  Store training data and other useful information, find a probabilistic graph structure via Naive Bayes or TAN, and compute conditional probability tables from that graph.  

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_instances,)
            Target values.
        metadata : json object
            Contains information about columns in the training data
        """
        self.metadata = metadata
        self.features_ = [i[0] for i in self.metadata["features"]]
        self.feature_levels = {i[0]:i[1] for i in self.metadata["features"]}
        self.X_train = X
        self.n_instances, self.n_features = self.X_train.shape
        self.y_train = y
        self.data = pd.concat([self.X_train, self.y_train], axis=1)
        self.classes_ = self.metadata["features"][-1][1]
        
        self.make_graph()
        
        # calculate CPT for P(y) and P(X|parents(X))
        self.Py = [(np.sum(self.y_train == c)+1)/(self.n_instances+2) for c in self.classes_] 

        self.Px_cond = {}
        for f in self.features_[:-1]:
            parents = self.graph_structure_[f]
            self.Px_cond[f] = self.Laplace_estimate(self.data, [f], given=parents)
      

    def predict(self, X, y, verbose=True, confidence=False):
        """
        Using the BayesNet that has been fitted, compute posterior probabilities P(Y=y|x) and predict a class by the maximum posterior 

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Testing vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_instances,)
            Target values. Only used for printing the true values.
        metadata : json object
            Contains information about columns in the training data
        """
        posterior = self.compute_posterior(X)
        y_pred = [self.classes_[i] for i in np.argmax(posterior, axis=1)]
        probs = np.max(posterior, axis=1)
        
        if verbose:
            self.print_graph()
            self.print_predictions(y_pred, y, probs)
            print(np.equal(y_pred, y).sum())
            print("")

        if confidence:
            return posterior[:,0]
        else:
            return y_pred



if __name__ == "__main__":

    if len(sys.argv) == 1:
        train = "datasets/lymphography_train.json"
        test = "datasets/lymphography_test.json"
        net_type = "n"        
    else:
        train = str(sys.argv[1])
        test = str(sys.argv[2])
        net_type = str(sys.argv[3])

    # parse the json file for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)
    
    bn = BayesNet(net_type)

    bn.fit(X_train, y_train, meta_train)
    bn.predict(X_test, y_test, verbose=True)
