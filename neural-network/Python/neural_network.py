from parse_json import parse_json
import sys
import json
import numpy as np
import pandas as pd


def sigmoid(X, derivative=False):
    """ Compute sigmoid function of an array """
    sigm = 1. / (1. + np.exp(-X))
    if derivative:
        return X * (1. - X) # assumes that X is output from a sigmoid
    return sigm

def cross_entropy(y_true, y_prob):
    """ 
    Returns cross entropy error for true y and predicted output probabilities 
    E = sum_d -y_d ln(o_d) - (1-y_d) ln(1-o_d) 
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # for i in range(len(y_true)):
    #     print("{} {}".format(y_true[i], y_prob[i]))

    error = - y_true * np.log(y_prob) - (1 - y_true) * np.log(1-y_prob)
    
    return sum(error)

def F1_score(y_true, y_pred):
    """
    Returns the F1 score given by 2*precision*recall / (precision + recall)
    
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred.flatten())
    
    true_pos = sum(y_true * y_pred)
    act_pos = sum(y_true)
    pred_pos = sum(y_pred)

    precision = 0 if pred_pos == 0 else float(true_pos) / float(pred_pos)
    recall = 0 if act_pos == 0 else float(true_pos) / float(act_pos)

    return 0 if precision + recall == 0 else (2*precision*recall) / (precision + recall)
    
def label_binarize(y, classes_=None):
    """Binarize labels in a one-vs-all fashion using one-hot encoding.

    The output will be a matrix where each column corresponds to one possible
    value of the input array, with the number of columns equal to the number
    of unique values in the input array.

    Parameters
    ----------
    y : array, shape = [n_instances,]
        Sequence of integer labels to encode.
    classes_ : array or list
        Set of values that y can take.  Default takes unique values from y.  

    Returns
    -------
    y_bin : array, shape = [n_instances, n_classes]
        Binarized array.
    """
    n_instances = len(y)
    if not classes_: 
        classes_ = np.unique(y)
    else: 
        classes_ = np.array(classes_)
    
    y_bin = np.zeros((n_instances, len(classes_)))
    for i,y_i in enumerate(classes_):
        idx = np.where(y == y_i)
        y_bin[idx, i] = 1

    return y_bin



class NeuralNet(object):
    """
    TODO: Description

    Assumptions:
    - Solves binary classification problem
    - Class (desired prediction) is last attribute in metadata and is named "class"
    
    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        Learning rate for weight updates.
    num_epoch : int, optional (default=20)
        Number of times to run through the data during stochastic gradient descent

    Attributes
    ----------
    metadata : json object

    X_train : array, shape = [n_instance, n_features]

    y_train : array-like, shape (n_instances,)

    """

    def __init__(self, hidden_dim=(100,), batch_size=1, learning_rate=0.01, num_epoch=20, random_state=0):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.random_state = random_state


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
        sd = X.ix[:,numeric_features].std(axis=0, ddof=0)

        std_constants = {"mean": mean, "sd": sd}
        
        return std_constants


    def _standardize(self, X):
        """
        For each continuous feature, subtract mean and divide by standard deviation
        """
        n_instances, n_features = X.shape
        numeric_features = [self.metadata["features"][i][1]=="numeric" for i in range(n_features)]
        X_std = X.copy()

        X_std.ix[:,numeric_features] = np.subtract(X.ix[:,numeric_features], self.std_constants_["mean"]) / self.std_constants_["sd"]
        
        return X_std
   
    def _categorical_to_onehot(self, X):
        """ 
        Transform categorical features into numeric by a onehot encoding in place.
        NOTE: returns a numpy array, but assumes pandas DataFrame as input
        """
        n_instances, n_features = X.shape
        categorical_features = [self.metadata["features"][i][0] for i in range(n_features) if self.metadata["features"][i][1]!= "numeric"]
        X_out = np.ones(n_instances) # start with column of 1's

        for i in range(n_features):
            feature = X.columns[i]
            if feature in categorical_features:
                new_cols = label_binarize(X[feature], classes_=self.metadata["features"][i][1])
            else:
                new_cols = np.array(X[feature]) 

            X_out = np.column_stack((X_out,new_cols))
    
        return X_out
       

    def _initialize_weights(self, layer_dim):
        """ Initialize weight parameters """
        np.random.seed(seed=0)
        self.n_layers_ = len(layer_dim)

        if self.n_layers_ == 2:
            w = np.random.uniform(low=-0.01, high=0.01, size=(1, layer_dim[0]))
            self.weight_ = [w]
        elif self.n_layers_ == 3:
            w_i_h = np.random.uniform(low=-0.01, high=0.01, size=(layer_dim[1], layer_dim[0])) 
            w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, layer_dim[1] + 1))
            self.weight_ = [w_i_h, w_h_o]
        else: 
            self.weight_ = []
            # self.bias_ = []
            for i in range(self.n_layers_ - 1):
                # W, b = self._init_normalized(layer_dim[i], layer_dim[i + 1])
                # self.weight_.append(W)
                # self.bias_.append(b)
                pass 

    def _forward_pass(self, activations):
        """ 
        Perform a forward pass on the network given the current weights
        """
        # at each layer i, multiply activations by weights and feed to appropriate function
        for i, w_l in enumerate(self.weight_):
            s_l = np.matmul(activations[i],w_l.T)
            o_l = sigmoid(s_l)

            if i != (self.n_layers_ - 2):  # for all hidden layers
                # bias = np.ones((1,1))
                # print(bias.shape)
                # print(s_l.shape)
                # s_l = np.concatenate((bias, s_l),axis=1)
                o_l = np.insert(o_l, 0, 1, axis=1) # add bias unit
                # print(s_l.shape)
            #     activations.append(tanh(s_l+self.bias_[i]))
            # else:  #feed to softmax for last layer
            #     activations.append(sigmoid(s_l))
            
            activations.append(o_l)

        return activations 

    def _backprop(self, y, activations, deltas):
        """ Backpropagation to compute sensitivities at each layer. """

        # compute delta at output layer
        delta_L = activations[len(deltas)] - y
        deltas[len(deltas)-1] = delta_L[:,0] # TODO: check this

        # compute deltas for each hidden layer (in backwards order)
        for i in range(len(deltas)-1, 0 , -1):  # subtract 1 to omit the last layer (which is computed above)
            # print(deltas[i].shape)
            # print(self.weight_[i].shape)
            delta_l = sigmoid(activations[i],derivative=True) * (np.matmul(deltas[i].T,self.weight_[i]))
            deltas[i-1] = delta_l

        return deltas        


    def _update_params(self, activations, deltas, batch_size):
        """Updates the weights and biases. """
        # print("act shapes:")
        # print([a.shape for a in activations])

        # print("delt shapes:")
        # print([d.shape for d in deltas])


        for i in range(len(activations)-1): #i counts the layers (both hidden and not hidden)
            # print(activations[i].T.shape)
            # print(deltas[i].shape) # off by one dimension

            w_grad = 1 / np.shape(activations[i])[0] * np.matmul(activations[i].T, deltas[i])
            w_grad = 1 * np.matmul(activations[i].T, deltas[i])
            
            # print("act shape {}".format(activations[i].shape))
            # print("delt shape {}".format(deltas[i].shape))
            # print("w_grad shape {}".format(w_grad.shape))
            # print("weight shape {}".format(self.weight_[i].shape))
            
            if i != len(activations)-2: # for all but the last layer
                w_grad = w_grad[:,1:] # remove bias unit from gradient
            # print(w_grad.shape)
            # w_grad, b_grad = self._compute_gradient(activations[i], deltas[i], batch_size)
            # b_grad = np.zeros(np.shape(self.bias_[i])) #not sure what to do with this

            # print(self.weight_[i].shape)
            # print(w_grad.shape)
            # update weights based on learning rate and gradient
            self.weight_[i] += - self.learning_rate * w_grad.T
            # self.bias_[i] += - self.learning_rate * b_grad

    def fit(self, X, y, metadata, verbose=False, F1=False):
        """Fit a multi-layer neural network.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_classes]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        self.metadata = metadata
        self.classes_ = self.metadata["features"][-1][1]
        
        self.std_constants_ = self._set_standardization(X)
        
        # standardize numeric features and make one-hot encoding for categorical features in X and y.
        X_std = self._standardize(X)
        self.X_train = self._categorical_to_onehot(X_std) 
        # self.y_train = label_binarize(y.to_numpy(), classes_=self.classes_)
        # self.y_train = y.to_numpy().reshape((len(y),1))
        self.y_train = 1*np.equal(y, self.classes_[1]).to_numpy().reshape((len(y),1)) # 1 if second class, 0 if first class
        

        # print(self.y_train)
        
        # self.classes_ = np.unique(y)
        n_instances, n_features = self.X_train.shape
        hidden_dim = list(self.hidden_dim)
        self.n_outputs_ = len(self.classes_)
        layer_dim = ([n_features] + hidden_dim + [self.n_outputs_])
        
        # initialize n_layers_, weights_, and bias_ in self
        self._initialize_weights(layer_dim)
        
        # print(self.weight_[0])
        # print(self.weight_[0].shape)
        # print(self.weight_[1])
        # print(self.weight_[1].shape)
        # # print(self.X_train[0,:])
        # print([w.shape for w in self.weight_])

        batch_size = min(self.batch_size, n_instances)
        for j in range(self.num_epoch):

            # break data into batches for minibatch gradient descent
            X_batches = []
            Y_batches = []
            for b in range(0, n_instances, self.batch_size):
                # TODO: see if there's a better way to do this 
                X_batch = np.vsplit(self.X_train,[b,b+self.batch_size])[1]
                Y_batch = np.vsplit(self.y_train,[b,b+self.batch_size])[1] 
                X_batches.append(X_batch)
                Y_batches.append(Y_batch)

            # in each batch, feed forward data, perform backprop, and update parameters
            cross_entropy_error = 0
            n_correctly_classified = 0
            for batch_x, batch_y in zip(X_batches, Y_batches):
                act = self._forward_pass([batch_x])
                d_in = np.ones(len(act)-1, dtype=object)
                delt = self._backprop(batch_y, act, d_in) 
                # if j==0 and np.all(batch_x == X_batches[0]):
                #     print(act[1])
                #     print(act[2])
                    
                
                # record contribution to CE error and correct classifications in each batch
                conf_pred = act[self.n_layers_ - 1]
                y_pred = conf_pred > 0.5
                cross_entropy_error += cross_entropy(batch_y.flatten(), conf_pred.flatten())
                n_correctly_classified += sum(y_pred.flatten() == batch_y.flatten())

                self._update_params(act, delt, batch_size)
                
            # print out results
            if verbose:
                n_misclassified = n_instances - n_correctly_classified
                print("{0} {1:.12f} {2} {3}".format(j+1,cross_entropy_error, n_correctly_classified, n_misclassified))

            # For getting F1 score at each epoch
            # if F1:
            #     y_true_01 = 1*(y_true != self.classes_[0]) #  0 for first class and 1 for second class
            #     F1_score(y_true_01, y_pred)
        
            #     F1_test = predict()




    
    def predict(self, X, y_true=None, confidence=False, verbose=False, F1=False):
        """Predict target values for instances in X.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Input data.
        y_true : array, shape = [n_instance, 1]
            True values of predctions, needed only for printing if verbose=True
        confidence: bool, default False
            If true, return confidence of prediction==0 (first class) instead of actual prediction
            

        Returns
        -------
        y_pred : array, shape = [n_instances,]
            Predicted target value for instances.
        """
        X = self._standardize(X)
        X = self._categorical_to_onehot(X) 
        # self.y_train = label_binarize(y.to_numpy())

        # makes X into a list 
        activations = [X]
        activations = self._forward_pass(activations)

        conf_pred = activations[self.n_layers_ -1]
        y_pred = conf_pred > 0.5
        y_pred = y_pred.astype(int)
        y_true_01 = 1*(y_true != self.classes_[0]) #  0 for first class and 1 for second class
        
        if verbose:
            for i in range(len(y_pred)):
                print("{0:.12f} {1} {2}".format(conf_pred[i,0], y_pred[i,0], y_true_01[i]))
            
            n_correctly_classified = sum(np.array(y_pred).flatten() == np.array(y_true_01).flatten())
            n_misclassified = sum(np.array(y_pred).flatten() != np.array(y_true_01).flatten())
            print("{} {}".format(n_correctly_classified, n_misclassified))    

            print("{0:.12f}".format(F1_score(y_true_01, y_pred)))
                
        if confidence:
            return conf_pred
        if F1:
            return F1_score(y_true_01, y_pred)
        else: 
            return y_pred





if __name__ == "__main__":
    # <learning-rate> <#epochs> <train-set-file> <test-set-file>
    if len(sys.argv) == 1:
        learning_rate = 0.01
        num_epochs = 10
        train = "datasets/banknote_train.json"
        test = "datasets/banknote_test.json"
    else:
        # learning_rate = str(sys.argv[1])
        # num_epochs = str(sys.argv[2])
        # train = str(sys.argv[3])
        # test = str(sys.argv[4])
        tmp, learning_rate, num_epochs, train, test = sys.argv
        learning_rate = float(learning_rate)
        num_epochs = int(num_epochs)
        
    # parse the json file for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)

    nn = NeuralNet(learning_rate=learning_rate, num_epoch=num_epochs,hidden_dim=(5,))

    nn.fit(X_train, y_train, meta_train, verbose=True)
    y_pred = nn.predict(X_test, y_true=y_test, verbose=True)
      
    