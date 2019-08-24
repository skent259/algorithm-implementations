from neural_network import NeuralNet
from parse_json import parse_json
import sys
import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
    learning_rate = 0.05
    hidden_units = 7
    num_epochs = 50
    train = "datasets/heart_train.json"
    test = "datasets/heart_test.json"
    
    # parse the json file for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)
    
    print("model,epoch,F1_train,F1_test")
    # Run logistic regression 
    for epoch in range(1,num_epochs+1):
        logistic = NeuralNet(learning_rate=learning_rate, num_epoch=epoch,hidden_dim=())
              
        logistic.fit(X_train, y_train, meta_train)
        F1_train = logistic.predict(X_train, y_true=y_train, F1=True)
        F1_test = logistic.predict(X_test, y_true=y_test, F1=True)
        print("logistic-0.05-heart,{},{},{}".format(epoch,F1_train,F1_test))

    # Run neural network
    for epoch in range(1,num_epochs+1):
        nnet = NeuralNet(learning_rate=learning_rate, num_epoch=epoch,hidden_dim=(hidden_units,))
              
        nnet.fit(X_train, y_train, meta_train)
        F1_train = nnet.predict(X_train, y_true=y_train, F1=True)
        F1_test = nnet.predict(X_test, y_true=y_test, F1=True)
        print("nnet-0.05-7-heart,{},{},{}".format(epoch,F1_train,F1_test))


    

      
    