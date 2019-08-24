from neural_network import NeuralNet
from parse_json import parse_json
import sys
import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # <learning-rate> <#epochs> <train-set-file> <test-set-file>
    if len(sys.argv) == 1:
        learning_rate = 0.05
        num_epochs = 20
        train = "datasets/heart_train.json"
        test = "datasets/heart_test.json"
    else:
        tmp, learning_rate, num_epochs, train, test = sys.argv
        learning_rate = float(learning_rate)
        num_epochs = int(num_epochs)
        
    # parse the json file for data
    X_train, y_train, meta_train = parse_json(train)
    X_test, y_test, meta_test = parse_json(test)

    nn = NeuralNet(learning_rate=learning_rate, num_epoch=num_epochs,hidden_dim=())

    nn.fit(X_train, y_train, meta_train, verbose=True)
    nn.predict(X_test, y_true=y_test, verbose=True)

      
    