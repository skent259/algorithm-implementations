"""
Given two files, check whether the output matches on each row, and print the row #, and relevant output that doesn't match
"""
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        file1 = pd.read_csv("output/tic-tac-toe_n.txt", sep=",", header=None)
        file2 = pd.read_csv("reference_output/tic-tac-toe_n.txt", sep=",", header=None)
        # file1 = pd.read_csv("output/output_knn_classifier_10_digits.txt", sep=",", header=None)
        # file2 = pd.read_csv("reference_output/output_knn_classifier_10_digits.txt", sep=",", header=None)
    elif len(sys.argv) != 3:
        print("usage: check_output_match.py <FILE 1> <FILE 2>")
    else:
        file1 = pd.read_csv(str(sys.argv[1]), sep=",", header=None)
        file2 = pd.read_csv(str(sys.argv[2]), sep=",", header=None)
    
    diff = file1 != file2
    to_check = diff.sum(axis=1) > 0

    out = pd.concat([file1.ix[to_check,:], file2.ix[to_check,:]], axis=1)
    print(out)
    