import json
import pandas as pd
import numpy as np

def parse_json(file_path, method="Pandas"):
        """
        Takes file path and parses json into data and metadata
        """
        with open(file_path, "r") as read_file:
            dataset = json.load(read_file)
        metadata = dataset["metadata"]
 
        """ Pandas """
        if method == "Pandas":
            col_names = [i[0] for i in dataset.get("metadata").get("features")]
            data = pd.DataFrame(dataset.get("data"), columns = col_names)   
            X = data.ix[:,:-1] # all but last column
            y = data.ix[:,-1] # just last column

        elif method == "Numpy":
            dataset = json.load(open(file_path,"r"))
            data = np.array(dataset['data'])
            y = data[:,-1]
            X = data[:,:-1]
            # meta = train['metadata']['features'] 
            metadata = dataset['metadata']
        else:
            print("please specify a valid method")

        return X, y, metadata
