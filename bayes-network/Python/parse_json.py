import json
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
