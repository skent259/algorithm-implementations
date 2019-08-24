import numpy as np

def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : array, shape = [n_instances]
        Actual target values.
    y_pred : array, shape = [n_instances]
        Predicted target values.

    Returns
    -------
    score : float
        Returns the fraction of correctly classified instances.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    score = y_true == y_pred
    return np.average(score)
