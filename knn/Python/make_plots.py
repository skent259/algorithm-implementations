import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

if __name__ == "__main__":

    # 2. Plot of hyperparameter search
    file_path = "output/output_hyperparam_tune_20_digits.txt"
    ht_digits = pd.read_csv(file_path, sep=",", header=None, names=["k","accuracy"])
    n, m = ht_digits.shape
    ht_digits = ht_digits.ix[0:n-3,:]
    
    fig1 = plt.Figure()
    fig1 = ht_digits.plot(x="k",y="accuracy", title="KNN Hyperparameter Search (over k) on Digits Dataset", xticks=range(21))
    plt.ylabel("Accuracy (%)")
    fig1 = fig1.get_figure()
    fig1.savefig("./plots/tune_k.pdf")

    # 3. Plot of learning curve for multiple k
    ks = [3,5,10,15]
    file_paths = ["output/output_learning_curve_{}_digits.txt".format(k) for k in ks]
    # print(file_paths)
    dfs = [pd.read_csv(file_path, sep=",", header=None, names=["n_examp","accuracy"]) for file_path in file_paths]
    for (df,k) in zip(dfs,ks):
        df["k"] = k 
    lc = pd.concat(dfs)
    lc = lc.pivot(index='n_examp', columns='k', values='accuracy')

    fig2 = plt.Figure()
    fig2 = lc.plot(title="Learning Curve for KNN on Digits Dataset")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Number of training examples")
    fig2 = fig2.get_figure()
    fig2.savefig("./plots/learning_curve.pdf")

    # 4. Plot ROC curve 
    ks = [5,10,20,30]
    file_paths = ["output/output_roc_curve_{}_votes.txt".format(k) for k in ks]
    dfs = [pd.read_csv(file_path, sep=",", header=None, names=["FPR","TPR"]) for file_path in file_paths]
    
    fig3, ax = plt.subplots()
    for k, df in zip(ks,dfs):
        ax = df.plot(ax=ax, kind='line', x='FPR', y='TPR', label=k)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve for KNN on Votes Dataset")
    plt.legend(loc='best')
    fig3 = ax.get_figure()
    fig3.savefig("./plots/roc_curve.pdf")

