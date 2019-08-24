
import sys
import json
import numpy as np
import pandas as pd
import math

def nnet(learningRate,nHidden,nEpoch,train,test):
    """
    Logistic Regression using stochastic gradient descent to minimize cross-entropy error

    Assumptions:
    - Solves binary classification problem
    - Input datasets are randomized already
    - Class (desired prediction) is last attribute in metadata and is named "class"

    Parameters
    ----------
    learning-rate : number for step size in stochastic gradient descent

    nEpochs : number of times to pass through training set
        Each epoch is one complete pass through the training set

    Data
    ----------
    train and test json files contain two parts, metadata and data
        metadata contains information in regards to feature name and type
        metadata is the same for train and test json files
        data are the train and test datasets

    Train data of 'm' instances and 'n' features

    Test data of 'M' instances and 'N' features
    """

    with open(train) as data_file1:
        trainData = json.load(data_file1)

    with open(test) as data_file2:
        testData = json.load(data_file2)


    ###################################################################
    ### Data Pre-processing

    # get data as pandas dataframes
    col_names = [i[0] for i in trainData.get("metadata").get("features")]
    trainpd = pd.DataFrame(trainData.get("data"), columns=col_names)
    trainX = trainpd.iloc[:, :-1]  # all but last column
    trainY = trainpd.iloc[:, -1]  # just last column

    testpd = pd.DataFrame(testData.get("data"), columns=col_names)
    testX = testpd.ix[:, :-1]  # all but last column
    testY = testpd.ix[:, -1]  # just last column

    meta = trainData['metadata']
    features = [i[0] for i in meta["features"]]
    # classes = meta["features"][-1][1]
    featureLevels = {i[0]: i[1] for i in meta["features"]}

    # change values of y to be 0 or 1 (second item is to be changed to 1)
    realy = trainY
    trainY = np.zeros(len(realy))
    trainY[np.where(realy == np.unique(realy)[1])] = 1

    realy = testY
    testY = np.zeros(len(realy))
    testY[np.where(realy == np.unique(realy)[1])] = 1

    actPos = testY.sum()  # number actual positive

    trainY = pd.DataFrame(trainY)
    testY = pd.DataFrame(testY)

    # determine which features are numeric and which are categorical
    # indices of numeric features
    inum = [i for i, v in enumerate([featureLevels[f] for f in features][0:trainX.shape[1]]) if v == 'numeric']
    # indices of categorical features
    icat = [i for i, v in enumerate([featureLevels[f] for f in features][0:trainX.shape[1]]) if v != 'numeric']

    # separate data set by feature type
    trainNum = trainX.iloc[:, inum]
    testNum = testX.iloc[:, inum]
    trainCat = trainX.iloc[:, icat]
    testCat = testX.iloc[:, icat]

    # standardize numeric data
    # if bool(inum) == True:
    #     trainstd = trainX.iloc[:,inum].std(axis=0, ddof=0)
    #     trainstd[trainstd == 0] = 1
    #     trainX.iloc[:,inum] = (trainX.iloc[:,inum] - trainX.iloc[:,inum].mean(axis=0)) / trainstd
    #     testX.iloc[:,inum] = (testX.iloc[:,inum] - trainX.iloc[:,inum].mean(axis=0)) / trainstd

    # old standardize numeric data code
    if trainNum.shape[1] != 0:
        trainstd = trainNum.std(axis=0, ddof=0)
        trainstd[trainstd == 0] = 1
        trainNorm = (trainNum - trainNum.mean(axis=0)) / trainstd
        testNorm = (testNum - trainNum.mean(axis=0)) / trainstd

    # one-hot encoding of categorical variables     HOW TO CODE IN PLACE?
    # for i in range(len(icat)):
    #     # since dropping old columns, need to always one-hot encode first column
    #     trainX.iloc[:,icat] = pd.concat([trainX.iloc[:,icat],
    #                                      pd.get_dummies(trainX.iloc[:,icat][trainX.iloc[:,icat].columns[0]],
    #                                                               prefix=trainX.iloc[:,icat].columns[0])],
    #                          axis=1).drop(trainX.iloc[:,icat].columns[0], axis=1)
    #     testX.iloc[:,icat] = pd.concat([testX.iloc[:,icat],
    #                                     pd.get_dummies(testX.iloc[:,icat][testX.iloc[:,icat].columns[0]],
    #                                                             prefix=testX.iloc[:,icat].columns[0])],
    #                         axis=1).drop(testX.iloc[:,icat].columns[0], axis=1)

    # old one-hot encoding of categorical variables
    for i in range(len(icat)):
        # since dropping old columns, need to always one-hot encode first column
        trainCat = pd.concat([trainCat,pd.get_dummies(trainCat[trainCat.columns[0]],prefix = trainCat.columns[0])],
                             axis=1).drop(trainCat.columns[0],axis=1)
        testCat = pd.concat([testCat, pd.get_dummies(testCat[testCat.columns[0]], prefix=testCat.columns[0])],
                             axis=1).drop(testCat.columns[0], axis=1)

    # recombine numerical and categorical variables
    Xtrain = pd.concat([trainNorm, trainCat], axis=1)
    Xtest = pd.concat([testNorm, testCat], axis=1)

    # add intercept (or bias) term
    Xtrain = pd.concat([pd.DataFrame(np.ones(Xtrain.shape[0])),Xtrain],axis=1)
    Xtest = pd.concat([pd.DataFrame(np.ones(Xtest.shape[0])), Xtest], axis=1)

    ###################################################################
    ### Functions

    # define functions used in algorithm
    def sigmoid(w, x):
        """
        Calculate sigmoid function given 1 dimensional (1 by n+1) arrays w and x
        Note: sigmoid function is output of the single layer in logistic regression

        :w: weight vector for instance d
        :x: instance d vector of features
        """

        net = np.dot(w, x)
        o = (1 + math.exp(-net)) ** (-1)
        return o

    def crossEntropyError(w, x, y):
        """
        Calculate cross entropy error given 1 dimensional (1 by n+1) arrays w and x and number y

        :w: weight vector for instance d
        :x: instance d vector of features
        :y: instance d class
        """
        o = sigmoid(w, x)
        error = -y * math.log(o) - (1 - y) * math.log(1 - o)
        return error

    def errorGradient(w, x, y):
        """
        Calculate gradient of error with respect to weight.
            For logistic regression, dE/dw = dE/do * do/dnet * dnet/dw = (o - y) * x
            where net = w0 + sum(wi * xi)
            o = (1 + exp(-net))^(-1)
            E = -y * ln(o) - (1 - y) * ln(1 - o)

        :w: weight vector for instance d
        :x: instance d vector of features
        :y: instance d class
        """

        o = sigmoid(w, x)
        dE = (o - y) * x
        return dE

    def updateWeights(eta, w, x, y):
        """
        Update weights.

        :learningRate:
        :w: weight vector for instance d
        :x: instance d vector of features
        :y: instance d class
        """

        dE = errorGradient(w, x, y)
        newWeight = w - eta * dE
        return newWeight

    ###################################################################
    ### Train Model

    # Algorithm to train single layer neural net using sigmoid funciton and cross entropy for all nodes

    # np.random.seed(0)
    # weights from input to hidden unit
    w_i_h = np.random.uniform(low=-0.01, high=0.01, size=( nHidden, Xtrain.shape[1]))
    # weights from hidden unit to output
    w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, nHidden + 1))
    w_h_o = w_h_o[0]
    # print(w_i_h)
    # print(w_i_h.shape)
    # print(w_h_o)
    # print(w_h_o.shape)
    epoch = 1  # initialize epoch counter
    while epoch <= nEpoch:
        sumErrorOut = 0
        correctClass = 0
        # one epoch
        # print(Xtrain.shape)
        for d in range(Xtrain.shape[0]):
            #print(d)
            o_h = pd.DataFrame(np.zeros([nHidden,1]))  # initialize hidden units
            for h in range(nHidden):
                #print(h)
                o_h.iloc[h,:] = sigmoid(w_i_h[h,:], Xtrain.iloc[d,:])  # is this correct indexing?

            o_h = pd.concat([pd.DataFrame(np.ones(1)), o_h], axis=0)
            o = sigmoid(w_h_o, o_h)
            # if d == 0 and epoch==1:
            #     print(o_h)
            #     print(o)
            if o >= 0.5:  # determine which class to predict
                predY = 1
            else:
                predY = 0
            if predY == trainY.iloc[d][0]:  # is prediction correct
                correctClass = correctClass + 1
            deltaOutput = (trainY.iloc[d][0] - o)
            errorOutput = crossEntropyError(w_h_o, o_h, int(trainY.iloc[d]))
            sumErrorOut = sumErrorOut + errorOutput
            errorHidden = deltaOutput * np.multiply(np.multiply(o_h, (1-o_h)), pd.DataFrame(w_h_o))
            newOutputWeights = w_h_o + np.array(learningRate * deltaOutput * o_h)[:,0]
            newHiddenWeights = w_i_h + learningRate * np.array(errorHidden.iloc[1:errorHidden.shape[0],:]) * np.array(Xtrain.iloc[d,:])
            w_h_o = newOutputWeights
            w_i_h = newHiddenWeights
            if d == 0 and epoch==1:
                print(w_h_o)
                print(w_i_h)

        missClass = Xtrain.shape[0] - correctClass
        # print("{0} {1:.12f} {2} {3}".format(epoch, sumErrorOut, correctClass, missClass))
        epoch = epoch + 1

    ###################################################################
    ### Test Model

    correctClass = 0
    tp = 0
    predPos = 0
    for d in range(Xtest.shape[0]):
        o_h = pd.DataFrame(np.zeros([nHidden, 1]))  # initialize hidden units
        for h in range(nHidden):
            # print(h)
            o_h.iloc[h, :] = sigmoid(w_i_h[h, :], Xtest.iloc[d, :])  # is this correct indexing?

        o_h = pd.concat([pd.DataFrame(np.ones(1)), o_h], axis=0)
        o = sigmoid(w_h_o, o_h)
        if o >= 0.5:  # determine which class to predict
            predY = 1
            predPos = predPos + 1
        else:
            predY = 0
        if predY == testY.iloc[d][0]:  # is prediction correct
            correctClass = correctClass + 1
            if predY == 1:
                tp = tp + 1  # positive class is 1
        # print("{0:.12f} {1} {2}".format(o, predY, int(testY.iloc[d][0])))
    missClass = testY.shape[0] - correctClass
    # print("{0} {1}".format(correctClass, missClass))

    recall = tp / actPos
    precision = tp / predPos
    f1 = (2 * recall * precision) / (precision + recall)
    print("{0:.12f}".format(f1))



# np.random.seed(0)
# nnet(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

# # heart
# # cross entropy error is off. Possibly due to data-preprocess code since bank and magic data sets work
# np.random.seed(0)
# nnet(0.05, 7, 20, 'homework3/heart_train.json', 'homework3/heart_test.json')

# # bank
np.random.seed(0)
nnet(0.01, 5, 10, 'datasets/banknote_train.json', 'datasets/banknote_test.json')
#
# magic
# np.random.seed(0)
# nnet(0.01, 10, 5, 'datasets/magic_train.json', 'datasets/magic_test.json')







