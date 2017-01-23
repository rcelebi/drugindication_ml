import sys
from sklearn import linear_model, cross_validation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn import svm
# from sklearn.neural_network import MLPClassifier # This is not available yet
# from sklearn.neural_network import MLPRegressor # Same as above
# from sklearn.cross_validation import cross_val_score
from sklearn.metrics import auc
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import pprint, argparse
import pandas as pd



def importData(dataPath):
    dataAll = np.genfromtxt(dataPath, skip_header= True, delimiter = '\t')
    #X = dataAll[:, range(0, (dataAll.shape[1] - 1))] # getting all features
    
    #X = dataAll[:, range(2, (dataAll.shape[1] - 1))] # getting all features expet the drug-disease
    #X = dataAll[:, range(6, (dataAll.shape[1] - 1))] # getting topological similarties
    X = dataAll[:, range(2, (dataAll.shape[1] - 1))] # getting just drug-disease similarities
    
    Y = dataAll[:, -1] # getting only the last column
    #print Y
    #YBinary = np.zeros(Y.shape)
    #YBinary[Y >= float(synThresh)] = 1
    return (X,Y)

##############################Input#################################
# trainingDataPath: Str path to training data. A header is expected.
# synThresh: For binary classification. Above this threshold is 
# synergistic and below is not.
# plotPath: Path to save jpg images of cv data plots. Optional 
# and not yet implemented.
##############################Output################################
# No return. Simply prints how well models are doing
##############################Description###########################
# Reads in the data in trainingDataPath and runs cv on a number of 
# different models to estimate the testing error. These estimates 
# are then printed
def runAllModels(trainingDataPath,testDataPath):
    pp = pprint.PrettyPrinter(indent=4)

    (X_train,y_train)=importData(trainingDataPath)
    (X_test,y_test)=importData(testDataPath)
    print "train",X_train.shape
    print "test ",X_test.shape
    #for i in YBinary:
    #    print i
    classifModels = {
        'Logistic-Regression': linear_model.LogisticRegression(),
        'Random-Forest-Class': RandomForestClassifier(max_features = 'auto', n_estimators = 100), # auto is sqrt(numFeats)
        'GradientBoostingClassifier' : GradientBoostingClassifier(n_estimators= 1000, max_depth= 5, random_state = 2, max_features=0.9),
        #'GradientBoostingClassifier' : GradientBoostingClassifier(),
        # n_estimators can be tweaked to find a better one
        'KNN Class' : KNeighborsClassifier(n_neighbors = 10, algorithm = 'auto', metric = 'minkowski'), # can choose different distance metrics too 
        # 'NN trries to figure out best alg when  'auto' is used. There are otheres we can specify though
        #'SVM' : svm.SVC(kernel = 'rbf') # This kernel looks to do the best
    }
    regressModels = {
       'Random-Forest-Reg' : RandomForestRegressor(max_features = 'auto', n_estimators = 10),
        'KNN Reg': KNeighborsRegressor(n_neighbors = 100, algorithm = 'auto'),# play around with n_neighbors
        'Linear': linear_model.LinearRegression(),
        'SVMR': svm.SVR(kernel = 'rbf'),
        'Lasso': linear_model.Lasso(alpha = 0.1)# This value of Alpha will not penalize features badly
        
    }
    classScoring = ['Accuracy', 'AUC']
    regScoring = ['Average Mean Squared Error', 'Average R^2']
    cvScores = {
     'Logistic-Regression': dict.fromkeys(classScoring),
        'Random-Forest-Class' : dict.fromkeys(classScoring),
        'KNN Class' : dict.fromkeys(classScoring),
        #'SVM' : dict.fromkeys(classScoring),
        'GradientBoostingClassifier' : dict.fromkeys(classScoring)
    }
    
    # Classification
    for idx, (modelName, model) in enumerate(classifModels.items()):
        model =model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
       

        cvScores[modelName][classScoring[0]] = metrics.accuracy_score(y_test, y_pred)
        cvScores[modelName][classScoring[1]] = metrics.roc_auc_score(y_test,y_pred)
            
    #pp.pprint(cvScores)
    #print "Classifier\t"+classScoring[0]+"\t"+classScoring[1]
    #for cv in cvScores:
    #    print cv+"\t"+str(cvScores[cv][classScoring[0]])+"\t"+str(cvScores[cv][classScoring[1]])

    return cvScores 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--trainingPath', help = 'Path to Training Data', required = True)
    parser.add_argument('-p2', '--testPath', help = 'Path to Test Data', required = True)
    parser.add_argument('-pl', '--plotPath', help = 'Path to save plots', required = False, default = None)
    args = vars(parser.parse_args())
    #print(args['synThresh'])
    runAllModels(trainingDataPath=args['trainingPath'], testDataPath=args['testPath'])


# Usage: python runAllModels.py -p pathToTrainingData -t synThresh(optional) -pl plotPath(optional
if __name__ == "__main__":
    main()
