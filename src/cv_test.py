import numpy
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score,accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn import preprocessing
import crossvalid
from sklearn.cross_validation import cross_val_score

import argparse
import random
import csv

import pandas as pd


def createFeatureMat(pairs, classes, drug_df, disease_df, featureMatfile=None):
    totalNumFeatures=drug_df.shape[1] + disease_df.shape[1]-2

    #print totalNumFeatures
    drug_features = drug_df.columns.difference( ['Drug'] )
    disease_features = disease_df.columns.difference( ['Disease'])
    featureMatrix = numpy.empty((0,totalNumFeatures), int)
    for pair,cls in zip(pairs,classes):
	(dr,di)=pair
        #print pair,cls
	values1 = drug_df.loc[drug_df['Drug'] == dr][drug_features].values
	values2 = disease_df.loc[disease_df['Disease']==di][disease_features].values
	featureArray =numpy.append(values1,values2 )
        #print len(featureArray)
        #print len(featureArray),totalNumFeatures
        featureMatrix=numpy.vstack([featureMatrix, featureArray])
    return featureMatrix

def encodeLabels(train_df):
     from sklearn import preprocessing
     le = preprocessing.LabelEncoder()
     le.fit(train_df['Drug'])
     train_df['Drug']=le.transform(train_df['Drug'])
     le.fit(train_df['Disease'])
     train_df['Disease']=le.transform(train_df['Disease'])
     return train_df

def runModel( pairs, classes,  drug_df, disease_df , cv, n_subset, n_proportion, n_fold, model_type, model_fun, features, disjoint_cv, n_seed, n_setsel, verbose=True, output_f=None):
    clf= crossvalid.get_classification_model(model_type, model_fun, n_seed)
    all_auc = []
    all_auprc = []
    all_fs = []
    le_drug = preprocessing.LabelEncoder()
    le_dis = preprocessing.LabelEncoder()
    le_drug.fit(pairs[:,0])
    le_dis.fit(pairs[:,1])

    for i, (train, test) in enumerate(cv):
        print i
        file_name = None # for saving results
        pairs_train = pairs[train]
        classes_train = classes[train]
        pairs_test = pairs[test]
        classes_test = classes[test]
        print len(pairs_train), len(pairs_test), len(pairs)
        
        #X = createFeatureMat(pairs_train, classes_train, drug_df, disease_df)
        #y = numpy.array(classes_train)
	pairs_train_df = pd.DataFrame( zip(pairs[train,0],pairs[train,1],classes[train]),columns=['Drug','Disease','Class'])
        train_df=pd.merge( pd.merge(drug_df,pairs_train_df, on='Drug'),disease_df,on='Disease')

	#train_df= encodeLabels(train_df)
     	train_df['Drug']=le_drug.transform(train_df['Drug'])
     	train_df['Disease']=le_dis.transform(train_df['Disease'])
        #features_cols= train_df.columns.difference(['Drug','Disease','Class'])
        features_cols= train_df.columns.difference(['Class'])
	X=train_df[features_cols].values
        y=train_df['Class'].values.ravel()

        #X_new = createFeatureMat(pairs_test, classes_test, drug_df, disease_df)
        #y_new = numpy.array(classes_test)
	
	pairs_test_df = pd.DataFrame( zip(pairs[test,0],pairs[test,1],classes[test]),columns=['Drug','Disease','Class'])
        test_df=pd.merge( pd.merge(drug_df,pairs_test_df, on='Drug'),disease_df,on='Disease')

     	test_df['Drug']=le_drug.transform(test_df['Drug'])
     	test_df['Disease']=le_dis.transform(test_df['Disease'])
        #features_cols= test_df.columns.difference(['Drug','Disease','Class'])
        features_cols= test_df.columns.difference(['Class'])
        X_new=test_df[features_cols].values
        y_new=test_df['Class'].values.ravel()


        probas_ = clf.fit(X, y).predict_proba(X_new)
        y_pred = clf.predict(X_new)
        tn, fp, fn, tp = confusion_matrix(y_new, y_pred).ravel()
        precision = float(tp)/(tp+fp)
        recall = float(tp)/(tp+fn)
        fs=100*float(2*precision*recall/(precision+recall))
	print "number of features",X.shape[1]
        print "True negatives:", tn, "False positives:", fp,"False negatives:", fn, "True positives:",tp
        print "Precision:", precision, "Recall:",recall,"Specifity:",float(tn)/(tn+fp)
        #print "F-Measure",fs
        fpr, tpr, thresholds = roc_curve(y_new, probas_[:, 1])
        roc_auc = 100*auc(fpr, tpr)
	#roc_auc = 100*roc_auc_score(y_new, y_pred)
        all_auc.append(roc_auc)
        prc_auc = 100*average_precision_score(y_new, probas_[:, 1])
        all_auprc.append(prc_auc)
        all_fs.append(fs)
    	#print("F1-Meaure",f1_score(y_new, y_pred, average="binary"))
    	#print("Precision",precision_score(y_new, y_pred, average="binary"))
    	#print("Recall",recall_score(y_new, y_pred, average="binary"))
	print "train positive set:",len(y[y==1])," negative set:",len(y[y==0])
        print "test positive set:",len(y_new[y_new==1])," negative set:",len(y_new[y_new==0])
        #print prc_auc
        if verbose:
            print "Fold:", i+1, "# train:", len(pairs_train), "# test:", len(pairs_test), "AUC: %.1f" % roc_auc, "AUPRC: %.1f" % prc_auc, "FScore: %.1f" % fs
	del X
	del X_new
	del train_df
	del test_df
    print numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc)
    if output_f is not None:
        #output_f.write("n_fold\tn_proportion\tn_setsel\tmodel type\tfeatures\tdisjoint\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\n")
        output_f.write("%d\t%d\t%d\t%s\t%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_setsel, model_type,  "|".join(features), disjoint_cv, numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc), numpy.mean(all_fs), numpy.std(all_fs)))
    return numpy.mean(all_auc), numpy.mean(all_auprc)



def getData(goldindfile, drugfeatfiles, diseasefeatfiles, selectedFeatures=None):
    if selectedFeatures != None:
	selectedFeatures += ['Drug','Disease']

    gold_df= pd.read_csv(goldindfile, delimiter='\t')

    drugs=gold_df.Drug.unique()
    diseases=gold_df.Disease.unique()

    for i,featureFilename in enumerate(drugfeatfiles):
        temp=pd.read_csv(featureFilename, delimiter='\t')
        if i != 0:
            drug_df=drug_df.merge(temp,on='Drug')
            #drug_df=drug_df.merge(temp,how='outer',on='Drug')
        else:
            drug_df =temp

    #drug_df.fillna(0,inplace=True)

    if selectedFeatures != None:
    	drug_feature_names = drug_df.columns.intersection(selectedFeatures)
    	drug_df=drug_df[drug_feature_names]

    for i,featureFilename in enumerate(diseasefeatfiles):
        temp=pd.read_csv(featureFilename, delimiter='\t')
        if i != 0:
            disease_df=disease_df.merge(temp,on='Disease')
        else:
            disease_df =temp

    if selectedFeatures != None:
    	disease_feature_names = disease_df.columns.intersection(selectedFeatures)
    	disease_df=disease_df[disease_feature_names]

    print "number of drugs ",len(drug_df)
    print "number of diseases ",len( disease_df)
    commonDrugs=set(drug_df['Drug'].unique()).intersection(set(drugs))
    commonDiseases=set(disease_df['Disease'].unique()).intersection(set(diseases))

    gold_df=gold_df.loc[gold_df['Drug'].isin(commonDrugs) & gold_df['Disease'].isin(commonDiseases) ] 
    drug_df=drug_df.loc[drug_df['Drug'].isin(gold_df.Drug.unique())]
    disease_df=disease_df.loc[disease_df['Disease'].isin(gold_df.Disease.unique())]
    print "#drugs in gold ",len( drugs)
    print "#diseases in gold ",len( diseases)
    print "Used indications ",len(gold_df)
       


    return gold_df, drug_df, disease_df


if __name__ =="__main__":

        parser =argparse.ArgumentParser()
        parser.add_argument('-g', required=True, dest='goldindications', help='enter path to file for drug indication gold standard ')
        parser.add_argument('-m', required=True, dest='modelfile', help='enter path to file for trained sklearn classification model ')
	parser.add_argument('-disjoint', required=True, dest='disjoint', help='enter disjoint [0,1,2]')
        parser.add_argument('-o', required=True, dest='output', help='enter path to output file for model')
	parser.add_argument('-p', required=True, dest='proportion', help='enter number of proportion')
        parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
        parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')

        args= parser.parse_args()

        goldindfile=args.goldindications
        model_type=args.modelfile
	disjoint=int(args.disjoint)
        output_file_name=args.output
        drugfeatfiles=args.drugfeat
        diseasefeatfiles=args.diseasefeat
	n_proportion = int(args.proportion)
	#Get parameters
	n_seed = 205
	#random.seed(n_seed) # for reproducibility
	n_subset =-1



	#features=drug_features+disease_features
	output_file=open( output_file_name,'a')



	#fs=open("../data/importantFeatures.txt")
	#selectedFeatures =[]
	#for l in csv.reader(fs):
	#	selectedFeatures.append(l[0])
	
	selectedFeatures =None
	gold_df, drug_df, disease_df = getData(goldindfile, drugfeatfiles, diseasefeatfiles, selectedFeatures)
	
	features=[ fn[fn.index('-')+1:fn.index('.txt')] for fn in drugfeatfiles+diseasefeatfiles]

	
	drugDiseaseKnown = set([tuple(x) for x in  gold_df[['Drug','Disease']].values])

    	commonDrugs=drug_df['Drug'].unique()
    	commonDiseases=disease_df['Disease'].unique()
	pairs=[]
	classes=[]
	print "commonDiseases",len(commonDiseases)
	print "commonDrugs",len(commonDrugs)
        for dr in commonDrugs:
            for di in commonDiseases:
                if (dr,di)  in drugDiseaseKnown:
                    cls=1
                else:
                    cls=0
                pairs.append((dr,di))
                classes.append(cls)
	
	n_run = 10
	n_seed = 205
	n_fold =10
	model_fun=None
	n_subset=-1

	values = []
	values2 = []
	
	output_file.write("n_fold\tn_proportion\tn_setsel\tmodel type\tfeatures\tdisjoint\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\tf-score.mean\tf-score.sd\n")
	for i in xrange(n_run):
                if n_seed is not None:
                        n_seed += i
                        random.seed(n_seed)
                        numpy.random.seed(n_seed)
                pairs_, classes_, cv = crossvalid.balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset, disjoint, n_seed )
                roc_auc,aupr = runModel( pairs_, classes_, drug_df, disease_df, cv, n_subset, n_proportion, n_fold, model_type, model_fun, features, disjoint, n_seed, 1, verbose=True, output_f=output_file)
                values.append(roc_auc)
                values2.append(aupr)

        print "AUC over runs: %.1f (+/-%.1f):" % (numpy.mean(values), numpy.std(values))

	
	 
