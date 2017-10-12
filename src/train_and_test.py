import numpy
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
import crossvalid

import argparse
import random
import csv

import pandas as pd

def getAbsoluteTrain(pairs, classes, n_proportion):
   
    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    idx_false_list = []
    idx_true_list = []
    for idx, (pair, class_) in enumerate(zip(pairs, classes)):
        if class_ == 0:
            idx_false_list.append(idx)
        else:
            idx_true_list.append(idx)

    if n_proportion >= 1:
            indicies_train_negative = random.sample(idx_false_list, n_proportion * len(idx_true_list))
    else:
            indicies_train_negative = idx_false_list
    indices_train = idx_true_list + indicies_train_negative
    return pairs,  classes, indices_train

def createFeatureMat(pairs, classes, drug_df, disease_df, featureMatfile=None):
    totalNumFeatures=drug_df.shape[1] + disease_df.shape[1]-2

    #print totalNumFeatures
    drug_features = drug_df.columns.difference( ['Drug'] )
    disease_features = disease_df.columns.difference( ['Disease'])
    featureMatrix = numpy.empty((0,totalNumFeatures), int)
    for pair,cls in zip(pairs,classes):
        (dr,di)=pair
        values1 = drug_df.loc[drug_df['Drug'] == dr][drug_features].values
        values2 = disease_df.loc[disease_df['Disease']==di][disease_features].values
        featureArray =numpy.append(values1,values2 )
        #print len(featureArray)
        #print len(featureArray),totalNumFeatures
        featureMatrix=numpy.vstack([featureMatrix, featureArray])
    return featureMatrix

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
        else:
            drug_df =temp


    if selectedFeatures != None:
    	drug_feature_names = drug_df.columns.intersection(selectedFeatures)
    	drug_df=drug_df[drug_feature_names]

    diseaseFeatureNames={}
    diseaseFeatures={}
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
        parser.add_argument('-o', required=True, dest='output', help='enter path to output file for model')
	parser.add_argument('-p', required=True, dest='proportion', help='enter number of proportion')
	parser.add_argument('-s', required=True, dest='seed', help='enter seed number')
        parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
        parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')

        args= parser.parse_args()

        goldindfile=args.goldindications
        model_type=args.modelfile
	#model_load_file=args.modelfile
        output_file_name=args.output
        drugfeatfiles=args.drugfeat
        diseasefeatfiles=args.diseasefeat
	n_proportion = int(args.proportion)
	#Get parameters
	n_seed = int(args.seed)
	#random.seed(n_seed) # for reproducibility
	n_subset =-1



	#features=drug_features+disease_features
	output_file=open( output_file_name,'w')

        #fs=open("../data/importantFeatures.txt")
        #selectedFeatures =[]
        #for l in csv.reader(fs):
        #        selectedFeatures.append(l[0])
	selectedFeatures = None
	gold_df, drug_df, disease_df = getData(goldindfile, drugfeatfiles, diseasefeatfiles, selectedFeatures)
	
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

	random.seed(n_seed)
	numpy.random.seed(n_seed)

	pairs,  classes, train_indicies  = getAbsoluteTrain(pairs, classes, n_proportion)
	pairs_train_df = pd.DataFrame( zip(pairs[train_indicies,0],pairs[train_indicies,1],classes[train_indicies]),columns=['Drug','Disease','Class'])
	
	train_df=pd.merge( pd.merge(drug_df,pairs_train_df, on='Drug'),disease_df,on='Disease')
	
	features= train_df.columns.difference(['Drug','Disease','Class'])
	
	print "train #",len(train_df)
	X=train_df[features].values
	y=train_df['Class'].values.ravel()

	model_fun=None
		
	clf= crossvalid.get_classification_model(model_type, model_fun, n_seed)
	#clfx= crossvalid.get_classification_model(model_type, model_fun, n_seed)
	#sfk=cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True, random_state=n_seed)
	#scores=  cross_val_score(clfx, X, y, scoring='roc_auc', cv=sfk)
	#print scores
	#print numpy.mean(scores)
	
	clf.fit(X,y)
        for i,pair in enumerate(pairs):
                c,p=pair
                if classes[i] == 0 :# and c=='DB00234': # Reboxetine
                        test_indicies.append(i)

        pairs_test=pairs[test_indicies]
        classes_test=classes[test_indicies]
	
	pairs_test_df = pd.DataFrame( zip(pairs[test_indicies,0],pairs[test_indicies,1],classes[test_indicies]),columns=['Drug','Disease','Class'])
	
	test_df=pd.merge( pd.merge(drug_df,pairs_test_df, on='Drug'),disease_df,on='Disease')

	X_new = test_df[features].values

	probs= clf.predict_proba(X_new)
	print "#features",X.shape[1]
        print "test samples #:",len(X_new)
	scores = zip(pairs_test[:,0],pairs_test[:,1],probs[:,1])
	scores.sort(key=lambda tup: tup[2],reverse=True)
	output_file.write('Drug\tDisease\tProb\n')
	for (drug,disease,prob) in scores:    	
        	output_file.write( drug +'\t'+ disease+'\t'+str(prob)+'\n')

	
	 
