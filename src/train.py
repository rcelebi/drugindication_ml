import numpy
from sklearn import cross_validation
import crossvalid

import argparse
import random

if __name__ =="__main__":

        parser =argparse.ArgumentParser()
        parser.add_argument('-g', required=True, dest='goldindications', help='enter path to file for drug indication gold standard ')
        parser.add_argument('-m', required=True, dest='modeltype', help='enter machine learning model [logistic, gbc]')
        parser.add_argument('-disjoint', required=True, dest='disjoint', help='enter disjoint [0,1,2]')
        parser.add_argument('-p', required=True, dest='proportion', help='enter number of proportion')
        parser.add_argument('-o', required=True, dest='output', help='enter path to output file fro results')
        parser.add_argument('-n', required=False, dest='negativeset', help='enter negative set selecetion (1: random pairs from diseases with at least one indication 2: random pairs from all diseases) ')
        parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
        parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')

        args= parser.parse_args()

        goldindfile=args.goldindications
        model_type=args.modeltype
        disjoint=int(args.disjoint)
        nProportion = int(args.proportion)
        output_file_name=args.output
        drugfeatfiles=args.drugfeat
        diseasefeatfiles=args.diseasefeat
        negativeSelectionScheme= int(args.negativeset)
	#Get parameters
	n_seed = 205
	#random.seed(n_seed) # for reproducibility
	drug_features = ["chemical", "target"]
	disease_features = ["phenotyic", "ontology"]
	#model_type = "gbc" # ML model

	#output_file_name = "data/output/validation.dat" # file containing run parameters and corresponding AUC values
	#nProportion = 2 # proportion of negative instances compared to positives

	# whether the drugs in the drug-disease pairs of the cross-validation folds should be non-overlapping disjoint_cv = False 
	#disjoint=True
	n_subset=-1
	model_fun=None

	n_run = 10 # number of repetitions of cross-validation analysis
	n_fold = 1 # number of folds in cross-validation

	#negativeSelectionScheme=2

	#goldindfile = "data/input/predict-gold-standard-umls.txt"
	#drugfeatfiles = ["data/features/drugs-fingerprint.txt","data/features/drugs-targets.txt"]
	#diseasefeatfiles = ["data/features/diseases-hpo.txt","data/features/diseases-meddra.txt"]
	#diseasefeatfiles.remove("../data/features/diseases-meddra.txt")

	drugDiseaseKnown,drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames  = crossvalid.getData(goldindfile, drugfeatfiles, diseasefeatfiles)

	pairs,classes=crossvalid.getAllPossiblePairs(drugDiseaseKnown, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, negativeSelectionScheme)

	#features=drug_features+disease_features
	features=[ fn[fn.index('-')+1:fn.index('.txt')] for fn in drugfeatfiles+diseasefeatfiles]
	output_file=open( output_file_name,'a')
	values = []
	values2 = []
	output_file.write("n_fold\tn_proportion\tn_setsel\tmodel type\tfeatures\tdisjoint\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\n")
	
	#train_indicies=[1,2]
	train_indicies = crossvalid.get_absolute_train(pairs, classes, n_fold, nProportion, n_subset, disjoint, n_seed )
	
	clf = crossvalid.trainModel( pairs, classes, train_indicies, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, n_subset, nProportion, n_fold, model_type, model_fun, features, disjoint, n_seed, negativeSelectionScheme, output_f=output_file)
	from sklearn.externals import joblib
	joblib.dump(clf, '../data/models/logistic.pkl') 
