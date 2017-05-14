import numpy
from sklearn import cross_validation
import crossvalid

import argparse
import random

if __name__ =="__main__":

        parser =argparse.ArgumentParser()
        parser.add_argument('-g', required=True, dest='goldindications', help='enter path to file for drug indication gold standard ')
        parser.add_argument('-m', required=True, dest='modelfile', help='enter path to file for trained sklearn classification model ')
        parser.add_argument('-o', required=True, dest='output', help='enter path to output file fro results')
        parser.add_argument('-n', required=False, dest='negativeset', help='enter negative set selecetion (1: random pairs from diseases with at least one indication 2: random pairs from all diseases) ')
        parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
        parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')

        args= parser.parse_args()

        goldindfile=args.goldindications
        model_load_file=args.modelfile
        output_file_name=args.output
        drugfeatfiles=args.drugfeat
        diseasefeatfiles=args.diseasefeat
        negativeSelectionScheme= int(args.negativeset)
	#Get parameters
	n_seed = 205
	#random.seed(n_seed) # for reproducibility
	n_subset =-1


	drugDiseaseKnown,drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames  = crossvalid.getData(goldindfile, drugfeatfiles, diseasefeatfiles)

	pairs,classes=crossvalid.getAllPossiblePairs(drugDiseaseKnown, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, negativeSelectionScheme)

	#features=drug_features+disease_features
	features=[ fn[fn.index('-')+1:fn.index('.txt')] for fn in drugfeatfiles+diseasefeatfiles]
	output_file=open( output_file_name,'a')
        drugs=['DB00571','DB00745','DB00313','DB00563']
	for drug in drugs:
        	crossvalid.make_predictions_for_drug( pairs, classes, drug, drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames, n_subset, model_load_file, output_file)
	
	 
