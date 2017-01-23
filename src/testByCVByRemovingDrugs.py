import sys
import random
import operator
import copy
from collections import OrderedDict
import math
import learnmodel

def addLinkToNetwork(net, source, target):
	if net.has_key(source):
		if target not in net[source]:  
			net[source].append(target)
	else:
		net[source] =[target]

def getNumberOfLinks(net):
	n=0
	for item in net.items():
		s = item[0]
		neighs = item[1]
		n+=len(neighs)
	return n 	

def printLinks(net, f):
	for item in net.items():
		s= item[0]
		neighs=item[1]
		for t in neighs:
			f.write(str(s)+"\t"+str(t)+"\t1\n")

def removeDrugs(wholenet,drugsTobeRemoved):
	trainset=dict()
	testset=dict()
	for dr in wholenet:
		if dr in drugsTobeRemoved:
			testset[dr]=wholenet[dr]
		else:
			trainset[dr]=wholenet[dr]

	return (trainset,testset)


if __name__== '__main__':
	if sys.argv is None or len(sys.argv) is not 2:
		print "Usage : python linkPrediction.py in_file "
		exit()
	
	wholenet = dict()
	remainnet = dict()

	knownDrugDis={}
	negDrugDis={}
	featuremap={}

	infile=file(sys.argv[1])
	header=infile.next()
	for line in infile:
		line=line.strip().split("\t")
		drug=line[0]
		disease=line[1]
		featuremap[(drug,disease)]="\t".join(line[2:])
		if int(line[-1])==1:
			knownDrugDis[drug,disease]=1
		else:
			negDrugDis[drug,disease]=1


	for indi in knownDrugDis:
		drug=indi[0]
		dis =indi[1]
		addLinkToNetwork(wholenet,drug,dis)


	n=10
	avgPrec =0
	avgAUC=0

	allNodes=list(wholenet.keys())
	numNodes=len(allNodes)
	random.shuffle(allNodes)
	

	n=10
	filename = "goldstd"
	classScoring = ['Accuracy', 'AUC']
	avgCvScores = {
		'Logistic-Regression': dict.fromkeys(classScoring,0),
		'Random-Forest-Class' : dict.fromkeys(classScoring,0),
		'KNN Class' : dict.fromkeys(classScoring,0),
	##'SVM' : dict.fromkeys(classScoring),
		'GradientBoostingClassifier' : dict.fromkeys(classScoring,0)
	}

	for i in xrange(n):
		start=i*numNodes/n
		end=(i+1)*numNodes/n
		drugsTobeRemoved=allNodes[start:end]
		(trainset,testset)=removeDrugs(wholenet,drugsTobeRemoved)
		#print "trainnet :", getNumberOfLinks(trainset)		
		#print "testnet :", getNumberOfLinks(testset)

		trainLinks=dict()
		trainNegLinks=dict()
		for dr in trainset:
			for di in trainset[dr]:
				trainLinks[(dr,di)]=1

		#print trainLinks

		testLinks=dict()
		testNegLinks=dict()

		for dr in testset:
			for di in testset[dr]:
					testLinks[(dr,di)]=1

		randomNegLinks= [ pair for pair in random.sample(negDrugDis,  2*(len(trainLinks)+len(testLinks))) ]
		trainNegLinks=randomNegLinks[0:len(trainLinks)*2-1]
		testNegLinks=randomNegLinks[len(trainLinks)*2:]
		
		trainingDataPath=filename+"-train"+str(i)+".txt"
		testDataPath=filename+"-test"+str(i)+".txt"
		train = open(trainingDataPath,"w")
		test = open(testDataPath,"w")  
		
		train.write(header)

		test.write(header)

		
		for (dr,di) in featuremap:
			#print dr,di
			#feat="\t".join(featuremap[(dr,di)])
			feat=featuremap[(dr,di)]
			if (dr,di) in trainLinks or (dr,di) in trainNegLinks:
				train.write(dr+"\t"+di+"\t"+feat+"\n")
			elif (dr,di) in testLinks or (dr,di) in testNegLinks:
				test.write(dr+"\t"+di+"\t"+feat+"\n")
		train.close()
		test.close()
		cvScores=learnmodel.runAllModels(trainingDataPath,testDataPath)
		for modelName in cvScores:
			print "Fold-"+str(i)+"\t"+modelName+"\t"+str(cvScores[modelName][classScoring[0]])+"\t"+str(cvScores[modelName][classScoring[1]])
			avgCvScores[modelName][classScoring[0]] += cvScores[modelName][classScoring[0]]
			avgCvScores[modelName][classScoring[1]] += cvScores[modelName][classScoring[1]]

	#print avgCvScores
	for modelName in avgCvScores:
		print "Avg. of Folds\t"+modelName+"\t"+str(avgCvScores[modelName][classScoring[0]]/n)+"\t"+str(avgCvScores[modelName][classScoring[1]]/n)
		#print avgCvScores[modelName][classScoring[0]]/n,"\t",avgCvScores[modelName][classScoring[1]]/n
