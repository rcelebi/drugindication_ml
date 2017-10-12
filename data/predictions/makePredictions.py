
import sys
import scipy.stats
import csv

def pearsonr(ranks1,ranks2):
	(rho,p)=scipy.stats.pearsonr(ranks1,ranks2)
	return rho

if __name__ =="__main__":


	indi_label={}
	drug_label={}
	
	for line in file('umls2disease.tab'):
		line=line.strip().replace('"','').split('\t')
		indi_label[line[0]]=line[1]

	for line in file('drugbank-drug-names.tab'):
		line=line.strip().replace('"','').split('\t')
		#print line
		drug_label[line[0]]=line[1]	

	oldindifile=file(sys.argv[1])
	oldindifile.next()
	indi_old={}
	for line in csv.reader(oldindifile,delimiter='\t'):
		indi_old[(line[0],line[1])]=float(line[2])
	 
	indi_old = sorted(indi_old.items(), key=lambda x:x[1], reverse = True)
	for ddi, val in (indi_old):
            dr,di =ddi
	    if val < 0.0 :
		continue
	    if dr in drug_label and di in indi_label:
	    	#if drug_label[dr] == 'Propranolol':
		print drug_label[dr], indi_label[di], val
	      #print ddi,indi_old[ddi],indi_new[ddi]
