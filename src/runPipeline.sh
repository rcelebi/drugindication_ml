export SPARQL_ENDPOINT=http://localhost:13065/sparql
# query drug target info and downlad in the input folder
curl -H "Accept: text/tab-separated-values" --data-urlencode query@drugbank-drug-target.sparql $SPARQL_ENDPOINT > drugbank-drug-target.tab
# create feature matrix for drug target 
python createTargetFeatureMatrix.py ../data/input/drugbank-drug-target.tab > ../data/features/drugs-targets.txt 
 
# query drug smiles info and downlad in the input folder
curl -H "Accept: text/tab-separated-values" --data-urlencode query@drugbank-drug-smiles.sparql $SPARQL_ENDPOINT > drugbank-drug-smiles.tab
# create feature matrix for drug fingerprint 
python createFingerprintFeatures.py ../data/input/drugbank-drug-smiles.tab > ../data/features/drugs-fingerprint.txt 

# disease hpo features
python createHpoFeatureMatrix.py ../data/input/sider-hpo-terms.tab > ../data/features/diseases-hpo.txt

#disease meddra features
curl -H "Accept: text/tab-separated-values" --data-urlencode query@sider-meddra-terms.sparql $SPARQL_ENDPOINT > sider-meddra-terms.tab
python createMeddraFeatures.py ../data/input/sider-meddra-terms.tab > ../data/features/diseases-meddra.txt

python createFullFeatures.py  -g ../data/input/predict-gold-standard-umls.txt -dr ../data/features/drugs-targets.txt ../data/features/drugs-fingerprint.txt -di ../data/features/diseases-meddra.txt ../data/features/diseases-hpo.txt -o ../data/output/predict-gold-standard-umls-drug-disease-features.txt

#ensemble all drug and disease feature for given gold standard drug indications (and size of 2*numIndications selected randomly - ( if a negativedisease set is given, negative set is seleceted randomly from diseases wheree no previous indications reported for)
python createFullFeaturesWithRandNegFromDisNoInd.py  -g ../data/input/predict-gold-standard-umls.txt -dr ../data/features/drugs-targets.txt ../data/features/drugs-fingerprint.txt -di ../data/features/diseases-meddra.txt ../data/features/diseases-hpo.txt -o ../data/output/predict-gold-standard-umls-drug-disease-features.txt -n ../data/input/diseases_not_in_drugcentral.txt


python createFullFeaturesWithRandNegFromDisNoInd.py  -g ../data/input/drugcentral-gold-standard-umls.txt -dr ../data/features/drugs-targets.txt ../data/features/drugs-fingerprint.txt -di ../data/features/diseases-meddra.txt ../data/features/diseases-hpo.txt -o ../data/output/drugcentral-gold-standard-umls-drug-disease-features.txt -n ../data/input/diseases_not_in_drugcentral.txt

# do 10-fold cross-validation, separate gold standard into train and test by removing 10% of drugs and theirs association
# train on training set and test on test set and report the performance of each fold
python testByCVByRemovingDrugs.py ../data/output/predict-gold-standard-umls-drug-disease-features.txt > ../data/output/results_predict_all_features.txt

python testByCVByRemovingDrugs.py ../data/input-data/drugcentral-gold-standard-umls-drug-disease-features.txt > results_drugcentral_all_features.txt
