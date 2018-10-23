#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: Workflow

inputs:
  drug_target: # Drug targets
    type: File
    format: tsv # Tab separated values
  target_seq: #  Target Protein Sequence 
    type: File
    format: tsv # Tab separated values
  drug_goa:
    type: File
    format: tsv # Tab separated values
  drug_smiles: #  Drug SMILES 
    type: File
    format: tsv # Tab separated values
  drug_se:  # Drug Side effects
    type: File
    format: tsv # Tab separated values
  human_ppi:  # Drug Side effects
    type: File
    format: tsv # Tab separated values
  gene_ontoloy:  # Drug Side effects
    type: File
    format: owl # Tab separated values
  human_phenotpe_ontology:  # Drug Side effects
    type: File
    format: owl # Tab separated values
  ind_gold_std:
    type: File
    format: tsv # Tab separated values


outputs:
   results:
     type: File
   predictions:
     type: File

steps:
  drug-target-sequence-similarity:
    run: drug-target-sequence-similarity.cwl
    in:
        drug_target_input: drug_target
        target_seq_input: target_seq
    out:
        drugs-target-seq-sim: drugs-target-seq-sim.txt
     
  drug-sideeffect-similarity:
    run: drug-sideeffect-similarity.cwl
    in:
        drug_se_input: drug_se
    out:
        drugs-se-sim: drugs-se-sim.txt
        
  drug-ppi-similarity:
    run: drug-ppi-similarity.cwl
    in:
        drug_target_input: drug_target
        human_ppi: human_ppi
    out:
        drugs-ppi-sim: drugs-ppi-sim.txt
        
  drug-chemical-similarity:
    run: drug-chemical-similarity.cwl
    in:
        drug_smiles_input: drug_smiles
    out:
        drugs-chem-sim: drugbank-drug-smiles-trimmed.csv
        
  drug-go-similarity:
    run: drug-go-similarity.cwl
    in:
        drug_go_input: drug_goa
        go_file : gene_ontoloy
    out:
        drugs-go-sim: gene.go.sim.out

  disease-mesh-similarity:
    run: disease-mesh-similarity.cwl
    in:
        disease_mesh_annotation: disease_mesh_annotation
        ind_gold_file : ind_gold_file
    out:
        disease-mesh-sim: diseases-pheno-sim.txt
        
  disease-hpo-similarity:
    run: disease-mesh-similarity.cwl
    in:
        disease_hpo_annotation: disease_hpo_annotation
        hpo_file : human_phenotpe_ontology
    out:
        disease-hpo-sim: hpo.sim.out
        
   concatenate-features:
           run: concatenate-features.cwl
        in: 
            disease-mesh-sim: disease-mesh-sim
            disease-hpo-sim: disease-hpo-sim
            drugs-go-sim: gdrugs-go-sim
            drugs-chem-sim: drugs-go-sim
            drugs-ppi-sim: drugs-ppi-sim
            drugs-se-sim: drugs-se-sim
            drugs-target-seq-sim: drugs-target-seq-sim
        out:
            drugs-df : drugs-df
            diseases-df : diseases-df
            
        
   combined-feature:
        run: combined-feature-calculation.cwl
        in: 
            known-drug-disease: ind_gold_file
            drugs-df : drugs-df
            diseases-df : diseases-df
        out:
            combined-scores : combined-scores
            
    cross-validation:
        run: cross-validation.cwl
        in:
            combined-scores : combined-scores
            ind_gold_file : ind_gold_file
            
        out:
            results : results
            
    make-predictions:
        run: make-predictions.cwl
        in:
            combined-scores : combined-scores
            ind_gold_file : ind_gold_file
            
        out:
            predictions : predictions        
            

