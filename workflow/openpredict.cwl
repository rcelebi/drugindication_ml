#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: Workflow

inputs:
  detect_json: 
      type: File
  directory_in:
     type: Directory
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
  disease_mesh_annotation:
    type: File
    format: tsv # Tab separated values
  disease_hpo_annotation:
    type: File
    format: tsv # Tab separated values



outputs:
   #results:
   #  type: File
   #predictions:
   #  type: File
   results:
     type: File
   predictions:
     type: File


steps:
  drug-ppi-similarity:
    run: tools/drug-ppi-similarity.cwl
    in:
      drug_target_input: drug_target
      human_ppi_input: human_ppi
    out:
      - drugs_ppi_sim
             
  drug-target-sequence-similarity:
    run: tools/drug-target-sequence-similarity.cwl
    in:
      drug_target_input: drug_target
      target_seq_input: target_seq
    out:
      - drugs_target_seq_sim
             
  drug-go-similarity:
    run: tools/drug-go-similarity.cwl
    in:
      drug_go_input: drug_goa
      go_input : gene_ontoloy
    out:
      - drugs_go_sim             
             
  drug-chemical-similarity:
    run: tools/drug-chemical-similarity.cwl
    in:
      drug_smiles_input: drug_smiles
    out:
      - drugs_chem_sim
             
  drug-sideeffect-similarity:
    run: tools/drug-sideeffect-similarity.cwl
    in:
      drug_se_input: drug_se
    out:
      - drugs_se_sim 
            
  disease-mesh-similarity:
    run: tools/disease-mesh-similarity.cwl
    in:
      mesh_annotation: disease_mesh_annotation
      gold_file : ind_gold_std
    out:
      -  disease_mesh_sim   
            
  disease-hpo-similarity:
    run: tools/disease-hpo-similarity.cwl
    in:
      hpo_annotation: disease_hpo_annotation
      hpo_input : human_phenotpe_ontology
    out:
      -  disease_hpo_sim     
            
  calculate-combined-features:
    run: tools/calculate-combined-features.cwl
    in:
      known_drug_disease : ind_gold_std
      drugs_chem_sim_input: drug-chemical-similarity/drugs_chem_sim
      drugs_se_sim_input : drug-sideeffect-similarity/drugs_se_sim
      drugs_go_sim_input: drug-go-similarity/drugs_go_sim
      drugs_ppi_sim_input: drug-ppi-similarity/drugs_ppi_sim
      drugs_target_seq_sim_input: drug-target-sequence-similarity/drugs_target_seq_sim
      disease_mesh_sim_input: disease-mesh-similarity/disease_mesh_sim
      disease_hpo_sim_input: disease-hpo-similarity/disease_hpo_sim
    out: 
      - combined_scores
          
  cross-validation:
    run: tools/cross-validation.cwl
    in:
      combined_scores_df : calculate-combined-features/combined_scores
      gold_file : ind_gold_std
    out: 
      - results
      - predictions

      
    
