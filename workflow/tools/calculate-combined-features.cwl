cwlVersion: cwl:v1.0
class: CommandLineTool

cwl:requirements:
  - class: DockerRequirement
    dockerPull: nlescsherlockdl/cropper

baseCommand: [python, /scripts/crop.py]
arguments: [--workflow_out, cropped.json, --cropped_folder, cropped]
inputs:
  known_drug_disease:
    type: File
    inputBinding:
      prefix: --json_input_file
      position: 1
  drugs_target_seq_sim_input:
    type: File
    inputBinding:
      prefix: --json_input_file
      position: 1
  drugs_go_sim_input:
    type: File
    inputBinding:
      prefix: --input_directory
  drugs_ppi_sim_input:
    type: File
    inputBinding:
      prefix: --input_directory
      position: 2
  drugs_chem_sim_input:
    type: File
    inputBinding:
      prefix: --input_directory
  drugs_se_sim_input:
    type: File
    inputBinding:
      prefix: --input_directory
      position: 2
  disease_hpo_sim_input:
    type: File
    inputBinding:
      prefix: --input_directory
  disease_mesh_sim_input:
    type: File
    inputBinding:
      prefix: --input_directory
      position: 2

outputs:                                                                                                                                 
  combined_scores:
    type: File
    outputBinding:
      glob: cropped.json
  predictions:
    type: File
    outputBinding:
      glob: cropped
