cwlVersion: cwl:v1.0
class: CommandLineTool

cwl:requirements:
  - class: DockerRequirement
    dockerPull: nlescsherlockdl/cropper

baseCommand: [python, /scripts/crop.py]
arguments: [--workflow_out, cropped.json, --cropped_folder, cropped]
inputs:
  drug_target_input:
    type: File
    inputBinding:
      prefix: --json_input_file
      position: 1
  target_seq_input:
    type: File
    inputBinding:
      prefix: --input_directory
      position: 2
  
outputs:                                                                                                                                 
  drugs_target_seq_sim:
    type: File
    outputBinding:
      glob: cropped.json

