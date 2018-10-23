cwlVersion: cwl:v1.0
class: CommandLineTool

cwl:requirements:
  - class: DockerRequirement
    dockerPull: nlescsherlockdl/cropper

baseCommand: [python, /scripts/crop.py]
arguments: [--workflow_out, cropped.json, --cropped_folder, cropped]
inputs:
  drug_se_input:
    type: File
    inputBinding:
      prefix: --json_input_file
      position: 1
  
outputs:                                                                                                                                 
  drugs_se_sim:
    type: File
    outputBinding:
      glob: cropped.json

