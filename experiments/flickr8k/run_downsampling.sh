#!/bin/bash
source run.sh

# Set default values
DOWNSAMPLING_FACTORS="1 3 9 27 81 243"
REPLIDS="a b c"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_downsampling asr asr
replicate basic-default retrieval
run_downsampling text-image retrieval
run_downsampling_text mtl-asr mtl
run_downsampling_text mtl-st mtl
run_downsampling pip-ind
run_downsampling_text pip-seq
echo "Finished."
