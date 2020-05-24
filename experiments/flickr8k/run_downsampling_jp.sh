#!/bin/bash
source run_jp.sh

# Set default values
DOWNSAMPLING_FACTORS="1 3 9 27 81 243"
REPLIDS="a b c"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_downsampling asr slt "-jp"
replicate basic-default retrieval "-jp"
run_downsampling text-image retrieval "-jp"
run_downsampling_text mtl-asr mtl "-jp"
run_downsampling_text mtl-st mtl "-jp"
run_downsampling pip-ind "" "-jp"
run_downsampling_text pip-seq "" "-jp"
echo "Finished."
