#!/usr/bin/bash

DOWNSAMPLING_FACTORS="1 3 9 27 81 243"
REPLIDS="a b c"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPTPATH=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPTPATH")

run_experiment() {
    expdir=$1
    cd $expdir
    git rev-parse HEAD >commit.txt
    python run.py >run.log 2>&1
    cd - >/dev/null
}

clean() {
    expdir=$1
    exptype=$2
    if [ "$exptype" != "" ]; then
        python $SCRIPTDIR/../../utils/copy_best.py --experiment_type $exptype $expdir
    fi
    rm $expdir/net.?.pt $expdir/net.??.pt
}

run_downsampling() {
    expname=$1
    exptype=$2
    for df in $DOWNSAMPLING_FACTORS; do
        dftag=ds$(printf %03d $df)
        for rid in $REPLIDS; do
            expdir=runs/$expname-$dftag-$rid-$TIMESTAMP
            mkdir -p $expdir
            cp $expname/run.py $expdir
            cp conf/seed-$rid.yml $expdir/config.yml
            echo -e "downsampling_factor\t$df" >> $expdir/config.yml
            if [[ $expname == pip* ]]; then
                path_asr=$(ls -d runs/asr-$dftag-$rid-* | xargs basename)
                echo -e "asr_model_dir\t../$path_asr" >> $expdir/config.yml
                if [ $expname == pip-ind ]; then
                     path_ti=$(ls -d runs/text-image-$dftag-$rid-* | xargs basename)
                     echo -e "text_image_model_dir\t../$path_ti" >> $expdir/config.yml
                fi
            fi
            run_experiment $expdir
            clean $expdir $exptype
        done
    done
}

run_downsampling_jp() {
    expname=$1
    exptype=$2
    for df in $DOWNSAMPLING_FACTORS; do
        dftag=ds$(printf %03d $df)
        for rid in $REPLIDS; do
            expdir=runs/$expname-jp-$dftag-$rid-$TIMESTAMP
            mkdir -p $expdir
            cp $expname/run.py $expdir
            cat conf/jp_human.yml conf/seed-$rid.yml > $expdir/config.yml
            echo -e "downsampling_factor\t$df" >> $expdir/config.yml
            if [[ $expname == pip* ]]; then
                path_asr=$(ls -d runs/asr-jp-$dftag-$rid-* | xargs basename)
                echo -e "asr_model_dir\t../$path_asr" >> $expdir/config.yml
                if [ $expname == pip-ind ]; then
                     path_ti=$(ls -d runs/text-image-jp-$dftag-$rid-* | xargs basename)
                     echo -e "text_image_model_dir\t../$path_ti" >> $expdir/config.yml
                fi
            fi
            run_experiment $expdir
            clean $expdir $exptype
        done
    done
}

run_downsampling_text() {
    expname=$1
    exptype=$2
    for df in $DOWNSAMPLING_FACTORS; do
        dftag=ds$(printf %03d $df)
        for rid in $REPLIDS; do
            expdir=runs/$expname-$dftag-$rid-$TIMESTAMP
            mkdir -p $expdir
            cp $expname/run.py $expdir
            cp conf/seed-$rid.yml $expdir/config.yml
            echo -e "downsampling_factor_text\t$df" >> $expdir/config.yml
            if [[ $expname == pip* ]]; then
                path_asr=$(ls -d runs/asr-$dftag-$rid-* | xargs basename)
                echo -e "asr_model_dir\t../$path_asr" >> $expdir/config.yml
                if [ $expname == pip-ind ]; then
                     path_ti=$(ls -d runs/text-image-$dftag-$rid-* | xargs basename)
                     echo -e "text_image_model_dir\t../$path_ti" >> $expdir/config.yml
                fi
            fi
            run_experiment $expdir
            clean $expdir $exptype
        done
    done
}

run_downsampling_text_jp() {
    expname=$1
    exptype=$2
    for df in $DOWNSAMPLING_FACTORS; do
        dftag=ds$(printf %03d $df)
        for rid in $REPLIDS; do
            expdir=runs/$expname-jp-$dftag-$rid-$TIMESTAMP
            mkdir -p $expdir
            cp $expname/run.py $expdir
            cat conf/jp_human.yml conf/seed-$rid.yml > $expdir/config.yml
            echo -e "downsampling_factor_text\t$df" >> $expdir/config.yml
            if [[ $expname == pip* ]]; then
                path_asr=$(ls -d runs/asr-jp-$dftag-$rid-* | xargs basename)
                echo -e "asr_model_dir\t../$path_asr" >> $expdir/config.yml
                if [ $expname == pip-ind ]; then
                     path_ti=$(ls -d runs/text-image-jp-$dftag-$rid-* | xargs basename)
                     echo -e "text_image_model_dir\t../$path_ti" >> $expdir/config.yml
                fi
            fi
            run_experiment $expdir
            clean $expdir $exptype
        done
    done
}

replicate() {
    expname=$1
    exptype=$2
    tag=$3
    [ "$tag" != "" ] && tag=-$tag
    for rid in $REPLIDS; do
        expdir=runs/$expname$tag-$rid-$TIMESTAMP
        mkdir -p $expdir
        cp $expname/run.py $expdir
        cp conf/seed-$rid.yml $expdir/config.yml
        if [ "$tag" == "-comp" ]; then
            echo -e "flickr8k_meta\tdataset_multilingual_human_only.json" >> $expdir/config.yml
        fi
        run_experiment $expdir
        clean $expdir $exptype
    done
}

rerun() {
    expname=$1
    exptype=$2
    tag=$3
    [ "$tag" != "" ] && tag=-$tag
    for df in $DOWNSAMPLING_FACTORS; do
        dftag=ds$(printf %03d $df)
        for rid in $REPLIDS; do
            expdir=$(ls -d runs/$expname$tag-$dftag-$rid-* | xargs basename)
            expdir=runs/$expdir
            run_experiment $expdir
            clean $expdir $exptype
        done
    done
}

# Experiments with transcriptions
run_downsampling asr asr
replicate basic-default retrieval
run_downsampling text-image retrieval
run_downsampling_text mtl-asr mtl
run_downsampling_text mtl-st mtl
run_downsampling pip-ind
run_downsampling_text pip-seq

# Experiments with translations
run_downsampling_jp asr slt
replicate basic-default retrieval jp
run_downsampling_jp text-image retrieval
run_downsampling_text_jp mtl-asr mtl
run_downsampling_text_jp mtl-st mtl
run_downsampling_jp pip-ind
run_downsampling_text_jp pip-seq

# Experiments with transcriptions, matching size of Japanese dataset
replicate asr asr comp
replicate basic-default retrieval comp
replicate text-image retrieval comp
replicate mtl-asr mtl comp
replicate mtl-st mtl comp
replicate pip-ind "" comp
replicate pip-seq "" comp

echo "Finished."
