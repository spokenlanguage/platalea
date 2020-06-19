SCRIPTPATH=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPTPATH")

for exp in runs/basic-default-?-* runs/mtl-asr-ds001-* runs/basic-default-jp-?-* runs/mtl-asr-jp-ds001-*; do
    cd $exp
    python $SCRIPTDIR/../../utils/evaluate_net.py -t net.best.pt >result_test.json
    cd - > /dev/null
done

for exp in runs/pip-seq-ds001-* runs/pip-seq-jp-ds001-*; do
    cd $exp
    python $SCRIPTDIR/../../utils/evaluate_pipeline.py -t . >result_test.json
    cd - > /dev/null
done
