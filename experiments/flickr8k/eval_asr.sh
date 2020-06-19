SCRIPTPATH=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPTPATH")

for exp in runs/asr*-ds*; do
    cd $exp
    python $SCRIPTDIR/../../utils/evaluate_net.py -b net.best.pt >result_beam.json
    cd - > /dev/null
done
