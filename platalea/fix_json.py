import json
import glob
from shutil import copyfile
import pandas as pd
import logging
import io

logging.basicConfig(level=logging.INFO)


def fix():
    paths = glob.glob("experiments/*/result.json")
    for path in paths:
        logging.info("Fixing {}".format(path))
        copyfile(path, path + ".orig")
        with open(path, 'w') as out:
            data = [eval(line) for line in open(path + ".orig")]
            for datum in data:
                print(json.dumps(datum), file=out)


def load_results():
    tables = []
    for file in glob.glob("experiments/vq*/result.json"):
        data = [flat(json.loads(line)) for line in open(file)]
        table = pd.read_json(io.StringIO(json.dumps(data)), orient='records')
        table['path'] = file
        tables.append(table)
    return tables


def flat(rec):
    return dict(epoch=rec['epoch'],
                medr=rec['medr'],
                recall1=rec['recall']['1'],
                recall5=rec['recall']['5'],
                recall10=rec['recall']['10'])
