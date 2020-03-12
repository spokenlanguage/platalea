import json
import glob
from shutil import copyfile
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def fix():
    paths = glob.glob("experiments/*/result.json")
    for path in paths:
        logging.info("Fixing {}".format(path))
        copyfile(path, path + ".orig")
        with open(path, 'w') as out:
            data = [ eval(line) for line in open(path + ".orig") ]
            for datum in data:
                print(json.dumps(datum), file=out)
                
