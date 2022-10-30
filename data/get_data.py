import os
import zipfile
import json

import numpy as np
import pandas as pd

def get_data(path):
    '''
    Get data from given zip and store as json
    :param
        path: Path of the zip file.
    :return:
        json file of data provided.
    '''
    path = os.getcwd() + path

    with zipfile.ZipFile(path, 'r') as z:
        # Extract all the contents of zip file in current directory
        z.extractall('./')

    file_name = z.namelist()[0]
    # storing as json file
    base = path.split(path + file_name)[0].split('.')[0]
    # naming new file
    os.rename(file_name, base + ".json")

    # tracking new file name
    new_file_name = base + ".json"

    return new_file_name

def get_databuffer(fname):
    '''
    Convert the json file provided to a dataframe.
    :param
        fname: json file
    :return:
        dataframe of the json file provided
    '''
    lines = []

    with open(fname, "r") as fp:
        for line in fp:
            obj = json.loads(line)
            # load(s) -> serialize
            # dump(s) -> deserialize
            lines.append(obj)

    df = pd.DataFrame(lines)

    # replacing empty lines with NA
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # Deleting intermediary file created
    os.remove(fname)

    return df

def main():
    datasets = '/data/datasets/'

    # get json
    fname = get_data(datasets + 'transactions.zip')

    # convert to dataframe
    df = get_databuffer(fname)

    # store as csv (for quick access)
    df.to_csv('.' + datasets + 'transactions.csv')

    # df = pd.read_csv('.' + wd + 'transactions.csv')
    df = df.drop(columns=df.columns[0], axis=1, inplace=True)

    return df