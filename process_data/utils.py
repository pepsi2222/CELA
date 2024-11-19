import os
import json
import pandas as pd
from collections import defaultdict



def get_data_from_json_by_line(json_file_path, fields):
    data = defaultdict(list)
    with open(json_file_path, 'r') as rf:
        while True:
            datum = rf.readline()
            if not datum:
                break

            try:
                datum = eval(datum)
            except Exception as e:
                print(e, datum)
                
            if not set(fields).issubset(set(datum.keys())):
                continue
            for f in fields:
                data[f].append(datum[f])
    df = pd.DataFrame(data)
    return df




