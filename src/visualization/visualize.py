import json
import pandas as pd
import os

def add_metric_tocsv(csv_path, metric_path):
    with open(metric_path, 'r') as f:
        metrics = json.load(f)

    result = {}
    for k in metrics:
        v = metrics[k]
        if type(v) is float:
            result[k] = v
        else:
            result[k] = v[0]

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df = df.append(result, ignore_index=True)
    else:
        df = pd.DataFrame(result, index=[0])
    df.to_csv(csv_path, index=False)